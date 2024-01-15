"""
This is the Stable water isotope tracer components of ipycles, will activate when namelist['IsotopeTracer']=true
"""
#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport Microphysics
cimport ParallelMPI
cimport Lookup
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
cimport Microphysics_SB_Liquid
cimport Microphysics_Arctic_1M
cimport Microphysics_SB_SI
cimport Microphysics_SB_2M
cimport ThermodynamicsSA
cimport ThermodynamicsSB
cimport TimeStepping

from libc.math cimport log, exp
from NetCDFIO cimport NetCDFIO_Stats
import cython

cimport numpy as np
import numpy as np

include 'parameters.pxi'

cdef extern from "isotope.h":
    void statsIO_isotope_scaling_magnitude(Grid.DimStruct *dims, double *tmp_values) nogil
    void iso_equilibrium_fractionation_No_Microphysics(Grid.DimStruct *dims, double *t, 
        double *qt, double *qv_DV, double *ql_DV, double *qv_std, double *ql_std,
        double *qt_O18, double *qv_O18, double *ql_O18,
        double *qt_HDO, double *qv_HDO, double *ql_HDO) nogil

    void delta_isotopologue(Grid.DimStruct *dims, double *q_iso, double *q_std,
        double *delta, int index) nogil
    void compute_sedimentaion(Grid.DimStruct *dims, double *w_q, double *w_q_iso, double *w_q_std) nogil
    void tracer_constrain_NoMicro(Grid.DimStruct *dims, double *ql, 
        double*ql_std, double *ql_O18, double *qv_std, double *qv_O18, 
        double*qt_std, double *qt_O18) nogil
    void iso_mix_phase_fractionation(Grid.DimStruct *dims, Lookup.LookupStruct *LT, 
        double(*lam_fp)(double), double(*L_fp)(double, double),
        double *temperature, double* s, double *p0, 
        double *qt_std, double *qv_std, double *ql_std, double *qi_std, 
        double *qt_O18, double *qv_O18, double *ql_O18, double *qi_O18, 
        double *qt_HDO, double *qv_HDO, double *ql_HDO, double *qi_HDO) nogil
        
    void tracer_sb_liquid_microphysics_sources(Grid.DimStruct *dims, Lookup.LookupStruct *LT, 
        double (*lam_fp)(double), double (*L_fp)(double, double),
        double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
        double* density, double* p0, double* temperature,  double* qt, double ccn,
        double* ql, double* nr, double* qr, double dt, 
        double* nr_tendency_micro, double* qr_tendency_micro,
        double* nr_tendency, double* qr_tendency,
        double* qr_O18, double* qt_O18, double* qv_O18, double* ql_O18,
        double* qr_HDO, double* qt_HDO, double* qv_HDO, double* ql_HDO,
        double* qr_O18_tendency_micro, double* qr_O18_tendency, 
        double* qr_HDO_tendency_micro, double* qr_HDO_tendency) nogil
    
    void tracer_arctic1m_microphysics_sources(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double), 
        double (*L_fp)(double, double), double* density, double* p0, double ccn, double n0_ice, double dt,
        double* temperature, double* qt, double* qv, double* ql, double* qi, 
        double* qrain, double* nrain, double* qsnow, double* nsnow, double* ql_std, double* qi_std,
        double* qrain_tendency_micro, double* qrain_tendency,
        double* qsnow_tendency_micro, double* qsnow_tendency,
        double* precip_rate, double* evap_rate, double* melt_rate,
        double* qt_O18, double* qv_O18, double* ql_O18, double* qi_O18, 
        double* qrain_O18, double* qsnow_O18,
        double* qrain_O18_tendency, double* qrain_O18_tendency_micro, 
        double* qsnow_O18_tendency, double* qsnow_O18_tendency_micro,
        double* precip_iso_rate_O18, double* evap_iso_rate_O18, 
        double* qt_HDO, double* qv_HDO, double* ql_HDO, double* qi_HDO, 
        double* qrain_HDO, double* qsnow_HDO,
        double* qrain_HDO_tendency, double* qrain_HDO_tendency_micro, 
        double* qsnow_HDO_tendency, double* qsnow_HDO_tendency_micro,
        double* precip_iso_rate_HDO, double* evap_iso_rate_HDO) nogil

cdef extern from "scalar_advection.h":
    void compute_advective_fluxes_a(Grid.DimStruct *dims, double *rho0, double*rho0_half, 
        double *velocity, double *scalar, double* flux, int d, int scheme) nogil

cdef extern from "isotope_functions.h":
    double q_2_delta(double q_iso, double q_std) nogil

cdef extern from "microphysics.h":
    void microphysics_stokes_sedimentation_velocity(Grid.DimStruct *dims, double* density, double ccn, double*  ql, double*  qt_velocity) nogil

cdef extern from "microphysics_sb.h":
    double sb_rain_shape_parameter_0(double density, double qr, double Dm) nogil
    double sb_rain_shape_parameter_1(double density, double qr, double Dm) nogil
    double sb_rain_shape_parameter_2(double density, double qr, double Dm) nogil
    double sb_rain_shape_parameter_4(double density, double qr, double Dm) nogil
    double sb_droplet_nu_0(double density, double ql) nogil
    double sb_droplet_nu_1(double density, double ql) nogil
    double sb_droplet_nu_2(double density, double ql) nogil
    void sb_sedimentation_velocity_rain(Grid.DimStruct *dims, double (*rain_mu)(double,double,double),
        double* density, double* nr, double* qr, double* nr_velocity, double* qr_velocity) nogil
    void sb_sedimentation_velocity_liquid(Grid.DimStruct *dims, double*  density, double ccn, 
        double* ql, double* qt_velocity)nogil

cdef extern from "microphysics_sb_liquid.h":
    void sb_qt_source_formation(Grid.DimStruct *dims,double* qr_tendency, double* qt_tendency )nogil

cdef extern from "microphysics_arctic_1m.h":
    void sedimentation_velocity_rain(Grid.DimStruct *dims, double* density, double* nrain, double* qrain,
                                     double* qrain_velocity) nogil
    void sedimentation_velocity_snow(Grid.DimStruct *dims, double* density, double* nsnow, double* qsnow,
                                     double* qsnow_velocity) nogil
    void qt_source_formation(Grid.DimStruct *dims, double* qt_tendency, double* precip_rate, double* evap_rate) nogil
    
cdef extern from "microphysics_sb_si.h":
    void sb_si_microphysics_sources(Grid.DimStruct *dims,
            double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
            double* density, double* p0, double* temperature,  double* qt, double ccn,
            double* ql, double* nr, double* qr, double* qisi, double* nisi, double dt, 
            double* nr_tendency_micro, double* qr_tendency_micro, double* nr_tendency, double* qr_tendency, 
            double* nisi_tendency_micro, double* qisi_tendency_micro, double* nisi_tendency, double* qisi_tendency,
            double* precip_rate, double* evap_rate, double* melt_rate) nogil
    void sb_si_qt_source_formation(Grid.DimStruct *dims, double* qisi_tendency, double* qr_tendency, double* qt_tendency)nogil
    void sb_sedimentation_velocity_ice(Grid.DimStruct *dims, double* nisi, double* qisi, double* density, double* nisi_velocity, 
            double* qisi_velocity) nogil
    void sbsi_NI(Grid.DimStruct *dims, double* qt, double* p0, double* rho0_half, double* temperature, double* NI_Mayer,
            double* NI_Flecher, double* NI_Copper, double* NI_Phillips, double* NI_contact_Young, double* NI_contact_Mayer)

def IsotopeTracersFactory(namelist, LatentHeat LH, ParallelMPI.ParallelMPI Par):
    try:
        #  micro_scheme = namelist['microphysics']['scheme']
        iso_scheme = namelist['isotopetracers']['scheme']
        if iso_scheme == 'None_SA':
            return IsotopeTracers_NoMicrophysics(namelist)
        elif iso_scheme == 'SB_Liquid':
            return IsotopeTracers_SB_Liquid(namelist)
        elif iso_scheme == 'Arctic_1M':
            return IsotopeTracers_Arctic_1M(namelist, LH, Par)
        elif iso_scheme == "SB_SI":
            return IsotopeTracers_SBSI(namelist)
        elif iso_scheme == "SB_Ice":
            return IsotopeTracers_SB_Ice(namelist)
    except:
        return IsotopeTracersNone()

cdef class IsotopeTracersNone:
    def __init__(self):
        self.isotope_tracer = False
        return
    cpdef initialize(self, namelist, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        Pa.root_print('initialized with IsotopeNone')
        return
    cpdef update(self):
        return
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return

cdef class IsotopeTracers_NoMicrophysics:
    def __init__(self, namelist):
        self.isotope_tracer = True
        
        return
    cpdef initialize(self, namelist, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
                    DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        Pa.root_print('initialized with IsotopeTracer with No Microphysics')

        # Prognostic variable: standerd water std of qt, ql and qv, which are totally same as qt, ql and qv 
        PV.add_variable('qt_std', 'kg/kg','qt_std','Total water std specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_std', 'kg/kg','qv_std','Vapor water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_std', 'kg/kg','ql_std','Cloud liquid water std specific humidity','sym', 'scalar', Pa)
        
        # Prognostic variable: qt_O18, total water isotopic specific humidity, defined as the ratio of isotopic mass of H2O18 to moist air.
        PV.add_variable('qt_O18', 'kg/kg','qt_O18_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_O18', 'kg/kg','qv_O18_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_O18', 'kg/kg','ql_O18_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        
        # Prognostic variable: qt_HDO, total water isotopic specific humidity, defined as the ratio of isotopic mass of HDO to moist air.
        PV.add_variable('qt_HDO', 'kg/kg','qt_HDO_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_HDO', 'kg/kg','qv_HDO_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_HDO', 'kg/kg','ql_HDO_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)

        initialize_NS_base(NS, Gr, Pa)
        # finial output results after selection and scaling
        return
        
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref, 
        Microphysics.No_Microphysics_SA Micro_SA, ThermodynamicsSA.ThermodynamicsSA Th_sa, 
        DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t t_shift = DV.get_varshift(Gr,'temperature')
            Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')
            Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t s_shift = PV.get_varshift(Gr,'s')
            Py_ssize_t qt_std_shift = PV.get_varshift(Gr,'qt_std')
            Py_ssize_t qv_std_shift = PV.get_varshift(Gr,'qv_std')
            Py_ssize_t ql_std_shift = PV.get_varshift(Gr,'ql_std')
            Py_ssize_t qt_O18_shift = PV.get_varshift(Gr,'qt_O18')
            Py_ssize_t qv_O18_shift = PV.get_varshift(Gr,'qv_O18')
            Py_ssize_t ql_O18_shift = PV.get_varshift(Gr,'ql_O18')
            Py_ssize_t qt_HDO_shift = PV.get_varshift(Gr,'qt_HDO')
            Py_ssize_t qv_HDO_shift = PV.get_varshift(Gr,'qv_HDO')
            Py_ssize_t ql_HDO_shift = PV.get_varshift(Gr,'ql_HDO')
            double[:] qv_std_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] ql_std_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qv_O18_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] ql_O18_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

        iso_equilibrium_fractionation_No_Microphysics(&Gr.dims, &DV.values[t_shift],
            &PV.values[qt_shift], &DV.values[qv_shift], &DV.values[ql_shift], 
            &PV.values[qv_std_shift], &PV.values[ql_std_shift], 
            &PV.values[qt_O18_shift], &PV.values[qv_O18_shift], &PV.values[ql_O18_shift], 
            &PV.values[qt_HDO_shift], &PV.values[qv_HDO_shift], &PV.values[ql_HDO_shift])
        return

    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
            DiagnosticVariables.DiagnosticVariables DV, 
            ReferenceState.ReferenceState Ref, Microphysics.No_Microphysics_SA Micro_SA, 
            TimeStepping.TimeStepping TS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        iso_stats_io_Base(Gr, PV, DV, Ref, NS, Pa)
        return

cdef extern from "isotope.h":
    void sb_iso_rain_evaporation_wrapper(
        Grid.DimStruct *dims, Lookup.LookupStruct *LT, 
        double (*lam_fp)(double), double (*L_fp)(double, double),
        double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
        double* density, double* p0, double* temperature, double* qt, 
        double* qv, double* qr, double* nr, double* qv_O18, double* qr_O18,
        double* qv_HDO, double* qr_HDO, 
        double* dpfv, double* Dp, double* g_thermo,
        double* qr_tend, double* nr_tend, double* qr_O18_tend, double* qr_HD0_tend)nogil

cdef class IsotopeTracers_SB_Liquid:
    def __init__(self, namelist):
        self.isotope_tracer = True
        return
    cpdef initialize(self, namelist, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
                    DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        Pa.root_print('initialized with IsotopeTracer with SB_Liquid scheme')
        
        # Prognostic variable: q_iso, isotopic specific humidity of qt, qv, ql and qr, defined as the ratio of isotopic mass of H2O18 to moist air.
        PV.add_variable('qt_O18', 'kg/kg','qt_O18_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_O18', 'kg/kg','qv_O18_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_O18', 'kg/kg','ql_O18_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qr_O18', 'kg/kg','qr_O18_isotope','Rain droplets water isotopic specific humidity','sym', 'scalar', Pa)

        # Prognostic variable: q_iso, isotopic specific humidity of qt, qv, ql and qr, defined as the ratio of isotopic mass of HDO to moist air.
        PV.add_variable('qt_HDO', 'kg/kg','qt_HDO_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_HDO', 'kg/kg','qv_HDO_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_HDO', 'kg/kg','ql_HDO_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qr_HDO', 'kg/kg','qr_HDO_isotope','Rain droplets water isotopic specific humidity','sym', 'scalar', Pa)

        PV.add_variable('qt_std', 'kg/kg','qt_std','Total water std specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_std', 'kg/kg','qv_std','Vapor water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_std', 'kg/kg','ql_std','Cloud liquid water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qr_std', 'kg/kg','ql_std','Rain water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('nr_std', 'kg/kg','ql_std','Rain water std specific humidity','sym', 'scalar', Pa)
        
        # sedimentation velocity of qt_O18(w_qt_O18) and qr_O18(w_qr_O18), which should be same as qt and qr, as DVs w_qt, w_qr 
        try:
            self.cloud_sedimentation = namelist['microphysics']['cloud_sedimentation']
        except:
            self.cloud_sedimentation = False
        
        if self.cloud_sedimentation:
            DV.add_variables('w_qt_O18', 'm/s', r'w_{qt_O18}', 'cloud liquid water isotopic sedimentation velocity', 'sym', Pa)
            DV.add_variables('w_qt_HDO', 'm/s', r'w_{qt_HDO}', 'cloud liquid water isotopic sedimentation velocity', 'sym', Pa)
            DV.add_variables('w_qt_std', 'm/s', r'w_{qt_O18}', 'cloud liquid water std sedimentation velocity', 'sym', Pa)
            NS.add_profile('qt_std_sedimentation_flux', Gr, Pa, 'kg/kg', '', '')
        DV.add_variables('w_qr_O18', 'm/s', r'w_{qr_O18}', 'rain mass isotopic sedimentation veloctiy', 'sym', Pa)
        DV.add_variables('w_qr_HDO', 'm/s', r'w_{qr_HDO}', 'rain mass isotopic sedimentation veloctiy', 'sym', Pa)
        DV.add_variables('w_qr_std', 'm/s', r'w_{qr_O18}', 'rain std mass sedimentation veloctiy', 'sym', Pa)
        DV.add_variables('w_nr_std', 'm/s', r'w_{qr_O18}', 'rain std mass sedimentation veloctiy', 'sym', Pa)

        NS.add_profile('qr_std', Gr, Pa, 'kg/kg', '', 'stander water tarcer rain')
        NS.add_profile('qr_O18', Gr, Pa, 'kg/kg', '', 'Finial result of rain isotopic sepcific humidity of H2O18')
        NS.add_profile('qr_HDO', Gr, Pa, 'kg/kg', '', 'Finial result of rain isotopic sepcific humidity of HDO')

        NS.add_profile('qr_tend_evap', Gr, Pa, '','','')
        NS.add_profile('nr_tend_evap', Gr, Pa, '','','')
        NS.add_profile('qr_O18_tend_evap', Gr, Pa, '','','')
        NS.add_profile('qr_HDO_tend_evap', Gr, Pa, '','','')
        NS.add_profile('dpfv', Gr, Pa, '','','')
        NS.add_profile('Dp', Gr, Pa, '','','')
        NS.add_profile('g_thermo', Gr, Pa, '','','')

        initialize_NS_base(NS, Gr, Pa)
        return

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref, 
        Microphysics_SB_Liquid.Microphysics_SB_Liquid Micro_SB_Liquid, ThermodynamicsSA.ThermodynamicsSA Th_sa, 
        DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t t_shift      = DV.get_varshift(Gr,'temperature')
            Py_ssize_t qt_shift     = PV.get_varshift(Gr,'qt')
            Py_ssize_t qv_shift     = DV.get_varshift(Gr,'qv')
            Py_ssize_t ql_shift     = DV.get_varshift(Gr,'ql')
            Py_ssize_t qr_shift     = PV.get_varshift(Gr,'qr')
            Py_ssize_t nr_shift     = PV.get_varshift(Gr,'nr')
            Py_ssize_t s_shift      = PV.get_varshift(Gr,'s')
            Py_ssize_t alpha_shift  = DV.get_varshift(Gr, 'alpha')
            Py_ssize_t wqr_shift    = DV.get_varshift(Gr, 'w_qr')
            Py_ssize_t wnr_shift    = DV.get_varshift(Gr, 'w_nr')

            Py_ssize_t qt_std_shift  = PV.get_varshift(Gr,'qt_std')
            Py_ssize_t qv_std_shift  = PV.get_varshift(Gr,'qv_std')
            Py_ssize_t ql_std_shift  = PV.get_varshift(Gr,'ql_std')
            Py_ssize_t qr_std_shift  = PV.get_varshift(Gr,'qr_std')
            Py_ssize_t nr_std_shift  = PV.get_varshift(Gr,'nr_std')
            Py_ssize_t wqr_std_shift = DV.get_varshift(Gr, 'w_qr_std')

            Py_ssize_t qt_O18_shift  = PV.get_varshift(Gr,'qt_O18')
            Py_ssize_t qv_O18_shift  = PV.get_varshift(Gr,'qv_O18')
            Py_ssize_t ql_O18_shift  = PV.get_varshift(Gr,'ql_O18')
            Py_ssize_t qr_O18_shift  = PV.get_varshift(Gr, 'qr_O18')
            Py_ssize_t wqr_O18_shift = DV.get_varshift(Gr, 'w_qr_O18')

            Py_ssize_t qt_HDO_shift  = PV.get_varshift(Gr,'qt_HDO')
            Py_ssize_t qv_HDO_shift  = PV.get_varshift(Gr,'qv_HDO')
            Py_ssize_t ql_HDO_shift  = PV.get_varshift(Gr,'ql_HDO')
            Py_ssize_t qr_HDO_shift  = PV.get_varshift(Gr, 'qr_HDO')
            Py_ssize_t wqr_HDO_shift = DV.get_varshift(Gr, 'w_qr_HDO')
            Py_ssize_t wqt_std_shift
            Py_ssize_t wqt_std_O18_shift
            Py_ssize_t wqt_std_HDO_shift

            double[:] qv_std_tmp     = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] ql_std_tmp     = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qv_O18_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] ql_HDO_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

            double[:] qr_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] nr_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qr_O18_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qr_HDO_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

        iso_equilibrium_fractionation_No_Microphysics(&Gr.dims, &DV.values[t_shift],
            &PV.values[qt_shift], &DV.values[qv_shift], &DV.values[ql_shift], 
            &PV.values[qv_std_shift], &PV.values[ql_std_shift], 
            &PV.values[qt_O18_shift], &PV.values[qv_O18_shift], &PV.values[ql_O18_shift], 
            &PV.values[qt_HDO_shift], &PV.values[qv_HDO_shift], &PV.values[ql_HDO_shift])

        tracer_sb_liquid_microphysics_sources(&Gr.dims, &Micro_SB_Liquid.CC.LT.LookupStructC, 
            Micro_SB_Liquid.Lambda_fp, Micro_SB_Liquid.L_fp, 
            Micro_SB_Liquid.compute_rain_shape_parameter, Micro_SB_Liquid.compute_droplet_nu, 
            &Ref.rho0_half[0],  &Ref.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift], 
            Micro_SB_Liquid.ccn, &DV.values[ql_shift], &PV.values[nr_shift], &PV.values[qr_std_shift], TS.dt, 
            &nr_std_tend_micro[0], &qr_std_tend_micro[0], &PV.tendencies[nr_std_shift], &PV.tendencies[qr_std_shift],
            &PV.values[qr_O18_shift], &PV.values[qt_O18_shift], &PV.values[qv_O18_shift], &PV.values[ql_O18_shift],
            &PV.values[qr_HDO_shift], &PV.values[qt_HDO_shift], &PV.values[qv_HDO_shift], &PV.values[ql_HDO_shift],
            &qr_O18_tend_micro[0], &PV.tendencies[qr_O18_shift], &qr_HDO_tend_micro[0], &PV.tendencies[qr_HDO_shift])

        sb_sedimentation_velocity_rain(&Gr.dims, Micro_SB_Liquid.compute_rain_shape_parameter, 
            &Ref.rho0_half[0], &PV.values[nr_shift], &PV.values[qr_shift],
            &DV.values[wnr_shift], &DV.values[wqr_std_shift])
        sb_sedimentation_velocity_rain(&Gr.dims, Micro_SB_Liquid.compute_rain_shape_parameter, 
            &Ref.rho0_half[0], &PV.values[nr_shift], &PV.values[qr_shift],
            &DV.values[wnr_shift], &DV.values[wqr_O18_shift])
        sb_sedimentation_velocity_rain(&Gr.dims, Micro_SB_Liquid.compute_rain_shape_parameter, 
            &Ref.rho0_half[0], &PV.values[nr_shift], &PV.values[qr_shift],
            &DV.values[wnr_shift], &DV.values[wqr_HDO_shift])
        if self.cloud_sedimentation:
            wqt_std_shift = DV.get_varshift(Gr, 'w_qt_std')
            wqt_O18_shift = DV.get_varshift(Gr, 'w_qt_O18')
            wqt_HDO_shift = DV.get_varshift(Gr, 'w_qt_HDO')
            if self.stokes_sedimentation:
                microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_Liquid.ccn, &DV.values[ql_shift], &DV.values[wqt_std_shift])
                microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_Liquid.ccn, &DV.values[ql_shift], &DV.values[wqt_O18_shift])
                microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_Liquid.ccn, &DV.values[ql_shift], &DV.values[wqt_HDO_shift])
            else:
                sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_Liquid.ccn, &DV.values[ql_shift], &DV.values[wqt_std_shift])
                sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_Liquid.ccn, &DV.values[ql_shift], &DV.values[wqt_O18_shift])
                sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_Liquid.ccn, &DV.values[ql_shift], &DV.values[wqt_HDO_shift])
        sb_qt_source_formation(&Gr.dims,  &qr_std_tend_micro[0], &PV.tendencies[qt_std_shift])
        sb_qt_source_formation(&Gr.dims,  &qr_O18_tend_micro[0], &PV.tendencies[qt_O18_shift])
        sb_qt_source_formation(&Gr.dims,  &qr_HDO_tend_micro[0], &PV.tendencies[qt_HDO_shift])

        return 

    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
            DiagnosticVariables.DiagnosticVariables DV, ReferenceState.ReferenceState Ref, 
            Microphysics_SB_Liquid.Microphysics_SB_Liquid Micro_SB_Liquid,
            TimeStepping.TimeStepping TS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t t_shift      = DV.get_varshift(Gr,'temperature')
            Py_ssize_t qt_std_shift  = PV.get_varshift(Gr,'qt_std')
            Py_ssize_t qv_std_shift  = PV.get_varshift(Gr,'qv_std')
            Py_ssize_t ql_std_shift  = PV.get_varshift(Gr,'ql_std')
            Py_ssize_t qr_std_shift  = PV.get_varshift(Gr,'qr_std')
            Py_ssize_t nr_std_shift  = PV.get_varshift(Gr,'nr_std')

            Py_ssize_t qt_O18_shift  = PV.get_varshift(Gr,'qt_O18')
            Py_ssize_t qv_O18_shift  = PV.get_varshift(Gr,'qv_O18')
            Py_ssize_t ql_O18_shift  = PV.get_varshift(Gr,'ql_O18')
            Py_ssize_t qr_O18_shift  = PV.get_varshift(Gr, 'qr_O18')

            Py_ssize_t qt_HDO_shift  = PV.get_varshift(Gr,'qt_HDO')
            Py_ssize_t qv_HDO_shift  = PV.get_varshift(Gr,'qv_HDO')
            Py_ssize_t ql_HDO_shift  = PV.get_varshift(Gr,'ql_HDO')
            Py_ssize_t qr_HDO_shift  = PV.get_varshift(Gr, 'qr_HDO')
            
            double[:] qr_tend_evap = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] nr_tend_evap = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qr_O18_tend_evap = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qr_HDO_tend_evap = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] tmp
            double[:] dpfv = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] Dp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] g_thermo = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

        sb_iso_rain_evaporation_wrapper(&Gr.dims, &Micro_SB_Liquid.CC.LT.LookupStructC, 
            Micro_SB_Liquid.Lambda_fp, Micro_SB_Liquid.L_fp, 
            Micro_SB_Liquid.compute_rain_shape_parameter, Micro_SB_Liquid.compute_droplet_nu, 
            &Ref.rho0_half[0],  &Ref.p0_half[0], &DV.values[t_shift], &PV.values[qt_std_shift], 
            &PV.values[qv_std_shift], &PV.values[qr_std_shift], &PV.values[nr_std_shift], 
            &PV.values[qv_O18_shift], &PV.values[qr_O18_shift], 
            &PV.values[qv_HDO_shift], &PV.values[qr_HDO_shift],
            &dpfv[0], &Dp[0], &g_thermo[0],
            &qr_tend_evap[0], &nr_tend_evap[0], &qr_O18_tend_evap[0], &qr_HDO_tend_evap[0])
        
        tmp = Pa.HorizontalMean(Gr, &qr_tend_evap[0])
        NS.write_profile('qr_tend_evap', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &nr_tend_evap[0])
        NS.write_profile('nr_tend_evap', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &qr_O18_tend_evap[0])
        NS.write_profile('qr_O18_tend_evap', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &qr_HDO_tend_evap[0])
        NS.write_profile('qr_HDO_tend_evap', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &g_thermo[0])
        NS.write_profile('g_thermo', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &dpfv[0])
        NS.write_profile('dpfv', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &Dp[0])
        NS.write_profile('Dp', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)


        iso_stats_io_Base(Gr, PV, DV, Ref, NS, Pa)
        return

cdef class IsotopeTracers_Arctic_1M:
    def __init__(self, dict namelist, LatentHeat LH, ParallelMPI.ParallelMPI Par):
        self.isotope_tracer = True
        return
    cpdef initialize(self, namelist, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV,
                DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        Pa.root_print('initialized with IsotopeTracers_Arctic_1M')
        
        PV.add_variable('qt_std', 'kg/kg','qt_std','Total water std specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_std', 'kg/kg','qv_std','Vapor water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_std', 'kg/kg','ql_std','Cloud liquid water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qi_std', 'kg/kg','qi_std','Cloud ice water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qrain_std', 'kg/kg','qrain_std','rain std water tracer specific humidity','sym', "scalar", Pa)
        PV.add_variable('qsnow_std', 'kg/kg','qsnow_std','snow std water tracer specific humidity','sym', "scalar", Pa)
        DV.add_variables('w_qrain_std', 'unit', r'w_qrain_std','declaration', 'sym', Pa)
        DV.add_variables('w_qsnow_std', 'unit', r'w_qsnow_std','declaration', 'sym', Pa)
        
        PV.add_variable('qt_O18', 'kg/kg','qt_O18_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_O18', 'kg/kg','qv_O18_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_O18', 'kg/kg','ql_O18_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qi_O18', 'kg/kg','qi_O18_isotope','Cloud ice water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qrain_O18', 'kg/kg','qrain_O18','rain iso water tracer specific humidity','sym', "scalar", Pa)
        PV.add_variable('qsnow_O18', 'kg/kg','qsnow_O18','declaration','sym', "scalar", Pa)
        DV.add_variables('w_qrain_O18', 'unit', r'w_qrain_O18','declaration', 'sym', Pa)
        DV.add_variables('w_qsnow_O18', 'unit', r'w_qsnow_O18','declaration', 'sym', Pa)

        PV.add_variable('qt_HDO', 'kg/kg','qt_HDO_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_HDO', 'kg/kg','qv_HDO_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_HDO', 'kg/kg','ql_HDO_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qi_HDO', 'kg/kg','qi_HDO_isotope','Cloud ice water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qrain_HDO', 'kg/kg','qrain_HDO','rain iso water tracer specific humidity','sym', "scalar", Pa)
        PV.add_variable('qsnow_HDO', 'kg/kg','qsnow_HDO','declaration','sym', "scalar", Pa)
        DV.add_variables('w_qrain_HDO', 'unit', r'w_qrain_HDO','declaration', 'sym', Pa)
        DV.add_variables('w_qsnow_HDO', 'unit', r'w_qsnow_HDO','declaration', 'sym', Pa)

        initialize_NS_base(NS, Gr, Pa)
        
        NS.add_profile('qi_std_cloud', Gr, Pa, 'unit', '', 'qi_std_cloud')
        NS.add_profile('qi_O18_cloud', Gr, Pa, 'unit', '', 'qi_O18_cloud')
        NS.add_profile('ql_std_cloud', Gr, Pa, 'unit', '', 'ql_std_cloud')
        NS.add_profile('ql_O18_cloud', Gr, Pa, 'unit', '', 'ql_O18_cloud')
        NS.add_profile('qrain_std_domain', Gr, Pa, 'kg/kg', '', 'qrain_std_domain')
        NS.add_profile('qrain_O18_domain', Gr, Pa, 'kg/kg', '', 'qrain_O18_domain')
        NS.add_profile('qsnow_std_domain', Gr, Pa, 'kg/kg', '', 'qsnow_std_domain')
        NS.add_profile('qsnow_O18_domain', Gr, Pa, 'kg/kg', '', 'qsnow_O18_domain')
        NS.add_profile('qrain_std_cloud_domain', Gr, Pa, 'unit', '', 'qrain_std_cloud_domain')
        NS.add_profile('qrain_O18_cloud_domain', Gr, Pa, 'unit', '', 'qrain_O18_cloud_domain')
        NS.add_profile('qsnow_std_cloud_domain', Gr, Pa, 'unit', '', 'qsnow_std_cloud_domain')
        NS.add_profile('qsnow_O18_cloud_domain', Gr, Pa, 'unit', '', 'qsnow_O18_cloud_domain')
        return

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref, 
        Microphysics_Arctic_1M.Microphysics_Arctic_1M Micro_Arctic_1M, ThermodynamicsSA.ThermodynamicsSA Th_sa, 
        DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t t_shift          = DV.get_varshift(Gr,'temperature')
            Py_ssize_t qt_shift         = PV.get_varshift(Gr,'qt')
            Py_ssize_t qv_shift         = DV.get_varshift(Gr,'qv')
            Py_ssize_t ql_shift         = DV.get_varshift(Gr,'ql')
            Py_ssize_t qi_shift         = DV.get_varshift(Gr,'qi')
            Py_ssize_t s_shift          = PV.get_varshift(Gr,'s')
            Py_ssize_t qt_std_shift     = PV.get_varshift(Gr,'qt_std')
            Py_ssize_t qv_std_shift     = PV.get_varshift(Gr,'qv_std')
            Py_ssize_t ql_std_shift     = PV.get_varshift(Gr,'ql_std')
            Py_ssize_t qi_std_shift     = PV.get_varshift(Gr,'qi_std')
            Py_ssize_t qrain_shift      = PV.get_varshift(Gr, 'qrain')
            Py_ssize_t qsnow_shift      = PV.get_varshift(Gr, 'qsnow')
            Py_ssize_t nrain_shift      = DV.get_varshift(Gr, 'nrain')
            Py_ssize_t nsnow_shift      = DV.get_varshift(Gr, 'nsnow')
            Py_ssize_t qrain_std_shift  = PV.get_varshift(Gr,'qrain_std')
            Py_ssize_t qsnow_std_shift  = PV.get_varshift(Gr,'qsnow_std')
            Py_ssize_t wqrain_std_shift = DV.get_varshift(Gr,'w_qrain_std')
            Py_ssize_t wqsnow_std_shift = DV.get_varshift(Gr,'w_qsnow_std')

            Py_ssize_t qt_O18_shift     = PV.get_varshift(Gr,'qt_O18')
            Py_ssize_t qv_O18_shift     = PV.get_varshift(Gr,'qv_O18')
            Py_ssize_t ql_O18_shift     = PV.get_varshift(Gr,'ql_O18')
            Py_ssize_t qi_O18_shift     = PV.get_varshift(Gr,'qi_O18')
            Py_ssize_t qsnow_O18_shift  = PV.get_varshift(Gr,'qsnow_O18')
            Py_ssize_t qrain_O18_shift  = PV.get_varshift(Gr,'qrain_O18')
            Py_ssize_t wqrain_O18_shift = DV.get_varshift(Gr,'w_qrain_O18')
            Py_ssize_t wqsnow_O18_shift = DV.get_varshift(Gr,'w_qsnow_O18')

            Py_ssize_t qt_HDO_shift     = PV.get_varshift(Gr,'qt_HDO')
            Py_ssize_t qv_HDO_shift     = PV.get_varshift(Gr,'qv_HDO')
            Py_ssize_t ql_HDO_shift     = PV.get_varshift(Gr,'ql_HDO')
            Py_ssize_t qi_HDO_shift     = PV.get_varshift(Gr,'qi_HDO')
            Py_ssize_t qsnow_HDO_shift  = PV.get_varshift(Gr,'qsnow_HDO')
            Py_ssize_t qrain_HDO_shift  = PV.get_varshift(Gr,'qrain_HDO')
            Py_ssize_t wqrain_HDO_shift = DV.get_varshift(Gr,'w_qrain_HDO')
            Py_ssize_t wqsnow_HDO_shift = DV.get_varshift(Gr,'w_qsnow_HDO')
            
            double [:] qrain_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] qsnow_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] precip_rate_std = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] evap_rate_std = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] melt_rate_std = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

            double [:] qrain_O18_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] qsnow_O18_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] precip_rate_O18 = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] evap_rate_O18 = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

            double [:] qrain_HDO_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] qsnow_HDO_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] precip_rate_HDO = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] evap_rate_HDO = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

        iso_mix_phase_fractionation(&Gr.dims, &Micro_Arctic_1M.CC.LT.LookupStructC, 
            Micro_Arctic_1M.Lambda_fp, Micro_Arctic_1M.L_fp, &DV.values[t_shift], &PV.values[s_shift], &Ref.p0_half[0],
            &PV.values[qt_std_shift], &PV.values[qv_std_shift], &PV.values[ql_std_shift], &PV.values[qi_std_shift], 
            &PV.values[qt_O18_shift], &PV.values[qv_O18_shift], &PV.values[ql_O18_shift], &PV.values[qi_O18_shift], 
            &PV.values[qt_HDO_shift], &PV.values[qv_HDO_shift], &PV.values[ql_HDO_shift], &PV.values[qi_HDO_shift])

        tracer_arctic1m_microphysics_sources(&Gr.dims, &Micro_Arctic_1M.CC.LT.LookupStructC, Micro_Arctic_1M.Lambda_fp, 
            Micro_Arctic_1M.L_fp, &Ref.rho0_half[0],&Ref.p0_half[0], Micro_Arctic_1M.ccn, Micro_Arctic_1M.n0_ice_input, TS.dt,
            &DV.values[t_shift], &PV.values[qt_shift], &DV.values[qv_shift], &DV.values[ql_shift], &DV.values[qi_shift], 
            &PV.values[qrain_std_shift], &DV.values[nrain_shift], &PV.values[qsnow_std_shift], &DV.values[nsnow_shift], 
            &PV.values[ql_std_shift], &PV.values[qi_std_shift], &qrain_std_tend_micro[0], &PV.tendencies[qrain_std_shift],
            &qsnow_std_tend_micro[0], &PV.tendencies[qsnow_std_shift], &precip_rate_std[0], &evap_rate_std[0],&melt_rate_std[0],
            &PV.values[qt_O18_shift], &PV.values[qv_O18_shift], &PV.values[ql_O18_shift], &PV.values[qi_O18_shift], 
            &PV.values[qrain_O18_shift], &PV.values[qsnow_O18_shift], &PV.tendencies[qrain_O18_shift], &qrain_O18_tend_micro[0], 
            &PV.tendencies[qsnow_O18_shift], &qsnow_O18_tend_micro[0], &precip_rate_O18[0], &evap_rate_O18[0],
            &PV.values[qt_HDO_shift], &PV.values[qv_HDO_shift], &PV.values[ql_HDO_shift], &PV.values[qi_HDO_shift], 
            &PV.values[qrain_HDO_shift], &PV.values[qsnow_HDO_shift], &PV.tendencies[qrain_HDO_shift], &qrain_HDO_tend_micro[0], 
            &PV.tendencies[qsnow_HDO_shift], &qsnow_HDO_tend_micro[0], &precip_rate_HDO[0], &evap_rate_HDO[0])

        sedimentation_velocity_rain(&Gr.dims, &Ref.rho0_half[0], &DV.values[nrain_shift], &PV.values[qrain_shift],
            &DV.values[wqrain_std_shift])
        sedimentation_velocity_snow(&Gr.dims, &Ref.rho0_half[0], &DV.values[nsnow_shift], &PV.values[qsnow_shift],
            &DV.values[wqsnow_std_shift])

        sedimentation_velocity_rain(&Gr.dims, &Ref.rho0_half[0], &DV.values[nrain_shift], &PV.values[qrain_shift],
            &DV.values[wqrain_O18_shift])
        sedimentation_velocity_snow(&Gr.dims, &Ref.rho0_half[0], &DV.values[nsnow_shift], &PV.values[qsnow_shift],
            &DV.values[wqsnow_O18_shift])
        sedimentation_velocity_rain(&Gr.dims, &Ref.rho0_half[0], &DV.values[nrain_shift], &PV.values[qrain_shift],
            &DV.values[wqrain_HDO_shift])
        sedimentation_velocity_snow(&Gr.dims, &Ref.rho0_half[0], &DV.values[nsnow_shift], &PV.values[qsnow_shift],
            &DV.values[wqsnow_HDO_shift])

        qt_source_formation(&Gr.dims, &PV.tendencies[qt_std_shift], &precip_rate_std[0], &evap_rate_std[0])
        qt_source_formation(&Gr.dims, &PV.tendencies[qt_O18_shift], &precip_rate_O18[0], &evap_rate_O18[0])
        qt_source_formation(&Gr.dims, &PV.tendencies[qt_HDO_shift], &precip_rate_HDO[0], &evap_rate_HDO[0])

        return

    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
                DiagnosticVariables.DiagnosticVariables DV, ReferenceState.ReferenceState Ref, 
                Microphysics_Arctic_1M.Microphysics_Arctic_1M Micro_Arctic_1M, 
                TimeStepping.TimeStepping TS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        iso_stats_io_Base(Gr, PV, DV, Ref, NS, Pa)
        cdef:
            Py_ssize_t i,j,k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t imin    = Gr.dims.gw
            Py_ssize_t jmin    = Gr.dims.gw
            Py_ssize_t kmin    = Gr.dims.gw
            Py_ssize_t imax    = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax    = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax    = Gr.dims.nlg[2] - Gr.dims.gw
            double [:] tmp
            Py_ssize_t qi_shift         = DV.get_varshift(Gr, 'qi')
            Py_ssize_t ql_shift         = DV.get_varshift(Gr,'ql')
            Py_ssize_t qi_std_shift     = PV.get_varshift(Gr,'qi_std')
            Py_ssize_t qi_O18_shift     = PV.get_varshift(Gr,'qi_O18')
            Py_ssize_t ql_std_shift     = PV.get_varshift(Gr,'ql_std')
            Py_ssize_t ql_O18_shift     = PV.get_varshift(Gr,'ql_O18')
            
            Py_ssize_t qrain_std_shift  = PV.get_varshift(Gr,'qrain_std')
            Py_ssize_t qrain_O18_shift  = PV.get_varshift(Gr,'qrain_O18')
            Py_ssize_t qsnow_std_shift  = PV.get_varshift(Gr,'qsnow_std')
            Py_ssize_t qsnow_O18_shift  = PV.get_varshift(Gr,'qsnow_O18')
            
            double[:] delta_qi          = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] cloud_liquid_mask = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] cloud_ice_mask    = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] rain_mask         = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] snow_mask         = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            
        with nogil:
            with cython.boundscheck(False):
                for i in range(imin,imax):
                    ishift = i * istride
                    for j in range(jmin,jmax):
                        jshift = j * jstride
                        for k in range(kmin,kmax):
                            ijk = ishift + jshift + k

                            # define ice domain mask
                            if DV.values[qi_shift + ijk] > 0.0:
                                cloud_ice_mask[ijk] = 1.0

                            if DV.values[ql_shift + ijk] > 0.0:
                                cloud_liquid_mask[ijk] = 1.0

                            # define rain domain mask
                            if PV.values[qrain_std_shift + ijk] > 1.0e-10:
                                rain_mask[ijk] = 1.0
                            
                            # define snow domain mask
                            if PV.values[qsnow_std_shift + ijk] > 1.0e-10:
                                snow_mask[ijk] = 1.0

        # liquid cloud domain stats_io fo ql_std, and ql_O18_shift
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[ql_std_shift], &cloud_liquid_mask[0])
        NS.write_profile('ql_std_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[ql_O18_shift], &cloud_liquid_mask[0])
        NS.write_profile('ql_O18_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qi_std_shift], &cloud_ice_mask[0])
        NS.write_profile('qi_std_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qi_O18_shift], &cloud_ice_mask[0])
        NS.write_profile('qi_O18_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        # rain domain stats_io of qrain_std, and qrain_O18
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qrain_std_shift], &rain_mask[0])
        NS.write_profile('qrain_std_domain', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qrain_O18_shift], &rain_mask[0])
        NS.write_profile('qrain_O18_domain', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        
        # snow domain stats_io of qsnow_std, and qsnow_O18
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qsnow_std_shift], &snow_mask[0])
        NS.write_profile('qsnow_std_domain', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qsnow_O18_shift], &snow_mask[0])
        NS.write_profile('qsnow_O18_domain', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        # liquid cloud domain stats_io fo qrain_std, and qrain_O18_shift
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qrain_std_shift], &cloud_liquid_mask[0])
        NS.write_profile('qrain_std_cloud_domain', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qrain_O18_shift], &cloud_liquid_mask[0])
        NS.write_profile('qrain_O18_cloud_domain', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        
        # ice cloud domain stats_io fo qsnow_std, and qsnow_O18_shift
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qsnow_std_shift], &cloud_ice_mask[0])
        NS.write_profile('qsnow_std_cloud_domain', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qsnow_O18_shift], &cloud_ice_mask[0])
        NS.write_profile('qsnow_O18_cloud_domain', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        
        return

cdef class IsotopeTracers_SBSI:
    def __init__(self, namelist):

        self.isotope_tracer = True

        self.ccn = 100.0e6
        if namelist['meta']['casename'] == 'DYCOMS_RF02':
            self.ccn = 55.0e6
        elif namelist['meta']['casename'] == 'Rico':
            self.ccn = 70.0e6
        try:
            self.ccn = namelist['microphysics']['ccn']
        except:
            pass
        # Set option for calculation of mu (distribution shape parameter)
        try:
            mu_opt = namelist['microphysics']['SB_Liquid']['mu_rain']
            if mu_opt == 1:
                self.compute_rain_shape_parameter = sb_rain_shape_parameter_1
            elif mu_opt == 2:
                self.compute_rain_shape_parameter = sb_rain_shape_parameter_2
            elif mu_opt == 4:
                self.compute_rain_shape_parameter = sb_rain_shape_parameter_4
            elif mu_opt == 0:
                self.compute_rain_shape_parameter  = sb_rain_shape_parameter_0
            else:
                # Par.root_print("SB_Liquid mu_rain option not recognized, defaulting to option 1")
                self.compute_rain_shape_parameter = sb_rain_shape_parameter_1
        except:
            # Par.root_print("SB_Liquid mu_rain option not selected, defaulting to option 1")
            self.compute_rain_shape_parameter = sb_rain_shape_parameter_1
        # Set option for calculation of nu parameter of droplet distribution
        try:
            nu_opt = namelist['microphysics']['SB_Liquid']['nu_droplet']
            if nu_opt == 0:
                self.compute_droplet_nu = sb_droplet_nu_0
            elif nu_opt == 1:
                self.compute_droplet_nu = sb_droplet_nu_1
            elif nu_opt ==2:
                self.compute_droplet_nu = sb_droplet_nu_2
            else:
                # Par.root_print("SB_Liquid nu_droplet_option not recognized, defaulting to option 0")
                self.compute_droplet_nu = sb_droplet_nu_0
        except:
            # Par.root_print("SB_Liquid nu_droplet_option not selected, defaulting to option 0")
            self.compute_droplet_nu = sb_droplet_nu_0

        try:
            self.order = namelist['scalar_transport']['order_sedimentation']
        except:
            self.order = namelist['scalar_transport']['order']

        try:
            self.cloud_sedimentation = namelist['microphysics']['cloud_sedimentation']
        except:
            self.cloud_sedimentation = False
        if namelist['meta']['casename'] == 'DYCOMS_RF02':
            self.stokes_sedimentation = True
        else:
            self.stokes_sedimentation = False

        return
    cpdef initialize(self, namelist, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
                    DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        Pa.root_print('initialized with IsotopeTracer with SBSI scheme')
        
        # Prognostic variable: q_iso, isotopic specific humidity of qt, qv, ql and qr, defined as the ratio of isotopic mass of H2O18 to moist air.
        PV.add_variable('qt_O18', 'kg/kg','qt_O18_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_O18', 'kg/kg','qv_O18_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_O18', 'kg/kg','ql_O18_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qr_O18', 'kg/kg','qr_O18_isotope','Rain droplets water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qisi_O18', 'kg/kg','qr_O18_isotope','Single Ice droplets water isotopic specific humidity','sym', 'scalar', Pa)

        PV.add_variable('qt_HDO', 'kg/kg','qt_HDO_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_HDO', 'kg/kg','qv_HDO_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_HDO', 'kg/kg','ql_HDO_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qr_HDO', 'kg/kg','qr_HDO_isotope','Rain droplets water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qisi_HDO', 'kg/kg','qr_HDO_isotope','Single Ice droplets water isotopic specific humidity','sym', 'scalar', Pa)

        PV.add_variable('qt_std', 'kg/kg','qt_std','Total water std specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_std', 'kg/kg','qv_std','Vapor water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_std', 'kg/kg','ql_std','Cloud liquid water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qr_std', 'kg/kg','ql_std','Rain water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qisi_std', 'kg/kg','ql_std','Rain water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('nr_std', '','nr_std','Rain water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('nisi_std', '','nisi_std','Single Ice water std specific humidity','sym', 'scalar', Pa)

        # following velocity calculation of rain and single-ice 
        # sedimentation velocity of qt_O18(w_qt_O18) and qr_O18(w_qr_O18),
        # which should be same as qt, qr and qisi, as DVs w_qt, w_qr, w_qisi;
        DV.add_variables('w_qr_std', 'unit', r'w_qr_std','declaration', 'sym', Pa)
        DV.add_variables('w_nr_std', 'unit', r'w_nr_std','declaration', 'sym', Pa)
        DV.add_variables('w_qisi_std', 'unit', r'w_qisi_std','declaration', 'sym', Pa)
        DV.add_variables('w_nisi_std', 'unit', r'w_nisi_std','declaration', 'sym', Pa)
        DV.add_variables('w_qr_O18', 'unit', r'w_qrain_O18','declaration', 'sym', Pa)
        DV.add_variables('w_qr_HDO', 'unit', r'w_qrain_O18','declaration', 'sym', Pa)
        DV.add_variables('w_nr_iso', 'unit', r'w_nr_iso','declaration', 'sym', Pa)
        DV.add_variables('w_qisi_O18', 'unit', r'w_qsnow_O18','declaration', 'sym', Pa)
        DV.add_variables('w_qisi_HDO', 'unit', r'w_qsnow_O18','declaration', 'sym', Pa)
        DV.add_variables('w_nisi_iso', 'unit', r'w_nisi_iso','declaration', 'sym', Pa)
        try:
            self.cloud_sedimentation = namelist['microphysics']['cloud_sedimentation']
        except:
            self.cloud_sedimentation = False
        
        if self.cloud_sedimentation:
            DV.add_variables('w_qt_O18', 'm/s', r'w_{qt_O18}', 'cloud liquid water isotopic sedimentation velocity', 'sym', Pa)
            DV.add_variables('w_qt_HDO', 'm/s', r'w_{qt_HDO}', 'cloud liquid water isotopic sedimentation velocity', 'sym', Pa)
            DV.add_variables('w_qt_std', 'm/s', r'w_{qt_O18}', 'cloud liquid water std sedimentation velocity', 'sym', Pa)
            NS.add_profile('qt_std_sedimentation_flux', Gr, Pa, 'kg/kg', '', '')
            NS.add_profile('qt_O18_sedimentation_flux', Gr, Pa, 'kg/kg', '', '')
            NS.add_profile('qt_HDO_sedimentation_flux', Gr, Pa, 'kg/kg', '', '')

        # diagnose number density results from different microphysics scheme
        self.NI_Mayer    = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.NI_Flecher  = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.NI_Copper   = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.NI_Phillips = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.NI_contact_Young = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.NI_contact_Mayer = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

        NS.add_profile('NI_Mayer', Gr, Pa, 'unit', '', 'NI_Mayer')
        NS.add_profile('NI_Flecher', Gr, Pa, 'unit', '', 'NI_Flecher')
        NS.add_profile('NI_Copper', Gr, Pa, 'unit', '', 'NI_Copper')
        NS.add_profile('NI_Phillips', Gr, Pa, 'unit', '', 'NI_Phillips')
        NS.add_profile('NI_contact_Young', Gr, Pa, 'unit', '', 'NI_contact_Young')
        NS.add_profile('NI_contact_Mayer', Gr, Pa, 'unit', '', 'NI_contact_Mayer')
        NS.add_profile('qr_std', Gr, Pa, 'kg/kg', '', 'stander water tarcer rain')

        NS.add_profile('qr_O18', Gr, Pa, 'kg/kg', '', 'Finial result of rain isotopic sepcific humidity')
        NS.add_profile('qr_HDO', Gr, Pa, 'kg/kg', '', 'Finial result of rain isotopic sepcific humidity')
        NS.add_profile('qisi_std', Gr, Pa, 'kg/kg', '', 'stander water tarcer of single ice')
        NS.add_profile('qisi_O18', Gr, Pa, 'kg/kg', '', 'Finial result of single ice isotopic sepcific humidity')
        NS.add_profile('qisi_HDO', Gr, Pa, 'kg/kg', '', 'Finial result of single ice isotopic sepcific humidity')

        NS.add_profile('qisi_mean_mask', Gr, Pa, 'kg/kg', '', 'qisi mean in domain')
        NS.add_profile('qsnow_mean_mask', Gr, Pa, 'kg/kg', '', 'qsnow mean in domain')

        initialize_NS_base(NS, Gr, Pa)

        return

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref, 
        Microphysics_Arctic_1M.Microphysics_Arctic_1M Micro_Arctic_1M, ThermodynamicsSA.ThermodynamicsSA Th_sa, 
        DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t t_shift      = DV.get_varshift(Gr,'temperature')
            Py_ssize_t qt_shift     = PV.get_varshift(Gr,'qt')
            Py_ssize_t qv_shift     = DV.get_varshift(Gr,'qv')
            Py_ssize_t ql_shift     = DV.get_varshift(Gr,'ql')
            Py_ssize_t s_shift      = PV.get_varshift(Gr,'s')
            Py_ssize_t alpha_shift  = DV.get_varshift(Gr, 'alpha')
            Py_ssize_t qt_std_shift = PV.get_varshift(Gr,'qt_std')
            Py_ssize_t qv_std_shift = PV.get_varshift(Gr,'qv_std')
            Py_ssize_t ql_std_shift = PV.get_varshift(Gr,'ql_std')
            Py_ssize_t qr_std_shift = PV.get_varshift(Gr,'qr_std')
            Py_ssize_t nr_std_shift = PV.get_varshift(Gr,'nr_std')
            Py_ssize_t qisi_std_shift = PV.get_varshift(Gr,'qisi_std')
            Py_ssize_t nisi_std_shift = PV.get_varshift(Gr,'nisi_std')
            Py_ssize_t wqr_std_shift = DV.get_varshift(Gr, 'w_qr_std')
            Py_ssize_t wnr_std_shift = DV.get_varshift(Gr, 'w_nr_std')
            Py_ssize_t wqisi_std_shift = DV.get_varshift(Gr, 'w_qisi_std')
            Py_ssize_t wnisi_std_shift = DV.get_varshift(Gr, 'w_nisi_std')
            
            Py_ssize_t qt_O18_shift = PV.get_varshift(Gr,'qt_O18')
            Py_ssize_t qv_O18_shift = PV.get_varshift(Gr,'qv_O18')
            Py_ssize_t ql_O18_shift = PV.get_varshift(Gr,'ql_O18')
            Py_ssize_t qt_HDO_shift = PV.get_varshift(Gr,'qt_HDO')
            Py_ssize_t qv_HDO_shift = PV.get_varshift(Gr,'qv_HDO')
            Py_ssize_t ql_HDO_shift = PV.get_varshift(Gr,'ql_HDO')
            Py_ssize_t wqt_std_shift

            double[:] precip_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] evap_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] melt_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            
            double[:] nr_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qr_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] nisi_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qisi_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

        # This equilibrium fractioantion processes only happened in cloud
        # based on SBSI scheme, the formation of cloud ice in in microphysics component.
        # iso_equilibrium_fractionation_No_Microphysics(&Gr.dims, &DV.values[t_shift],
        #     &PV.values[qt_std_shift], &PV.values[qv_std_shift], &PV.values[ql_std_shift], 
        #     &PV.values[qt_O18_shift], &PV.values[qv_O18_shift], &PV.values[ql_O18_shift], 
        #     &PV.values[qt_HDO_shift], &PV.values[qv_HDO_shift], &PV.values[ql_HDO_shift], 
        #     &DV.values[qv_shift], &DV.values[ql_shift])
        
        sbsi_NI(&Gr.dims, &PV.values[qt_shift], &Ref.p0_half[0], &Ref.rho0_half[0], &DV.values[t_shift], 
            &self.NI_Mayer[0], &self.NI_Flecher[0], &self.NI_Copper[0], &self.NI_Phillips[0], &self.NI_contact_Young[0],
            &self.NI_contact_Mayer[0])

        sb_si_microphysics_sources(&Gr.dims, self.compute_rain_shape_parameter, self.compute_droplet_nu, 
            &Ref.rho0_half[0],  &Ref.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift], self.ccn, 
            &DV.values[ql_shift], &PV.values[nr_std_shift], &PV.values[qr_std_shift], &PV.values[qisi_std_shift], &PV.values[nisi_std_shift], TS.dt,   
            &nr_std_tend_micro[0], &qr_std_tend_micro[0], &PV.tendencies[nr_std_shift], &PV.tendencies[qr_std_shift],
            &nisi_std_tend_micro[0], &qisi_std_tend_micro[0], &PV.tendencies[nisi_std_shift], &PV.tendencies[qisi_std_shift],
            &precip_rate[0], &evap_rate[0], &melt_rate[0])

        sb_si_qt_source_formation(&Gr.dims, &qisi_std_tend_micro[0], &qr_std_tend_micro[0], &PV.tendencies[qt_shift])

        # sedimentation processes of rain and single_ice: w_qr and w_qisi

        sb_sedimentation_velocity_rain(&Gr.dims, self.compute_rain_shape_parameter, &Ref.rho0_half[0], &PV.values[nr_std_shift],
            &PV.values[qr_std_shift], &DV.values[wnr_std_shift], &DV.values[wqr_std_shift])
        sb_sedimentation_velocity_ice(&Gr.dims, &PV.values[nisi_std_shift], &PV.values[qisi_std_shift], &Ref.rho0_half[0], 
            &DV.values[wnisi_std_shift], &DV.values[wqisi_std_shift])

        if self.cloud_sedimentation:
            wqt_std_shift = DV.get_varshift(Gr, 'w_qt_std')
            if self.stokes_sedimentation:
                microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], self.ccn, &DV.values[ql_shift], &DV.values[wqt_std_shift])
            else:
                sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], self.ccn, &DV.values[ql_shift], &DV.values[wqt_std_shift])

        return 
    
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
            ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            double[:] tmp 

        tmp = Pa.HorizontalMean(Gr, &self.NI_Mayer[0])
        NS.write_profile('NI_Mayer', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        
        tmp = Pa.HorizontalMean(Gr, &self.NI_Flecher[0])
        NS.write_profile('NI_Flecher', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        
        tmp = Pa.HorizontalMean(Gr, &self.NI_Copper[0])
        NS.write_profile('NI_Copper', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        
        tmp = Pa.HorizontalMean(Gr, &self.NI_Phillips[0])
        NS.write_profile('NI_Phillips', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.NI_contact_Young[0])
        NS.write_profile('NI_contact_Young', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        
        tmp = Pa.HorizontalMean(Gr, &self.NI_contact_Mayer[0])
        NS.write_profile('NI_contact_Mayer', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        iso_stats_io_Base(Gr, PV, DV, Ref, NS, Pa)

        return

cdef extern from "microphysics_sb_ice.h":
    
    void saturation_ratio(Grid.DimStruct *dims,  
        Lookup.LookupStruct *LT, double* p0, 
        double* temperature,  double* qt, 
        double* S_lookup, double* S_liq, double* S_ice)

    void sb_ice_deposition_wrapper(Grid.DimStruct *dims, 
        Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        double* temperature, double* qt, double* p0, double* density,
        double* qi, double* ni, double dt, double* qi_tendency, double* ni_tendency, 
        double* ice_dep_tend,double* ice_sub_tend,double* qv_tendency) nogil

    void sb_snow_deposition_wrapper(Grid.DimStruct *dims, 
        Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        double* temperature, double* qt, double* p0, double* density,
        double* qs, double* ns, double dt, double* qs_tendency, double* ns_tendency, 
        double* snow_dep_tend,double* snow_sub_tend,double* qv_tendency) nogil

cdef extern from "isotope.h":
    void tracer_sb_ice_microphysics_sources(Grid.DimStruct *dims, 
        Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
        double* density, double* p0, double dt, 
        double CCN, double IN, 
        double* temperature, double* s, double* w,
        double* qt, double* qv,
        double* nl, double* ql,
        double* ni, double* qi,
        double* nr, double* qr, 
        double* ns, double* qs, 
        double* Dm, double* mass,
        double* diagnose_1, double* diagnose_2,
        double* diagnose_3, double* diagnose_4, double* diagnose_5,
        double* nl_tend, double* ql_tend,
        double* ni_tend, double* qi_tend,
        double* nr_tend_micro, double* qr_tend_micro,
        double* nr_tend, double* qr_tend,
        double* ns_tend_micro, double* qs_tend_micro,
        double* ns_tend, double* qs_tend,
        double* precip_rate, double* evap_rate, double* melt_rate,
        double* qt_O18, double* qv_O18, double* ql_O18, 
        double* qi_O18, double* qr_O18, double* qs_O18,
        double* qt_HDO, double* qv_HDO, double* ql_HDO, 
        double* qi_HDO, double* qr_HDO, double* qs_HDO,
        double* ql_O18_tend, double* qi_O18_tend, 
        double* qr_O18_tend_micro, double* qs_O18_tend_micro,
        double* qr_O18_tend, double* qs_O18_tend,
        double* precip_O18_rate, double* evap_O18_rate, double* melt_O18_rate,
        double* ql_HDO_tend, double* qi_HDO_tend, 
        double* qr_HDO_tend_micro, double* qs_HDO_tend_micro,
        double* qr_HDO_tend, double* qs_HDO_tend,
        double* precip_HDO_rate, double* evap_HDO_rate, double* melt_HDO_rate)nogil
    
    void tracer_sb_cloud_fractionation(Grid.DimStruct * dims, 
        Lookup.LookupStruct * LT, double(*lam_fp)(double), double(*L_fp)(double, double),
        double* p0, double IN, double CCN, double* sat_ratio, double dt,
        double* s, double* w, double* qt, double* temperature,
        double* qv, double* ql, double* nl, double* qi, double* ni,
        double* qt_O18, double* qv_O18,
        double* ql_O18, double* qi_O18,
        double* qt_HDO, double* qv_HDO,
        double* ql_HDO, double* qi_HDO,
        double* ql_tend, double* nl_tend,
        double* qi_tend, double* ni_tend,
        double* ql_O18_tend, double* qi_O18_tend,
        double* ql_HDO_tend, double* qi_HDO_tend) nogil

    void sb_sedimentation_velocity_snow(Grid.DimStruct *dims, 
        double* ns, double* qs, double* ns_velocity, double* qs_velocity) nogil

    void sb_2m_qt_source_formation(Grid.DimStruct *dims, 
        double* qt_tendency, double* precip_rate, double* evap_rate) nogil
    
    void sb_2m_qt_source_debug(Grid.DimStruct *dims, 
        double* qt_tendency, double* qr_tend, double* qs_tend) nogil

    void sb_nuc(Grid.DimStruct *dims,  
        Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        double* rho0, double* p0, double dt, double IN, double* w, double* s,
        double* temperature, double* S, double* qt, double* qv,
        double* nl, double* ql, double* ni, double* qi, 
        double* diag_1, double* diag_2, double* diag_3,
        double* nl_tendency, double* ql_tendency,
        double* ni_tendency, double* qi_tendency)

    # ======= Wapper =======
    void cloud_liquid_wrapper(Grid.DimStruct * dims, 
        Lookup.LookupStruct * LT, double(*lam_fp)(double), double(*L_fp)(double, double),
        double* p0, double IN, double dt,
        double* s, double* qt, double* temperature,
        double* qv, double* ql, double* qi, double* ni,
        double* qt_O18, double* qv_O18,
        double* ql_O18, double* qi_O18,
        double* qt_HDO, double* qv_HDO,
        double* ql_HDO, double* qi_HDO,
        double* ql_cond, double* ql_evap,
        double* ql_O18_cond, double* ql_O18_evap,
        double* ql_HDO_cond, double* ql_HDO_evap)

    void sb_iso_ice_nucleation_wrapper(Grid.DimStruct *dims,  
        Lookup.LookupStruct *LT, double ice_in, double* temperature, double* qt,
        double* p0, double* qv, double* ni, double* qv_o18, double* qv_HDO, 
        double dt, double* qi_tend, double* qi_O18_tend, double* qi_HDO_tend) nogil
    
    void sb_iso_ice_deposition_wrapper(Grid.DimStruct *dims,
        Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        double* temperature, double* qt, double* p0, double* density,double dt,
        double* qi, double* ni, double* qv, double* qv_O18, double* qv_HDO,
        double* qi_O18, double* qi_HDO, double* qi_O18_tend_dep, double* qi_O18_tend_sub, 
        double* qi_HDO_tend_dep, double* qi_HDO_tend_sub) nogil
        
    void sb_iso_snow_deposition_wrapper(Grid.DimStruct *dims,  
        Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        double* temperature, double* qt, double* p0, double* density, double dt,
        double* qs, double* ns, double* qv, double* qv_O18, double* qv_HDO,
        double* qs_O18, double* qs_HDO, double* qs_O18_tend_dep, double* qs_O18_tend_sub, 
        double* qs_HDO_tend_dep, double* qs_HDO_tend_sub) nogil

cdef class IsotopeTracers_SB_Ice:

    def __init__(self, namelist):

        self.isotope_tracer = True

        # Set option for calculation of mu (distribution shape parameter)
        try:
            mu_opt = namelist['microphysics']['SB_Liquid']['mu_rain']
            if mu_opt == 1:
                self.compute_rain_shape_parameter = sb_rain_shape_parameter_1
            elif mu_opt == 2:
                self.compute_rain_shape_parameter = sb_rain_shape_parameter_2
            elif mu_opt == 4:
                self.compute_rain_shape_parameter = sb_rain_shape_parameter_4
            elif mu_opt == 0:
                self.compute_rain_shape_parameter  = sb_rain_shape_parameter_0
            else:
                # Par.root_print("SB_Liquid mu_rain option not recognized, defaulting to option 1")
                self.compute_rain_shape_parameter = sb_rain_shape_parameter_1
        except:
            # Par.root_print("SB_Liquid mu_rain option not selected, defaulting to option 1")
            self.compute_rain_shape_parameter = sb_rain_shape_parameter_1
        # Set option for calculation of nu parameter of droplet distribution
        try:
            nu_opt = namelist['microphysics']['SB_Liquid']['nu_droplet']
            if nu_opt == 0:
                self.compute_droplet_nu = sb_droplet_nu_0
            elif nu_opt == 1:
                self.compute_droplet_nu = sb_droplet_nu_1
            elif nu_opt ==2:
                self.compute_droplet_nu = sb_droplet_nu_2
            else:
                # Par.root_print("SB_Liquid nu_droplet_option not recognized, defaulting to option 0")
                self.compute_droplet_nu = sb_droplet_nu_0
        except:
            # Par.root_print("SB_Liquid nu_droplet_option not selected, defaulting to option 0")
            self.compute_droplet_nu = sb_droplet_nu_0

        try:
            self.order = namelist['scalar_transport']['order_sedimentation']
        except:
            self.order = namelist['scalar_transport']['order']

        try:
            self.cloud_sedimentation = namelist['microphysics']['cloud_sedimentation']
        except:
            self.cloud_sedimentation = False
        if namelist['meta']['casename'] == 'DYCOMS_RF02':
            self.stokes_sedimentation = True
        else:
            self.stokes_sedimentation = False

        return

    cpdef initialize(self, namelist, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
                    DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        Pa.root_print('initialized with IsotopeTracer with SB_Ice scheme')
        
        # Prognostic variable: q_iso, isotopic specific humidity of qt, qv, ql, qi, qr and qs
        # defined as the ratio of isotopic mass of H2O18 to moist air.

        PV.add_variable('qt_O18', 'kg/kg','qt_O18_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        # PV.add_variable('qv_O18', 'kg/kg','qv_O18_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_O18', 'kg/kg','ql_O18_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qi_O18', 'kg/kg','qi_O18_isotope','Cloud Ice water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qr_O18', 'kg/kg','qr_O18_isotope','Rain droplets water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qs_O18', 'kg/kg','qs_O18_isotope','Snow isotopic specific humidity','sym', 'scalar', Pa)

        PV.add_variable('qt_HDO', 'kg/kg','qt_HDO_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        # PV.add_variable('qv_HDO', 'kg/kg','qv_HDO_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_HDO', 'kg/kg','ql_HDO_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qi_HDO', 'kg/kg','qi_HDO_isotope','Cloud ice isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qr_HDO', 'kg/kg','qr_HDO_isotope','Rain droplets water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qs_HDO', 'kg/kg','qr_HDO_isotope','Snow isotopic specific humidity','sym', 'scalar', Pa)

        PV.add_variable('qt_std', 'kg/kg','qt_std','Total water std specific humidity','sym', "scalar", Pa)
        # PV.add_variable('qv_std', 'kg/kg','qv_std','Vapor water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_std', 'kg/kg','ql_std','Cloud liquid water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('nl_std', 'kg/kg','nl_std','Cloud liquid water std number density','sym', 'scalar', Pa)
        PV.add_variable('qi_std', 'kg/kg','ql_std','Cloud ice water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ni_std', 'kg/kg','ni_std','Cloud ice water std number density','sym', 'scalar', Pa)
        PV.add_variable('qr_std', 'kg/kg','ql_std','Rain water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qs_std', 'kg/kg','ql_std','Snow std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('nr_std', '','nr_std','Rain water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ns_std', '','ns_std','Snow water std number density','sym', 'scalar', Pa)

        # following velocity calculation of rain and single-ice 
        # sedimentation velocity of qt_O18(w_qt_O18) and qr_O18(w_qr_O18),
        # which should be same as qt, qr and qs, as DVs w_qt, w_qr, w_qs;

        DV.add_variables('qv_std', 'unit', r'qv_std','DiagnosticVariables of qv std tracer', 'sym', Pa)
        DV.add_variables('qv_O18', 'unit', r'qv_O18','DiagnosticVariables of qv O18 tracer', 'sym', Pa)
        DV.add_variables('qv_HDO', 'unit', r'qv_HDO','DiagnosticVariables of qv HDO tracer', 'sym', Pa)
        DV.add_variables('w_qr_std', 'unit', r'w_qr_std','sedimentation velocity of rain mass', 'sym', Pa)
        DV.add_variables('w_nr_std', 'unit', r'w_nr_std','sedimentation velocity of rain number density', 'sym', Pa)
        DV.add_variables('w_qs_std', 'unit', r'w_qs_std','sedimentation velocity of snow mass', 'sym', Pa)
        DV.add_variables('w_ns_std', 'unit', r'w_ns_std','sedimentation velocity of snow number density', 'sym', Pa)

        DV.add_variables('w_qr_O18', 'unit', r'w_qrain_O18','', 'sym', Pa)
        DV.add_variables('w_nr_O18', 'unit', r'w_nr_iso','', 'sym', Pa)
        DV.add_variables('w_qr_HDO', 'unit', r'w_qrain_O18','', 'sym', Pa)
        DV.add_variables('w_nr_HDO', 'unit', r'w_nr_iso','', 'sym', Pa)

        DV.add_variables('w_qs_O18', 'unit', r'w_qsain_O18','', 'sym', Pa)
        DV.add_variables('w_ns_O18', 'unit', r'w_ns_iso','', 'sym', Pa)
        DV.add_variables('w_qs_HDO', 'unit', r'w_qsain_O18','', 'sym', Pa)
        DV.add_variables('w_ns_HDO', 'unit', r'w_ns_iso','', 'sym', Pa)
        
        try:
            self.cloud_sedimentation = namelist['microphysics']['cloud_sedimentation']
        except:
            self.cloud_sedimentation = False
        
        if self.cloud_sedimentation:
            DV.add_variables('w_qt_O18', 'm/s', r'w_{qt_O18}', 'cloud liquid water isotopic sedimentation velocity', 'sym', Pa)
            DV.add_variables('w_qt_HDO', 'm/s', r'w_{qt_HDO}', 'cloud liquid water isotopic sedimentation velocity', 'sym', Pa)
            DV.add_variables('w_qt_std', 'm/s', r'w_{qt_O18}', 'cloud liquid water std sedimentation velocity', 'sym', Pa)
            NS.add_profile('qt_std_sedimentation_flux', Gr, Pa, 'kg/kg', '', '')
            NS.add_profile('qt_O18_sedimentation_flux', Gr, Pa, 'kg/kg', '', '')
            NS.add_profile('qt_HDO_sedimentation_flux', Gr, Pa, 'kg/kg', '', '')

        # diagnose number density results from different microphysics scheme

        NS.add_profile('qr_std', Gr, Pa, 'kg/kg', '', 'stander water tarcer rain')
        NS.add_profile('qr_O18', Gr, Pa, 'kg/kg', '', 'Finial result of rain isotopic sepcific humidity')
        NS.add_profile('qr_HDO', Gr, Pa, 'kg/kg', '', 'Finial result of rain isotopic sepcific humidity')

        NS.add_profile('qs_std', Gr, Pa, 'kg/kg', '', 'stander water tarcer of snow')
        NS.add_profile('qs_O18', Gr, Pa, 'kg/kg', '', 'Finial result of snow isotopic sepcific humidity')
        NS.add_profile('qs_HDO', Gr, Pa, 'kg/kg', '', 'Finial result of snow isotopic sepcific humidity')

        self.Dm         = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.mass       = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.diagnose_1 = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.diagnose_2 = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.diagnose_3 = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.diagnose_4 = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.diagnose_5 = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        
        NS.add_profile('Dm_tracer', Gr, Pa, '','','')
        NS.add_profile('mass_tracer', Gr, Pa, '','','')
        NS.add_profile('diagnose_1_tracer', Gr, Pa, '','','')
        NS.add_profile('diagnose_2_tracer', Gr, Pa, '','','')
        NS.add_profile('diagnose_3_tracer', Gr, Pa, '','','')
        NS.add_profile('diagnose_4_tracer', Gr, Pa, '','','')
        NS.add_profile('diagnose_5_tracer', Gr, Pa, '','','')

        initialize_NS_base(NS, Gr, Pa)

        NS.add_profile('qi_tend_nuc', Gr, Pa, '', '', '')
        NS.add_profile('qi_O18_tend_nuc', Gr, Pa, '', '', '')
        NS.add_profile('qi_HDO_tend_nuc', Gr, Pa, '', '', '')
            
        NS.add_profile('ql_tend_cond', Gr, Pa, '','','')
        NS.add_profile('ql_tend_evap', Gr, Pa, '','','')
        NS.add_profile('ql_O18_tend_cond', Gr, Pa, '','','')
        NS.add_profile('ql_O18_tend_evap', Gr, Pa, '','','')
        NS.add_profile('ql_HDO_tend_cond', Gr, Pa, '','','')
        NS.add_profile('ql_HDO_tend_evap', Gr, Pa, '','','')

        NS.add_profile('qi_O18_tend_dep', Gr, Pa, '', '', '')
        NS.add_profile('qi_HDO_tend_dep', Gr, Pa, '', '', '')
        NS.add_profile('qi_O18_tend_sub', Gr, Pa, '', '', '')
        NS.add_profile('qi_HDO_tend_sub', Gr, Pa, '', '', '')
        
        NS.add_profile('qs_O18_tend_dep', Gr, Pa, '', '', '')
        NS.add_profile('qs_HDO_tend_dep', Gr, Pa, '', '', '')
        NS.add_profile('qs_O18_tend_sub', Gr, Pa, '', '', '')
        NS.add_profile('qs_HDO_tend_sub', Gr, Pa, '', '', '')

        return

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref, 
        Microphysics_SB_2M.Microphysics_SB_2M Micro_SB_2M, ThermodynamicsSB.ThermodynamicsSB Th_sb, 
        DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t t_shift  = DV.get_varshift(Gr,'temperature')
            Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')
            Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')
            Py_ssize_t ql_shift = PV.get_varshift(Gr,'ql')
            # Py_ssize_t qi_shift = PV.get_varshift(Gr,'qi')
            Py_ssize_t s_shift  = PV.get_varshift(Gr,'s')
            Py_ssize_t w_shift  = PV.get_varshift(Gr,'w')
            
            Py_ssize_t alpha_shift  = DV.get_varshift(Gr, 'alpha')

            Py_ssize_t qt_std_shift = PV.get_varshift(Gr,'qt_std')
            Py_ssize_t qv_std_shift = DV.get_varshift(Gr,'qv_std')
            Py_ssize_t ql_std_shift = PV.get_varshift(Gr,'ql_std')
            Py_ssize_t qi_std_shift = PV.get_varshift(Gr,'qi_std')
            Py_ssize_t qr_std_shift = PV.get_varshift(Gr,'qr_std')
            Py_ssize_t nr_std_shift = PV.get_varshift(Gr,'nr_std')
            Py_ssize_t qs_std_shift = PV.get_varshift(Gr,'qs_std')
            Py_ssize_t ns_std_shift = PV.get_varshift(Gr,'ns_std')

            Py_ssize_t nl_std_shift = PV.get_varshift(Gr,'nl_std')
            Py_ssize_t ni_std_shift = PV.get_varshift(Gr,'ni_std')

            Py_ssize_t wqr_std_shift = DV.get_varshift(Gr, 'w_qr_std')
            Py_ssize_t wnr_std_shift = DV.get_varshift(Gr, 'w_nr_std')
            Py_ssize_t wqs_std_shift = DV.get_varshift(Gr, 'w_qs_std')
            Py_ssize_t wns_std_shift = DV.get_varshift(Gr, 'w_ns_std')
            
            Py_ssize_t wqt_std_shift
            
            # TMP avoid isotope index component defination

            Py_ssize_t qt_O18_shift = PV.get_varshift(Gr,'qt_O18')
            Py_ssize_t qv_O18_shift = DV.get_varshift(Gr,'qv_O18')
            Py_ssize_t ql_O18_shift = PV.get_varshift(Gr,'ql_O18')
            Py_ssize_t qi_O18_shift = PV.get_varshift(Gr,'qi_O18')
            Py_ssize_t qr_O18_shift = PV.get_varshift(Gr,'qr_O18')
            Py_ssize_t qs_O18_shift = PV.get_varshift(Gr,'qs_O18')
            Py_ssize_t wqr_O18_shift = DV.get_varshift(Gr, 'w_qr_O18')
            Py_ssize_t wqs_O18_shift = DV.get_varshift(Gr, 'w_qs_O18')
            Py_ssize_t wnr_O18_shift = DV.get_varshift(Gr, 'w_nr_O18')
            Py_ssize_t wns_O18_shift = DV.get_varshift(Gr, 'w_ns_O18')
            
            Py_ssize_t qt_HDO_shift = PV.get_varshift(Gr,'qt_HDO')
            Py_ssize_t qv_HDO_shift = DV.get_varshift(Gr,'qv_HDO')
            Py_ssize_t ql_HDO_shift = PV.get_varshift(Gr,'ql_HDO')
            Py_ssize_t qi_HDO_shift = PV.get_varshift(Gr,'qi_HDO')
            Py_ssize_t qr_HDO_shift = PV.get_varshift(Gr,'qr_HDO')
            Py_ssize_t qs_HDO_shift = PV.get_varshift(Gr,'qs_HDO')
            Py_ssize_t wqr_HDO_shift = DV.get_varshift(Gr, 'w_qr_HDO')
            Py_ssize_t wqs_HDO_shift = DV.get_varshift(Gr, 'w_qs_HDO')
            Py_ssize_t wnr_HDO_shift = DV.get_varshift(Gr, 'w_nr_HDO')
            Py_ssize_t wns_HDO_shift = DV.get_varshift(Gr, 'w_ns_HDO')

            double[:] precip_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] evap_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] melt_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] precip_O18_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] evap_O18_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] melt_O18_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] precip_HDO_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] evap_HDO_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] melt_HDO_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            
            double[:] qr_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] nr_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qs_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] ns_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] S_ratio = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

            double[:] qr_O18_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qs_O18_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qr_HDO_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qs_HDO_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] saturation_ratio_lookup = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] saturation_ratio_liquid = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] saturation_ratio_ice = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            
        saturation_ratio(&Gr.dims, 
            &Micro_SB_2M.CC.LT.LookupStructC,
            &Ref.p0_half[0], &DV.values[t_shift], 
            &PV.values[qt_shift],
            &saturation_ratio_lookup[0], &saturation_ratio_liquid[0], &saturation_ratio_ice[0])

        tracer_sb_cloud_fractionation(&Gr.dims,
            &Micro_SB_2M.CC.LT.LookupStructC, Micro_SB_2M.Lambda_fp, Micro_SB_2M.L_fp,
            &Ref.p0_half[0], Micro_SB_2M.ice_nucl, 
            Micro_SB_2M.CCN, &saturation_ratio_liquid[0], TS.dt,
            &PV.values[s_shift], &PV.values[w_shift],
            &PV.values[qt_std_shift], &DV.values[t_shift],
            &DV.values[qv_std_shift], &PV.values[ql_std_shift], &PV.values[nl_std_shift],
            &PV.values[qi_std_shift], &PV.values[ni_std_shift],
            &PV.values[qt_O18_shift], &DV.values[qv_O18_shift], 
            &PV.values[ql_O18_shift], &PV.values[qi_O18_shift],
            &PV.values[qt_HDO_shift], &DV.values[qv_HDO_shift], 
            &PV.values[ql_HDO_shift], &PV.values[qi_HDO_shift],
            &PV.tendencies[ql_std_shift], &PV.tendencies[nl_std_shift],
            &PV.tendencies[qi_std_shift], &PV.tendencies[ni_std_shift],
            &PV.tendencies[ql_O18_shift], &PV.tendencies[qi_O18_shift],
            &PV.tendencies[ql_HDO_shift], &PV.tendencies[qi_HDO_shift])
        
        tracer_sb_ice_microphysics_sources(&Gr.dims, 
            # thermodynamics setting
            &Micro_SB_2M.CC.LT.LookupStructC, Micro_SB_2M.Lambda_fp, Micro_SB_2M.L_fp,
            # two moment rain droplet mu variable setting
            Micro_SB_2M.compute_rain_shape_parameter, Micro_SB_2M.compute_droplet_nu, 
            # INPUT ARRAY INDEX
            &Ref.rho0_half[0], &Ref.p0_half[0], TS.dt,
            Micro_SB_2M.CCN, Micro_SB_2M.ice_nucl, 
            &DV.values[t_shift], &PV.values[s_shift], &PV.values[w_shift],
            &PV.values[qt_std_shift], &DV.values[qv_std_shift],
            &PV.values[nl_std_shift], &PV.values[ql_std_shift],
            &PV.values[ni_std_shift], &PV.values[qi_std_shift],
            &PV.values[nr_std_shift], &PV.values[qr_std_shift],
            &PV.values[ns_std_shift], &PV.values[qs_std_shift], 
            # ------ DIAGNOSED VARIABLES ---------
            &self.Dm[0], &self.mass[0],
            &self.diagnose_1[0], &self.diagnose_2[0],
            &self.diagnose_3[0], &self.diagnose_4[0], &self.diagnose_5[0],
            # ------------------------------------
            &PV.tendencies[nl_std_shift], &PV.tendencies[ql_std_shift],
            &PV.tendencies[ni_std_shift], &PV.tendencies[qi_std_shift],
            &nr_std_tend_micro[0], &qr_std_tend_micro[0], &PV.tendencies[nr_std_shift], &PV.tendencies[qr_std_shift],
            &ns_std_tend_micro[0], &qs_std_tend_micro[0], &PV.tendencies[ns_std_shift], &PV.tendencies[qs_std_shift],
            &precip_rate[0], &evap_rate[0], &melt_rate[0],
            &PV.values[qt_O18_shift], &DV.values[qv_O18_shift], &PV.values[ql_O18_shift],
            &PV.values[qi_O18_shift], &PV.values[qs_O18_shift], &PV.values[qr_O18_shift],
            &PV.values[qt_HDO_shift], &DV.values[qv_HDO_shift], &PV.values[ql_HDO_shift],
            &PV.values[qi_HDO_shift], &PV.values[qs_HDO_shift], &PV.values[qr_HDO_shift],
            &PV.tendencies[ql_O18_shift], &PV.tendencies[qi_O18_shift],
            &qr_O18_tend_micro[0], &qs_O18_tend_micro[0],
            &PV.tendencies[qs_O18_shift], &PV.tendencies[qr_O18_shift],
            &precip_O18_rate[0], &evap_O18_rate[0], &melt_O18_rate[0],
            &PV.tendencies[ql_HDO_shift], &PV.tendencies[qi_HDO_shift],
            &qr_HDO_tend_micro[0], &qs_HDO_tend_micro[0],
            &PV.tendencies[qs_HDO_shift], &PV.tendencies[qr_HDO_shift],
            &precip_HDO_rate[0], &evap_HDO_rate[0], &melt_HDO_rate[0])

        sb_2m_qt_source_debug(&Gr.dims, &PV.tendencies[qt_std_shift], 
            &qr_std_tend_micro[0], &qs_std_tend_micro[0])

        # sedimentation processes of rain and single_ice: w_qr and w_qs
        sb_sedimentation_velocity_rain(&Gr.dims, Micro_SB_2M.compute_rain_shape_parameter, 
            &Ref.rho0_half[0], &PV.values[nr_std_shift], &PV.values[qr_std_shift], 
            &DV.values[wnr_std_shift], &DV.values[wqr_std_shift])
        sb_sedimentation_velocity_snow(&Gr.dims, 
            &PV.values[ns_std_shift], &PV.values[qs_std_shift], 
            &DV.values[wns_std_shift], &DV.values[wqs_std_shift])

        if self.cloud_sedimentation:
            wqt_std_shift = DV.get_varshift(Gr, 'w_qt_std')
            if self.stokes_sedimentation:
                microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_2M.CCN, 
                &PV.values[ql_shift], &DV.values[wqt_std_shift])
            else:
                sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_2M.CCN, 
                &PV.values[ql_shift], &DV.values[wqt_std_shift])

        # ============ Isotope Balance ===========

        # TODO: check weather change the std source of qr and qs 
        # into O18 and HDO will change too much of isotopic values
        sb_2m_qt_source_debug(&Gr.dims, &PV.tendencies[qt_O18_shift], 
            &qr_O18_tend_micro[0], &qs_O18_tend_micro[0])

        # sedimentation processes of rain and single_ice: w_qr and w_qs
        sb_sedimentation_velocity_rain(&Gr.dims, Micro_SB_2M.compute_rain_shape_parameter, 
            &Ref.rho0_half[0], &PV.values[nr_std_shift], &PV.values[qr_std_shift], 
            &DV.values[wnr_O18_shift], &DV.values[wqr_O18_shift])
        sb_sedimentation_velocity_snow(&Gr.dims,
            &PV.values[ns_std_shift], &PV.values[qs_O18_shift], 
            &DV.values[wns_O18_shift], &DV.values[wqs_O18_shift])

        if self.cloud_sedimentation:
            wqt_O18_shift = DV.get_varshift(Gr, 'w_qt_O18')
            if self.stokes_sedimentation:
                microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_2M.CCN, 
                &PV.values[ql_shift], &DV.values[wqt_O18_shift])
            else:
                sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_2M.CCN, 
                &PV.values[ql_shift], &DV.values[wqt_O18_shift])

        sb_2m_qt_source_debug(&Gr.dims, &PV.tendencies[qt_HDO_shift], 
            &qr_HDO_tend_micro[0], &qs_HDO_tend_micro[0])

        # sedimentation processes of rain and single_ice: w_qr and w_qs
        sb_sedimentation_velocity_rain(&Gr.dims, Micro_SB_2M.compute_rain_shape_parameter, 
            &Ref.rho0_half[0], &PV.values[nr_std_shift], &PV.values[qr_std_shift], 
            &DV.values[wnr_HDO_shift], &DV.values[wqr_HDO_shift])
        sb_sedimentation_velocity_snow(&Gr.dims,
            &PV.values[ns_std_shift], &PV.values[qs_HDO_shift], 
            &DV.values[wns_HDO_shift], &DV.values[wqs_HDO_shift])

        if self.cloud_sedimentation:
            wqt_HDO_shift = DV.get_varshift(Gr, 'w_qt_HDO')
            if self.stokes_sedimentation:
                microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_2M.CCN, 
                &PV.values[ql_shift], &DV.values[wqt_HDO_shift])
            else:
                sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_2M.CCN, 
                &PV.values[ql_shift], &DV.values[wqt_HDO_shift])

        return 

    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
            DiagnosticVariables.DiagnosticVariables DV,
            ReferenceState.ReferenceState Ref, Microphysics_SB_2M.Microphysics_SB_2M Micro_SB_2M, 
            TimeStepping.TimeStepping TS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t t_shift  = DV.get_varshift(Gr,'temperature')
            Py_ssize_t s_shift  = PV.get_varshift(Gr,'s')
            Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')
            Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')
            Py_ssize_t ql_shift = PV.get_varshift(Gr,'ql')
            Py_ssize_t qi_shift = PV.get_varshift(Gr,'qi')
            Py_ssize_t ni_shift = PV.get_varshift(Gr,'ni')
            Py_ssize_t qs_shift = PV.get_varshift(Gr,'qs')
            Py_ssize_t ns_shift = PV.get_varshift(Gr,'ns')

            Py_ssize_t qt_O18_shift = PV.get_varshift(Gr,'qt_O18')
            Py_ssize_t qv_O18_shift = DV.get_varshift(Gr,'qv_O18')
            Py_ssize_t ql_O18_shift = PV.get_varshift(Gr,'ql_O18')
            Py_ssize_t qi_O18_shift = PV.get_varshift(Gr,'qi_O18')
            Py_ssize_t qr_O18_shift = PV.get_varshift(Gr,'qr_O18')
            Py_ssize_t qs_O18_shift = PV.get_varshift(Gr,'qs_O18')
            Py_ssize_t qt_HDO_shift = PV.get_varshift(Gr,'qt_HDO')
            Py_ssize_t qv_HDO_shift = DV.get_varshift(Gr,'qv_HDO')
            Py_ssize_t ql_HDO_shift = PV.get_varshift(Gr,'ql_HDO')
            Py_ssize_t qi_HDO_shift = PV.get_varshift(Gr,'qi_HDO')
            Py_ssize_t qr_HDO_shift = PV.get_varshift(Gr,'qr_HDO')
            Py_ssize_t qs_HDO_shift = PV.get_varshift(Gr,'qs_HDO')

            double[:] tmp
            double[:] qi_tend_nuc = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            double[:] qi_O18_tend_nuc = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            double[:] qi_HDO_tend_nuc = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            
            double[:] ql_tend_cond = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            double[:] ql_tend_evap = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            double[:] ql_O18_tend_cond = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            double[:] ql_O18_tend_evap = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            double[:] ql_HDO_tend_cond = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            double[:] ql_HDO_tend_evap = np.zeros(Gr.dims.npg, dtype=np.double, order='c')

            double[:] qi_O18_tend_dep = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            double[:] qi_HDO_tend_dep = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            double[:] qi_O18_tend_sub = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            double[:] qi_HDO_tend_sub = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            
            double[:] qs_O18_tend_dep = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            double[:] qs_HDO_tend_dep = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            double[:] qs_O18_tend_sub = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            double[:] qs_HDO_tend_sub = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            double[:] tmp_tend = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            
        cloud_liquid_wrapper(&Gr.dims,
            &Micro_SB_2M.CC.LT.LookupStructC, Micro_SB_2M.Lambda_fp, Micro_SB_2M.L_fp,
            &Ref.p0_half[0], Micro_SB_2M.ice_nucl, TS.dt,
            &PV.values[s_shift], &PV.values[qt_shift], &DV.values[t_shift],
            &DV.values[qv_shift], &PV.values[ql_shift],
            &PV.values[qi_shift], &PV.values[ni_shift],
            &PV.values[qt_O18_shift], &DV.values[qv_O18_shift], 
            &PV.values[ql_O18_shift], &PV.values[qi_O18_shift],
            &PV.values[qt_HDO_shift], &DV.values[qv_HDO_shift], 
            &PV.values[ql_HDO_shift], &PV.values[qi_HDO_shift],
            &ql_tend_cond[0], &ql_tend_evap[0], &ql_O18_tend_cond[0],
            &ql_O18_tend_evap[0], &ql_HDO_tend_cond[0], &ql_HDO_tend_evap[0])

        tmp = Pa.HorizontalMean(Gr, &ql_tend_cond[0])
        NS.write_profile('ql_tend_cond', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &ql_tend_evap[0])
        NS.write_profile('ql_tend_evap', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &ql_O18_tend_cond[0])
        NS.write_profile('ql_O18_tend_cond', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &ql_O18_tend_evap[0])
        NS.write_profile('ql_O18_tend_evap', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &ql_HDO_tend_cond[0])
        NS.write_profile('ql_HDO_tend_cond', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &ql_HDO_tend_evap[0])
        NS.write_profile('ql_HDO_tend_evap', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)

        sb_iso_ice_nucleation_wrapper(&Gr.dims, 
            &Micro_SB_2M.CC.LT.LookupStructC, Micro_SB_2M.ice_nucl,
            &DV.values[t_shift], &PV.values[qt_shift], &Ref.p0_half[0], 
            &PV.values[qv_shift], &PV.values[ni_shift],
            &PV.values[qv_O18_shift], &PV.values[qv_HDO_shift], TS.dt,
            &qi_tend_nuc[0], &qi_O18_tend_nuc[0], &qi_HDO_tend_nuc[0])

        tmp = Pa.HorizontalMean(Gr, &qi_tend_nuc[0])
        NS.write_profile('qi_tend_nuc', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &qi_O18_tend_nuc[0])
        NS.write_profile('qi_O18_tend_nuc', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &qi_HDO_tend_nuc[0])
        NS.write_profile('qi_HDO_tend_nuc', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)

        # sb_ice_deposition_wrapper(&Gr.dims, 
        #     &Micro_SB_2M.CC.LT.LookupStructC, Micro_SB_2M.Lambda_fp, Micro_SB_2M.L_fp,
        #     &DV.values[t_shift], &PV.values[qt_shift], &Ref.p0_half[0], &Ref.rho0_half[0],
        #     &PV.values[qi_shift], &PV.values[ni_shift], TS.dt, 
        #     &tmp_tend[0], &tmp_tend[0],
        #     &qi_O18_tend_dep[0], &qi_O18_tend_sub[0], &tmp_tend[0])

        sb_iso_ice_deposition_wrapper(&Gr.dims, 
            &Micro_SB_2M.CC.LT.LookupStructC, Micro_SB_2M.Lambda_fp, Micro_SB_2M.L_fp,
            &DV.values[t_shift], &PV.values[qt_shift], &Ref.p0_half[0], &Ref.rho0_half[0], TS.dt,
            &PV.values[qi_shift], &PV.values[ni_shift], &DV.values[qv_shift], 
            &DV.values[qv_O18_shift], &DV.values[qv_HDO_shift], &PV.values[qi_O18_shift], &PV.values[qi_HDO_shift],
            &qi_O18_tend_dep[0], &qi_O18_tend_sub[0], &qi_HDO_tend_dep[0], &qi_HDO_tend_sub[0])

        tmp = Pa.HorizontalMean(Gr, &qi_O18_tend_dep[0])
        NS.write_profile('qi_O18_tend_dep', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &qi_HDO_tend_dep[0])
        NS.write_profile('qi_HDO_tend_dep', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &qi_O18_tend_sub[0])
        NS.write_profile('qi_O18_tend_sub', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &qi_HDO_tend_sub[0])
        NS.write_profile('qi_HDO_tend_sub', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)

        sb_iso_snow_deposition_wrapper(&Gr.dims,
            &Micro_SB_2M.CC.LT.LookupStructC, Micro_SB_2M.Lambda_fp, Micro_SB_2M.L_fp,
            &DV.values[t_shift], &PV.values[qt_shift], &Ref.p0_half[0],&Ref.rho0_half[0], TS.dt, 
            &PV.values[qs_shift], &PV.values[ns_shift], &DV.values[qv_shift], 
            &DV.values[qs_O18_shift], &DV.values[qs_HDO_shift], &PV.values[qs_O18_shift], &PV.values[qs_HDO_shift], 
            &qs_O18_tend_dep[0], &qs_O18_tend_sub[0], &qs_HDO_tend_dep[0], &qs_HDO_tend_sub[0])
        
        tmp = Pa.HorizontalMean(Gr, &qs_O18_tend_dep[0])
        NS.write_profile('qs_O18_tend_dep', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &qs_HDO_tend_dep[0])
        NS.write_profile('qs_HDO_tend_dep', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &qs_O18_tend_sub[0])
        NS.write_profile('qs_O18_tend_sub', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &qs_HDO_tend_sub[0])
        NS.write_profile('qs_HDO_tend_sub', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        
        tmp = Pa.HorizontalMean(Gr, &self.Dm[0])
        NS.write_profile('Dm_tracer', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.mass[0])
        NS.write_profile('mass_tracer', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.diagnose_1[0])
        NS.write_profile('diagnose_1_tracer', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.diagnose_2[0])
        NS.write_profile('diagnose_2_tracer', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.diagnose_3[0])
        NS.write_profile('diagnose_3_tracer', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.diagnose_4[0])
        NS.write_profile('diagnose_4_tracer', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.diagnose_5[0])
        NS.write_profile('diagnose_5_tracer', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)

        return

cpdef iso_stats_io_Base(Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
            ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
    # cdef:
    #     Py_ssize_t imin    = 0
    #     Py_ssize_t jmin    = 0
    #     Py_ssize_t kmin    = 0
    #     Py_ssize_t imax    = Gr.dims.nlg[0]
    #     Py_ssize_t jmax    = Gr.dims.nlg[1]
    #     Py_ssize_t kmax    = Gr.dims.nlg[2]
    #     Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
    #     Py_ssize_t jstride = Gr.dims.nlg[2]
    #     Py_ssize_t ishift, jshift, ijk, i,j,k, iter = 0
    #     Py_ssize_t ql_shift     = DV.get_varshift(Gr, 'ql')
    #     Py_ssize_t qt_std_shift = PV.get_varshift(Gr,'qt_std')
    #     Py_ssize_t qv_std_shift = PV.get_varshift(Gr,'qv_std')
    #     Py_ssize_t ql_std_shift = PV.get_varshift(Gr,'ql_std')
    #     Py_ssize_t qt_O18_shift = PV.get_varshift(Gr,'qt_O18')
    #     Py_ssize_t qv_O18_shift = PV.get_varshift(Gr,'qv_O18')
    #     Py_ssize_t ql_O18_shift = PV.get_varshift(Gr,'ql_O18')
    #     double[:] tmp
    #     double[:] delta_qv       = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
    #     double[:] delta_qt       = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
    #     double[:] delta_ql_cloud = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
    #     double[:] cloud_mask     = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
    #     double[:] no_cloud_mask  = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
    #
    # with nogil:
    #     with cython.boundscheck(False):
    #         for i in xrange(imin,imax):
    #             ishift = i*istride
    #             for j in xrange(jmin,jmax):
    #                 jshift = j*jstride
    #                 for k in xrange(kmin,kmax):
    #                     ijk = ishift + jshift + k
    #                     if PV.values[ql_shift + ijk] > 0.0:
    #                         cloud_mask[ijk] = 1.0
    #                         delta_ql_cloud[ijk] = q_2_delta(PV.values[ql_O18_shift + ijk], PV.values[ql_std_shift + ijk])
    #                     else:
    #                         no_cloud_mask[ijk] = 1.0 
    #                     delta_qv[ijk] = q_2_delta(PV.values[qv_O18_shift + ijk], PV.values[qv_std_shift + ijk])
    #                     delta_qt[ijk] = q_2_delta(PV.values[qt_O18_shift + ijk], PV.values[qt_std_shift + ijk])
    #
    #
    # tmp = Pa.HorizontalMean(Gr, &PV.values[qt_std_shift])
    # NS.write_profile('qt_std', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
    # 
    # tmp = Pa.HorizontalMean(Gr, &PV.values[qv_std_shift])
    # NS.write_profile('qv_std', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
    # 
    # tmp = Pa.HorizontalMean(Gr, &PV.values[ql_std_shift])
    # NS.write_profile('ql_std', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
    #
    # tmp = Pa.HorizontalMean(Gr, &PV.values[qt_O18_shift])
    # statsIO_isotope_scaling_magnitude(&Gr.dims, &tmp[0]) # scaling back to correct magnitude
    # NS.write_profile('qt_O18', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
    #
    # tmp = Pa.HorizontalMean(Gr, &PV.values[ql_O18_shift])
    # statsIO_isotope_scaling_magnitude(&Gr.dims, &tmp[0]) # scaling back to correct magnitude
    # NS.write_profile('ql_O18', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
    #
    # tmp = Pa.HorizontalMean(Gr, &PV.values[qv_O18_shift])
    # statsIO_isotope_scaling_magnitude(&Gr.dims, &tmp[0]) # scaling back to correct magnitude
    # NS.write_profile('qv_O18', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
    # 
    # tmp = Pa.HorizontalMean(Gr, &delta_qv[0])
    # NS.write_profile('delta_qv', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)   
    #
    # tmp = Pa.HorizontalMean(Gr, &delta_qt[0])
    # NS.write_profile('delta_qt', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)   
    # 
    # tmp = Pa.HorizontalMeanConditional(Gr, &delta_ql_cloud[0], &cloud_mask[0])
    # NS.write_profile('delta_ql_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)   
    #     
    # tmp = Pa.HorizontalMeanConditional(Gr, &delta_qv[0], &cloud_mask[0])
    # NS.write_profile('delta_qv_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)   
    #
    # tmp = Pa.HorizontalMeanConditional(Gr, &delta_qv[0], &no_cloud_mask[0])
    # NS.write_profile('delta_qv_Nocloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)   
    return
cpdef initialize_NS_base(NetCDFIO_Stats NS, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):
    NS.add_profile('qt_std', Gr, Pa, 'kg/kg', '', 'stander water tracer total')
    NS.add_profile('qv_std', Gr, Pa, 'kg/kg', '', 'stander water tracer vapor')
    NS.add_profile('ql_std', Gr, Pa, 'kg/kg', '', 'stander water tracer liquid')
    NS.add_profile('qt_O18', Gr, Pa, 'kg/kg', '', 'Finial result of total water isotopic specific humidity')
    NS.add_profile('qv_O18', Gr, Pa, 'kg/kg', '', 'Finial result of vapor isotopic specific humidity')
    NS.add_profile('ql_O18', Gr, Pa, 'kg/kg', '', 'Finial result of liquid isotopic sepcific humidity')
    
    NS.add_profile('qr_O18_evap_tendency', Gr, Pa, 'kg/kg', '', '')
    NS.add_profile('qr_std_evap_tendency', Gr, Pa, 'kg/kg', '', '')
    NS.add_profile('qr_std_auto_tendency', Gr, Pa, 'kg/kg', '', '')
    NS.add_profile('qr_std_accre_tendency', Gr, Pa, 'kg/kg', '', '')
    NS.add_profile('qr_tend_micro_diff', Gr, Pa, 'kg/kg', '', '')
    NS.add_profile('qt_tendencies_diff', Gr, Pa, 'kg/kg', '', '')
    NS.add_profile('qr_std_sedimentation_flux', Gr, Pa, 'kg/kg', '', '')

    NS.add_profile('delta_qt', Gr, Pa, 'permil', '', 'delta of qt, calculated by qt_O18/qt_std during fractioantion')
    NS.add_profile('delta_qv', Gr, Pa, 'permil', '', 'delta of qv, calculated by qt_O18/qt_std during fractioantion')
    NS.add_profile('delta_ql_cloud', Gr, Pa, 'permil', '', 'delta of ql in cloud, calculated by qt_O18/qt_std during fractioantion')
    NS.add_profile('delta_qv_cloud', Gr, Pa, 'permil', '', 'delta of qv in cloud, calculated by qt_O18/qt_std during fractioantion')
    NS.add_profile('delta_qv_Nocloud', Gr, Pa, 'permil', '', 'delta of qv in cloud, calculated by qt_O18/qt_std during fractioantion')
    return
