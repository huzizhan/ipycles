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
cimport TimeStepping
cimport Lookup
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
cimport Microphysics_SB_Liquid
cimport Microphysics_Arctic_1M
cimport Microphysics_SB_SI
cimport ThermodynamicsSA
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
        double *qt_iso_O18, double *qv_iso_O18, double *ql_iso_O18,
        double *qt_iso_HDO, double *qv_iso_HDO, double *ql_iso_HDO) nogil

    void delta_isotopologue(Grid.DimStruct *dims, double *q_iso, double *q_std,
        double *delta, int index) nogil
    void compute_sedimentaion(Grid.DimStruct *dims, double *w_q, double *w_q_iso, double *w_q_std) nogil
    void tracer_constrain_NoMicro(Grid.DimStruct *dims, double *ql, 
        double*ql_std, double *ql_iso_O18, double *qv_std, double *qv_iso_O18, 
        double*qt_std, double *qt_iso_O18) nogil
    void iso_mix_phase_fractionation(Grid.DimStruct *dims, Lookup.LookupStruct *LT, 
        double(*lam_fp)(double), double(*L_fp)(double, double),
        double *temperature, double *p0,
        double *qt_std, double *qv_std, double *ql_std, double *qi_std, 
        double *qt_iso_O18, double *qv_iso_O18, double *ql_iso_O18, double *qi_iso_O18, 
        double *qt_iso_HDO, double *qv_iso_HDO, double *ql_iso_HDO, double *qi_iso_HDO, 
        double *qv_DV, double *ql_DV, double *qi_DV) nogil
        
    void tracer_sb_liquid_microphysics_sources(Grid.DimStruct *dims, Lookup.LookupStruct *LT, 
        double (*lam_fp)(double), double (*L_fp)(double, double),
        double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
        double* density, double* p0, double* temperature,  double* qt, double ccn,
        double* ql, double* nr, double* qr, double dt, double* nr_tendency_micro, double* qr_tendency_micro,
        double* nr_tendency, double* qr_tendency,
        double* qr_iso_O18, double* qt_iso_O18, double* qv_iso_O18, double* ql_iso_O18,
        double* qr_iso_HDO, double* qt_iso_HDO, double* qv_iso_HDO, double* ql_iso_HDO,
        double* qr_iso_O18_tendency_micro, double* qr_iso_O18_tendency, 
        double* qr_iso_HDO_tendency_micro, double* qr_iso_HDO_tendency) nogil
    
    void tracer_arctic1m_microphysics_sources(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double), 
        double (*L_fp)(double, double), double* density, double* p0, double ccn, double n0_ice, double dt,
        double* temperature, double* qt, double* qv, double* ql, double* qi, 
        double* qrain, double* nrain, double* qsnow, double* nsnow, double* ql_std, double* qi_std,
        double* qrain_tendency_micro, double* qrain_tendency,
        double* qsnow_tendency_micro, double* qsnow_tendency,
        double* precip_rate, double* evap_rate, double* melt_rate,
        double* qt_iso_O18, double* qv_iso_O18, double* ql_iso_O18, double* qi_iso_O18, 
        double* qrain_iso_O18, double* qsnow_iso_O18,
        double* qrain_iso_O18_tendency, double* qrain_iso_O18_tendency_micro, 
        double* qsnow_iso_O18_tendency, double* qsnow_iso_O18_tendency_micro,
        double* precip_iso_rate_O18, double* evap_iso_rate_O18, 
        double* qt_iso_HDO, double* qv_iso_HDO, double* ql_iso_HDO, double* qi_iso_HDO, 
        double* qrain_iso_HDO, double* qsnow_iso_HDO,
        double* qrain_iso_HDO_tendency, double* qrain_iso_HDO_tendency_micro, 
        double* qsnow_iso_HDO_tendency, double* qsnow_iso_HDO_tendency_micro,
        double* precip_iso_rate_HDO, double* evap_iso_rate_HDO) nogil

cdef extern from "scalar_advection.h":
    void compute_advective_fluxes_a(Grid.DimStruct *dims, double *rho0, double*rho0_half, 
        double *velocity, double *scalar, double* flux, int d, int scheme) nogil

cdef extern from "isotope_functions.h":
    double q_2_delta(double q_iso, double q_std) nogil

cdef extern from "microphysics.h":
    void microphysics_stokes_sedimentation_velocity(Grid.DimStruct *dims, double* density, double ccn, double*  ql, double*  qt_velocity) nogil

cdef extern from "microphysics_sb.h":
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
    void sb_si_microphysics_sources(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
            double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
            double* density, double* p0, double* temperature,  double* qt, double ccn,
            double* ql, double* nr, double* qr, double* qisi, double* nisi, double dt, 
            double* nr_tendency_micro, double* qr_tendency_micro, double* nr_tendency, double* qr_tendency, 
            double* nisi_tendency_micro, double* qisi_tendency_micro, double* nisi_tendency, double* qisi_tendency,
            double* precip_rate, double* evap_rate, double* melt_rate) nogil
    void sb_si_qt_source_formation(Grid.DimStruct *dims, double* qisi_tendency, double* qr_tendency, double* qt_tendency)nogil
    void sb_sedimentation_velocity_ice(Grid.DimStruct *dims, double* nisi, double* qisi, double* density, double* nisi_velocity, 
            double* qisi_velocity) nogil

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
        
        # Prognostic variable: qt_iso_O18, total water isotopic specific humidity, defined as the ratio of isotopic mass of H2O18 to moist air.
        PV.add_variable('qt_iso_O18', 'kg/kg','qt_iso_O18_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_iso_O18', 'kg/kg','qv_iso_O18_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_iso_O18', 'kg/kg','ql_iso_O18_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        
        # Prognostic variable: qt_iso_HDO, total water isotopic specific humidity, defined as the ratio of isotopic mass of HDO to moist air.
        PV.add_variable('qt_iso_HDO', 'kg/kg','qt_iso_HDO_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_iso_HDO', 'kg/kg','qv_iso_HDO_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_iso_HDO', 'kg/kg','ql_iso_HDO_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)

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
            Py_ssize_t qt_iso_O18_shift = PV.get_varshift(Gr,'qt_iso_O18')
            Py_ssize_t qv_iso_O18_shift = PV.get_varshift(Gr,'qv_iso_O18')
            Py_ssize_t ql_iso_O18_shift = PV.get_varshift(Gr,'ql_iso_O18')
            Py_ssize_t qt_iso_HDO_shift = PV.get_varshift(Gr,'qt_iso_HDO')
            Py_ssize_t qv_iso_HDO_shift = PV.get_varshift(Gr,'qv_iso_HDO')
            Py_ssize_t ql_iso_HDO_shift = PV.get_varshift(Gr,'ql_iso_HDO')
            double[:] qv_std_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] ql_std_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qv_iso_O18_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] ql_iso_O18_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

        iso_equilibrium_fractionation_No_Microphysics(&Gr.dims, &DV.values[t_shift],
            &PV.values[qt_std_shift], &PV.values[qv_std_shift], &PV.values[ql_std_shift], 
            &PV.values[qt_iso_O18_shift], &PV.values[qv_iso_O18_shift], &PV.values[ql_iso_O18_shift], 
            &PV.values[qt_iso_HDO_shift], &PV.values[qv_iso_HDO_shift], &PV.values[ql_iso_HDO_shift], 
            &DV.values[qv_shift], &DV.values[ql_shift])
        return

    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        iso_stats_io_Base(Gr, PV, DV, Ref, NS, Pa)
        return

cdef class IsotopeTracers_SB_Liquid:
    def __init__(self, namelist):
        self.isotope_tracer = True
        return
    cpdef initialize(self, namelist, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
                    DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        Pa.root_print('initialized with IsotopeTracer with SB_Liquid scheme')
        
        # Prognostic variable: q_iso, isotopic specific humidity of qt, qv, ql and qr, defined as the ratio of isotopic mass of H2O18 to moist air.
        PV.add_variable('qt_iso_O18', 'kg/kg','qt_iso_O18_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_iso_O18', 'kg/kg','qv_iso_O18_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_iso_O18', 'kg/kg','ql_iso_O18_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qr_iso_O18', 'kg/kg','qr_iso_O18_isotope','Rain droplets water isotopic specific humidity','sym', 'scalar', Pa)

        # Prognostic variable: q_iso, isotopic specific humidity of qt, qv, ql and qr, defined as the ratio of isotopic mass of HDO to moist air.
        PV.add_variable('qt_iso_HDO', 'kg/kg','qt_iso_HDO_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_iso_HDO', 'kg/kg','qv_iso_HDO_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_iso_HDO', 'kg/kg','ql_iso_HDO_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qr_iso_HDO', 'kg/kg','qr_iso_HDO_isotope','Rain droplets water isotopic specific humidity','sym', 'scalar', Pa)

        PV.add_variable('qt_std', 'kg/kg','qt_std','Total water std specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_std', 'kg/kg','qv_std','Vapor water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_std', 'kg/kg','ql_std','Cloud liquid water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qr_std', 'kg/kg','ql_std','Rain water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('nr_std', 'kg/kg','ql_std','Rain water std specific humidity','sym', 'scalar', Pa)
        
        # sedimentation velocity of qt_iso_O18(w_qt_iso_O18) and qr_iso_O18(w_qr_iso_O18), which should be same as qt and qr, as DVs w_qt, w_qr 
        try:
            self.cloud_sedimentation = namelist['microphysics']['cloud_sedimentation']
        except:
            self.cloud_sedimentation = False
        
        if self.cloud_sedimentation:
            DV.add_variables('w_qt_iso_O18', 'm/s', r'w_{qt_iso_O18}', 'cloud liquid water isotopic sedimentation velocity', 'sym', Pa)
            DV.add_variables('w_qt_iso_HDO', 'm/s', r'w_{qt_iso_HDO}', 'cloud liquid water isotopic sedimentation velocity', 'sym', Pa)
            DV.add_variables('w_qt_std', 'm/s', r'w_{qt_iso_O18}', 'cloud liquid water std sedimentation velocity', 'sym', Pa)
            NS.add_profile('qt_std_sedimentation_flux', Gr, Pa, 'kg/kg', '', '')
        DV.add_variables('w_qr_iso_O18', 'm/s', r'w_{qr_iso_O18}', 'rain mass isotopic sedimentation veloctiy', 'sym', Pa)
        DV.add_variables('w_qr_iso_HDO', 'm/s', r'w_{qr_iso_HDO}', 'rain mass isotopic sedimentation veloctiy', 'sym', Pa)
        DV.add_variables('w_qr_std', 'm/s', r'w_{qr_iso_O18}', 'rain std mass sedimentation veloctiy', 'sym', Pa)
        DV.add_variables('w_nr_std', 'm/s', r'w_{qr_iso_O18}', 'rain std mass sedimentation veloctiy', 'sym', Pa)

        NS.add_profile('qr_std', Gr, Pa, 'kg/kg', '', 'stander water tarcer rain')
        NS.add_profile('qr_iso_O18', Gr, Pa, 'kg/kg', '', 'Finial result of rain isotopic sepcific humidity of H2O18')
        NS.add_profile('qr_iso_HDO', Gr, Pa, 'kg/kg', '', 'Finial result of rain isotopic sepcific humidity of HDO')

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

            Py_ssize_t qt_iso_O18_shift  = PV.get_varshift(Gr,'qt_iso_O18')
            Py_ssize_t qv_iso_O18_shift  = PV.get_varshift(Gr,'qv_iso_O18')
            Py_ssize_t ql_iso_O18_shift  = PV.get_varshift(Gr,'ql_iso_O18')
            Py_ssize_t qr_iso_O18_shift  = PV.get_varshift(Gr, 'qr_iso_O18')
            Py_ssize_t wqr_iso_O18_shift = DV.get_varshift(Gr, 'w_qr_iso_O18')

            Py_ssize_t qt_iso_HDO_shift  = PV.get_varshift(Gr,'qt_iso_HDO')
            Py_ssize_t qv_iso_HDO_shift  = PV.get_varshift(Gr,'qv_iso_HDO')
            Py_ssize_t ql_iso_HDO_shift  = PV.get_varshift(Gr,'ql_iso_HDO')
            Py_ssize_t qr_iso_HDO_shift  = PV.get_varshift(Gr, 'qr_iso_HDO')
            Py_ssize_t wqr_iso_HDO_shift = DV.get_varshift(Gr, 'w_qr_iso_HDO')
            Py_ssize_t wqt_std_shift
            Py_ssize_t wqt_std_O18_shift
            Py_ssize_t wqt_std_HDO_shift

            double[:] qv_std_tmp     = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] ql_std_tmp     = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qv_iso_O18_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] ql_iso_HDO_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

            double[:] qr_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] nr_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qr_iso_O18_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qr_iso_HDO_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

        iso_equilibrium_fractionation_No_Microphysics(&Gr.dims, &DV.values[t_shift],
            &PV.values[qt_shift], &DV.values[qv_shift], &DV.values[ql_shift], 
            &PV.values[qv_std_shift], &PV.values[ql_std_shift], 
            &PV.values[qt_iso_O18_shift], &PV.values[qv_iso_O18_shift], &PV.values[ql_iso_O18_shift], 
            &PV.values[qt_iso_HDO_shift], &PV.values[qv_iso_HDO_shift], &PV.values[ql_iso_HDO_shift])

        tracer_sb_liquid_microphysics_sources(&Gr.dims, &Micro_SB_Liquid.CC.LT.LookupStructC, 
            Micro_SB_Liquid.Lambda_fp, Micro_SB_Liquid.L_fp, 
            Micro_SB_Liquid.compute_rain_shape_parameter, Micro_SB_Liquid.compute_droplet_nu, 
            &Ref.rho0_half[0],  &Ref.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift], 
            Micro_SB_Liquid.ccn, &DV.values[ql_shift], &PV.values[nr_shift], &PV.values[qr_std_shift], TS.dt, 
            &nr_std_tend_micro[0], &qr_std_tend_micro[0], &PV.tendencies[nr_std_shift], &PV.tendencies[qr_std_shift],
            &PV.values[qr_iso_O18_shift], &PV.values[qt_iso_O18_shift], &PV.values[qv_iso_O18_shift], &PV.values[ql_iso_O18_shift],
            &PV.values[qr_iso_HDO_shift], &PV.values[qt_iso_HDO_shift], &PV.values[qv_iso_HDO_shift], &PV.values[ql_iso_HDO_shift],
            &qr_iso_O18_tend_micro[0], &PV.tendencies[qr_iso_O18_shift], &qr_iso_HDO_tend_micro[0], &PV.tendencies[qr_iso_HDO_shift])

        sb_sedimentation_velocity_rain(&Gr.dims, Micro_SB_Liquid.compute_rain_shape_parameter, 
            &Ref.rho0_half[0], &PV.values[nr_shift], &PV.values[qr_shift],
            &DV.values[wnr_shift], &DV.values[wqr_std_shift])
        sb_sedimentation_velocity_rain(&Gr.dims, Micro_SB_Liquid.compute_rain_shape_parameter, 
            &Ref.rho0_half[0], &PV.values[nr_shift], &PV.values[qr_shift],
            &DV.values[wnr_shift], &DV.values[wqr_iso_O18_shift])
        sb_sedimentation_velocity_rain(&Gr.dims, Micro_SB_Liquid.compute_rain_shape_parameter, 
            &Ref.rho0_half[0], &PV.values[nr_shift], &PV.values[qr_shift],
            &DV.values[wnr_shift], &DV.values[wqr_iso_HDO_shift])
        if self.cloud_sedimentation:
            wqt_std_shift = DV.get_varshift(Gr, 'w_qt_std')
            wqt_iso_O18_shift = DV.get_varshift(Gr, 'w_qt_iso_O18')
            wqt_iso_HDO_shift = DV.get_varshift(Gr, 'w_qt_iso_HDO')
            if self.stokes_sedimentation:
                microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_Liquid.ccn, &DV.values[ql_shift], &DV.values[wqt_std_shift])
                microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_Liquid.ccn, &DV.values[ql_shift], &DV.values[wqt_iso_O18_shift])
                microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_Liquid.ccn, &DV.values[ql_shift], &DV.values[wqt_iso_HDO_shift])
            else:
                sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_Liquid.ccn, &DV.values[ql_shift], &DV.values[wqt_std_shift])
                sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_Liquid.ccn, &DV.values[ql_shift], &DV.values[wqt_iso_O18_shift])
                sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_Liquid.ccn, &DV.values[ql_shift], &DV.values[wqt_iso_HDO_shift])
        sb_qt_source_formation(&Gr.dims,  &qr_std_tend_micro[0], &PV.tendencies[qt_std_shift])
        sb_qt_source_formation(&Gr.dims,  &qr_iso_O18_tend_micro[0], &PV.tendencies[qt_iso_O18_shift])
        sb_qt_source_formation(&Gr.dims,  &qr_iso_HDO_tend_micro[0], &PV.tendencies[qt_iso_HDO_shift])

        return 

    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
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
        
        PV.add_variable('qt_iso_O18', 'kg/kg','qt_iso_O18_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_iso_O18', 'kg/kg','qv_iso_O18_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_iso_O18', 'kg/kg','ql_iso_O18_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qi_iso_O18', 'kg/kg','qi_iso_O18_isotope','Cloud ice water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qrain_iso_O18', 'kg/kg','qrain_iso_O18','rain iso water tracer specific humidity','sym', "scalar", Pa)
        PV.add_variable('qsnow_iso_O18', 'kg/kg','qsnow_iso_O18','declaration','sym', "scalar", Pa)
        DV.add_variables('w_qrain_iso_O18', 'unit', r'w_qrain_iso_O18','declaration', 'sym', Pa)
        DV.add_variables('w_qsnow_iso_O18', 'unit', r'w_qsnow_iso_O18','declaration', 'sym', Pa)

        PV.add_variable('qt_iso_HDO', 'kg/kg','qt_iso_HDO_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_iso_HDO', 'kg/kg','qv_iso_HDO_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_iso_HDO', 'kg/kg','ql_iso_HDO_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qi_iso_HDO', 'kg/kg','qi_iso_HDO_isotope','Cloud ice water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qrain_iso_HDO', 'kg/kg','qrain_iso_HDO','rain iso water tracer specific humidity','sym', "scalar", Pa)
        PV.add_variable('qsnow_iso_HDO', 'kg/kg','qsnow_iso_HDO','declaration','sym', "scalar", Pa)
        DV.add_variables('w_qrain_iso_HDO', 'unit', r'w_qrain_iso_HDO','declaration', 'sym', Pa)
        DV.add_variables('w_qsnow_iso_HDO', 'unit', r'w_qsnow_iso_HDO','declaration', 'sym', Pa)

        initialize_NS_base(NS, Gr, Pa)
        
        NS.add_profile('qi_std_cloud', Gr, Pa, 'unit', '', 'qi_std_cloud')
        NS.add_profile('qi_iso_O18_cloud', Gr, Pa, 'unit', '', 'qi_iso_O18_cloud')
        NS.add_profile('ql_std_cloud', Gr, Pa, 'unit', '', 'ql_std_cloud')
        NS.add_profile('ql_iso_O18_cloud', Gr, Pa, 'unit', '', 'ql_iso_O18_cloud')
        NS.add_profile('qrain_std_domain', Gr, Pa, 'kg/kg', '', 'qrain_std_domain')
        NS.add_profile('qrain_iso_O18_domain', Gr, Pa, 'kg/kg', '', 'qrain_iso_O18_domain')
        NS.add_profile('qsnow_std_domain', Gr, Pa, 'kg/kg', '', 'qsnow_std_domain')
        NS.add_profile('qsnow_iso_O18_domain', Gr, Pa, 'kg/kg', '', 'qsnow_iso_O18_domain')
        NS.add_profile('qrain_std_cloud_domain', Gr, Pa, 'unit', '', 'qrain_std_cloud_domain')
        NS.add_profile('qrain_iso_O18_cloud_domain', Gr, Pa, 'unit', '', 'qrain_iso_O18_cloud_domain')
        NS.add_profile('qsnow_std_cloud_domain', Gr, Pa, 'unit', '', 'qsnow_std_cloud_domain')
        NS.add_profile('qsnow_iso_O18_cloud_domain', Gr, Pa, 'unit', '', 'qsnow_iso_O18_cloud_domain')
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

            Py_ssize_t qt_iso_O18_shift     = PV.get_varshift(Gr,'qt_iso_O18')
            Py_ssize_t qv_iso_O18_shift     = PV.get_varshift(Gr,'qv_iso_O18')
            Py_ssize_t ql_iso_O18_shift     = PV.get_varshift(Gr,'ql_iso_O18')
            Py_ssize_t qi_iso_O18_shift     = PV.get_varshift(Gr,'qi_iso_O18')
            Py_ssize_t qsnow_iso_O18_shift  = PV.get_varshift(Gr,'qsnow_iso_O18')
            Py_ssize_t qrain_iso_O18_shift  = PV.get_varshift(Gr,'qrain_iso_O18')
            Py_ssize_t wqrain_iso_O18_shift = DV.get_varshift(Gr,'w_qrain_iso_O18')
            Py_ssize_t wqsnow_iso_O18_shift = DV.get_varshift(Gr,'w_qsnow_iso_O18')

            Py_ssize_t qt_iso_HDO_shift     = PV.get_varshift(Gr,'qt_iso_HDO')
            Py_ssize_t qv_iso_HDO_shift     = PV.get_varshift(Gr,'qv_iso_HDO')
            Py_ssize_t ql_iso_HDO_shift     = PV.get_varshift(Gr,'ql_iso_HDO')
            Py_ssize_t qi_iso_HDO_shift     = PV.get_varshift(Gr,'qi_iso_HDO')
            Py_ssize_t qsnow_iso_HDO_shift  = PV.get_varshift(Gr,'qsnow_iso_HDO')
            Py_ssize_t qrain_iso_HDO_shift  = PV.get_varshift(Gr,'qrain_iso_HDO')
            Py_ssize_t wqrain_iso_HDO_shift = DV.get_varshift(Gr,'w_qrain_iso_HDO')
            Py_ssize_t wqsnow_iso_HDO_shift = DV.get_varshift(Gr,'w_qsnow_iso_HDO')
            
            double [:] qrain_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] qsnow_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] precip_rate_std = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] evap_rate_std = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] melt_rate_std = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

            double [:] qrain_iso_O18_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] qsnow_iso_O18_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] precip_rate_iso_O18 = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] evap_rate_iso_O18 = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

            double [:] qrain_iso_HDO_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] qsnow_iso_HDO_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] precip_rate_iso_HDO = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] evap_rate_iso_HDO = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

        iso_mix_phase_fractionation(&Gr.dims, &Micro_Arctic_1M.CC.LT.LookupStructC, 
            Micro_Arctic_1M.Lambda_fp, Micro_Arctic_1M.L_fp, &DV.values[t_shift], &Ref.p0_half[0],
            &PV.values[qt_std_shift], &PV.values[qv_std_shift], &PV.values[ql_std_shift], &PV.values[qi_std_shift], 
            &PV.values[qt_iso_O18_shift], &PV.values[qv_iso_O18_shift], &PV.values[ql_iso_O18_shift], &PV.values[qi_iso_O18_shift], 
            &PV.values[qt_iso_HDO_shift], &PV.values[qv_iso_HDO_shift], &PV.values[ql_iso_HDO_shift], &PV.values[qi_iso_HDO_shift], 
            &DV.values[qv_shift], &DV.values[ql_shift], &DV.values[qi_shift])

        tracer_arctic1m_microphysics_sources(&Gr.dims, &Micro_Arctic_1M.CC.LT.LookupStructC, Micro_Arctic_1M.Lambda_fp, 
            Micro_Arctic_1M.L_fp, &Ref.rho0_half[0],&Ref.p0_half[0], Micro_Arctic_1M.ccn, Micro_Arctic_1M.n0_ice_input, TS.dt,
            &DV.values[t_shift], &PV.values[qt_shift], &DV.values[qv_shift], &DV.values[ql_shift], &DV.values[qi_shift], 
            &PV.values[qrain_std_shift], &DV.values[nrain_shift], &PV.values[qsnow_std_shift], &DV.values[nsnow_shift], 
            &PV.values[ql_std_shift], &PV.values[qi_std_shift], &qrain_std_tend_micro[0], &PV.tendencies[qrain_std_shift],
            &qsnow_std_tend_micro[0], &PV.tendencies[qsnow_std_shift], &precip_rate_std[0], &evap_rate_std[0],&melt_rate_std[0],
            &PV.values[qt_iso_O18_shift], &PV.values[qv_iso_O18_shift], &PV.values[ql_iso_O18_shift], &PV.values[qi_iso_O18_shift], 
            &PV.values[qrain_iso_O18_shift], &PV.values[qsnow_iso_O18_shift], &PV.tendencies[qrain_iso_O18_shift], &qrain_iso_O18_tend_micro[0], 
            &PV.tendencies[qsnow_iso_O18_shift], &qsnow_iso_O18_tend_micro[0], &precip_rate_iso_O18[0], &evap_rate_iso_O18[0],
            &PV.values[qt_iso_HDO_shift], &PV.values[qv_iso_HDO_shift], &PV.values[ql_iso_HDO_shift], &PV.values[qi_iso_HDO_shift], 
            &PV.values[qrain_iso_HDO_shift], &PV.values[qsnow_iso_HDO_shift], &PV.tendencies[qrain_iso_HDO_shift], &qrain_iso_HDO_tend_micro[0], 
            &PV.tendencies[qsnow_iso_HDO_shift], &qsnow_iso_HDO_tend_micro[0], &precip_rate_iso_HDO[0], &evap_rate_iso_HDO[0])

        sedimentation_velocity_rain(&Gr.dims, &Ref.rho0_half[0], &DV.values[nrain_shift], &PV.values[qrain_shift],
            &DV.values[wqrain_std_shift])
        sedimentation_velocity_snow(&Gr.dims, &Ref.rho0_half[0], &DV.values[nsnow_shift], &PV.values[qsnow_shift],
            &DV.values[wqsnow_std_shift])

        sedimentation_velocity_rain(&Gr.dims, &Ref.rho0_half[0], &DV.values[nrain_shift], &PV.values[qrain_shift],
            &DV.values[wqrain_iso_O18_shift])
        sedimentation_velocity_snow(&Gr.dims, &Ref.rho0_half[0], &DV.values[nsnow_shift], &PV.values[qsnow_shift],
            &DV.values[wqsnow_iso_O18_shift])
        sedimentation_velocity_rain(&Gr.dims, &Ref.rho0_half[0], &DV.values[nrain_shift], &PV.values[qrain_shift],
            &DV.values[wqrain_iso_HDO_shift])
        sedimentation_velocity_snow(&Gr.dims, &Ref.rho0_half[0], &DV.values[nsnow_shift], &PV.values[qsnow_shift],
            &DV.values[wqsnow_iso_HDO_shift])

        qt_source_formation(&Gr.dims, &PV.tendencies[qt_std_shift], &precip_rate_std[0], &evap_rate_std[0])
        qt_source_formation(&Gr.dims, &PV.tendencies[qt_iso_O18_shift], &precip_rate_iso_O18[0], &evap_rate_iso_O18[0])
        qt_source_formation(&Gr.dims, &PV.tendencies[qt_iso_HDO_shift], &precip_rate_iso_HDO[0], &evap_rate_iso_HDO[0])

        return

    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
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
            Py_ssize_t qi_iso_O18_shift     = PV.get_varshift(Gr,'qi_iso_O18')
            Py_ssize_t ql_std_shift     = PV.get_varshift(Gr,'ql_std')
            Py_ssize_t ql_iso_O18_shift     = PV.get_varshift(Gr,'ql_iso_O18')
            
            Py_ssize_t qrain_std_shift  = PV.get_varshift(Gr,'qrain_std')
            Py_ssize_t qrain_iso_O18_shift  = PV.get_varshift(Gr,'qrain_iso_O18')
            Py_ssize_t qsnow_std_shift  = PV.get_varshift(Gr,'qsnow_std')
            Py_ssize_t qsnow_iso_O18_shift  = PV.get_varshift(Gr,'qsnow_iso_O18')
            
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

        # liquid cloud domain stats_io fo ql_std, and ql_iso_O18_shift
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[ql_std_shift], &cloud_liquid_mask[0])
        NS.write_profile('ql_std_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[ql_iso_O18_shift], &cloud_liquid_mask[0])
        NS.write_profile('ql_iso_O18_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qi_std_shift], &cloud_ice_mask[0])
        NS.write_profile('qi_std_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qi_iso_O18_shift], &cloud_ice_mask[0])
        NS.write_profile('qi_iso_O18_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        # rain domain stats_io of qrain_std, and qrain_iso_O18
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qrain_std_shift], &rain_mask[0])
        NS.write_profile('qrain_std_domain', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qrain_iso_O18_shift], &rain_mask[0])
        NS.write_profile('qrain_iso_O18_domain', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        
        # snow domain stats_io of qsnow_std, and qsnow_iso_O18
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qsnow_std_shift], &snow_mask[0])
        NS.write_profile('qsnow_std_domain', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qsnow_iso_O18_shift], &snow_mask[0])
        NS.write_profile('qsnow_iso_O18_domain', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        # liquid cloud domain stats_io fo qrain_std, and qrain_iso_O18_shift
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qrain_std_shift], &cloud_liquid_mask[0])
        NS.write_profile('qrain_std_cloud_domain', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qrain_iso_O18_shift], &cloud_liquid_mask[0])
        NS.write_profile('qrain_iso_O18_cloud_domain', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        
        # ice cloud domain stats_io fo qsnow_std, and qsnow_iso_O18_shift
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qsnow_std_shift], &cloud_ice_mask[0])
        NS.write_profile('qsnow_std_cloud_domain', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qsnow_iso_O18_shift], &cloud_ice_mask[0])
        NS.write_profile('qsnow_iso_O18_cloud_domain', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        
        return

cdef class IsotopeTracers_SBSI:
    def __init__(self, namelist):
        self.isotope_tracer = True
        return
    cpdef initialize(self, namelist, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
                    DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        Pa.root_print('initialized with IsotopeTracer with SBSI scheme')
        
        # Prognostic variable: q_iso, isotopic specific humidity of qt, qv, ql and qr, defined as the ratio of isotopic mass of H2O18 to moist air.
        PV.add_variable('qt_iso_O18', 'kg/kg','qt_iso_O18_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_iso_O18', 'kg/kg','qv_iso_O18_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_iso_O18', 'kg/kg','ql_iso_O18_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qr_iso_O18', 'kg/kg','qr_iso_O18_isotope','Rain droplets water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qisi_iso_O18', 'kg/kg','qr_iso_O18_isotope','Single Ice droplets water isotopic specific humidity','sym', 'scalar', Pa)

        PV.add_variable('qt_iso_HDO', 'kg/kg','qt_iso_HDO_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_iso_HDO', 'kg/kg','qv_iso_HDO_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_iso_HDO', 'kg/kg','ql_iso_HDO_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qr_iso_HDO', 'kg/kg','qr_iso_HDO_isotope','Rain droplets water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qisi_iso_HDO', 'kg/kg','qr_iso_HDO_isotope','Single Ice droplets water isotopic specific humidity','sym', 'scalar', Pa)

        PV.add_variable('qt_std', 'kg/kg','qt_std','Total water std specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_std', 'kg/kg','qv_std','Vapor water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_std', 'kg/kg','ql_std','Cloud liquid water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qr_std', 'kg/kg','ql_std','Rain water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qisi_std', 'kg/kg','ql_std','Rain water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('nr_std', '','nr_std','Rain water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('nisi_std', '','nisi_std','Single Ice water std specific humidity','sym', 'scalar', Pa)

        # following velocity calculation of rain and single-ice 
        # sedimentation velocity of qt_iso_O18(w_qt_iso_O18) and qr_iso_O18(w_qr_iso_O18),
        # which should be same as qt, qr and qisi, as DVs w_qt, w_qr, w_qisi;
        DV.add_variables('w_qr_std', 'unit', r'w_qr_std','declaration', 'sym', Pa)
        DV.add_variables('w_nr_std', 'unit', r'w_nr_std','declaration', 'sym', Pa)
        DV.add_variables('w_qisi_std', 'unit', r'w_qisi_std','declaration', 'sym', Pa)
        DV.add_variables('w_nisi_std', 'unit', r'w_nisi_std','declaration', 'sym', Pa)
        DV.add_variables('w_qr_iso_O18', 'unit', r'w_qrain_iso_O18','declaration', 'sym', Pa)
        DV.add_variables('w_qr_iso_HDO', 'unit', r'w_qrain_iso_O18','declaration', 'sym', Pa)
        DV.add_variables('w_nr_iso', 'unit', r'w_nr_iso','declaration', 'sym', Pa)
        DV.add_variables('w_qisi_iso_O18', 'unit', r'w_qsnow_iso_O18','declaration', 'sym', Pa)
        DV.add_variables('w_qisi_iso_HDO', 'unit', r'w_qsnow_iso_O18','declaration', 'sym', Pa)
        DV.add_variables('w_nisi_iso', 'unit', r'w_nisi_iso','declaration', 'sym', Pa)
        try:
            self.cloud_sedimentation = namelist['microphysics']['cloud_sedimentation']
        except:
            self.cloud_sedimentation = False
        
        if self.cloud_sedimentation:
            DV.add_variables('w_qt_iso_O18', 'm/s', r'w_{qt_iso_O18}', 'cloud liquid water isotopic sedimentation velocity', 'sym', Pa)
            DV.add_variables('w_qt_iso_HDO', 'm/s', r'w_{qt_iso_HDO}', 'cloud liquid water isotopic sedimentation velocity', 'sym', Pa)
            DV.add_variables('w_qt_std', 'm/s', r'w_{qt_iso_O18}', 'cloud liquid water std sedimentation velocity', 'sym', Pa)
            NS.add_profile('qt_std_sedimentation_flux', Gr, Pa, 'kg/kg', '', '')
            NS.add_profile('qt_iso_O18_sedimentation_flux', Gr, Pa, 'kg/kg', '', '')
            NS.add_profile('qt_iso_HDO_sedimentation_flux', Gr, Pa, 'kg/kg', '', '')

        NS.add_profile('qr_std', Gr, Pa, 'kg/kg', '', 'stander water tarcer rain')
        NS.add_profile('qr_iso_O18', Gr, Pa, 'kg/kg', '', 'Finial result of rain isotopic sepcific humidity')
        NS.add_profile('qr_iso_HDO', Gr, Pa, 'kg/kg', '', 'Finial result of rain isotopic sepcific humidity')
        NS.add_profile('qisi_std', Gr, Pa, 'kg/kg', '', 'stander water tarcer of single ice')
        NS.add_profile('qisi_iso_O18', Gr, Pa, 'kg/kg', '', 'Finial result of single ice isotopic sepcific humidity')
        NS.add_profile('qisi_iso_HDO', Gr, Pa, 'kg/kg', '', 'Finial result of single ice isotopic sepcific humidity')

        initialize_NS_base(NS, Gr, Pa)
        return

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref, 
        Microphysics_SB_SI.Microphysics_SB_SI Micro_SB_SI, ThermodynamicsSA.ThermodynamicsSA Th_sa, 
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
            
            Py_ssize_t qt_iso_O18_shift = PV.get_varshift(Gr,'qt_iso_O18')
            Py_ssize_t qv_iso_O18_shift = PV.get_varshift(Gr,'qv_iso_O18')
            Py_ssize_t ql_iso_O18_shift = PV.get_varshift(Gr,'ql_iso_O18')
            Py_ssize_t qt_iso_HDO_shift = PV.get_varshift(Gr,'qt_iso_HDO')
            Py_ssize_t qv_iso_HDO_shift = PV.get_varshift(Gr,'qv_iso_HDO')
            Py_ssize_t ql_iso_HDO_shift = PV.get_varshift(Gr,'ql_iso_HDO')
            Py_ssize_t wqt_std_shift

            # double[:] qv_std_tmp        = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            # double[:] ql_std_tmp        = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            # double[:] qv_iso_O18_tmp        = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            # double[:] ql_iso_O18_tmp        = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            # double[:] qv_iso_HDO_tmp        = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            # double[:] ql_iso_HDO_tmp        = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            #
            # double[:] qt_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            # double[:] qv_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            # double[:] ql_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            #
            # double[:] qt_iso_O18_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            # double[:] qv_iso_O18_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            # double[:] ql_iso_O18_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            # double[:] qt_iso_HDO_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            # double[:] qv_iso_HDO_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            # double[:] ql_iso_HDO_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
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
        #     &PV.values[qt_iso_O18_shift], &PV.values[qv_iso_O18_shift], &PV.values[ql_iso_O18_shift], 
        #     &PV.values[qt_iso_HDO_shift], &PV.values[qv_iso_HDO_shift], &PV.values[ql_iso_HDO_shift], 
        #     &DV.values[qv_shift], &DV.values[ql_shift])

        sb_si_microphysics_sources(&Gr.dims, &Micro_SB_SI.CC.LT.LookupStructC, Micro_SB_SI.Lambda_fp, Micro_SB_SI.L_fp, 
            Micro_SB_SI.compute_rain_shape_parameter, Micro_SB_SI.compute_droplet_nu, 
            &Ref.rho0_half[0],  &Ref.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift], Micro_SB_SI.ccn, 
            &DV.values[ql_shift], &PV.values[nr_shift], &PV.values[qr_std_shift], &PV.values[qisi_std_shift], &PV.values[nisi_std_shift], TS.dt,   
            &nr_std_tend_micro[0], &qr_std_tend_micro[0], &PV.tendencies[nr_std_shift], &PV.tendencies[qr_std_shift],
            &nisi_std_tend_micro[0], &qisi_std_tend_micro[0], &PV.tendencies[nisi_std_shift], &PV.tendencies[qisi_std_shift],
            &precip_rate[0], &evap_rate[0], &melt_rate[0])

        sb_si_qt_source_formation(&Gr.dims, &qisi_std_tend_micro[0], &qr_std_tend_micro[0], &PV.tendencies[qt_shift])

        # sedimentation processes of rain and single_ice: w_qr and w_qisi
        sb_sedimentation_velocity_rain(&Gr.dims, Micro_SB_SI.compute_rain_shape_parameter, &Ref.rho0_half[0], &PV.values[nr_std_shift],
            &PV.values[qr_std_shift], &DV.values[wnr_std_shift], &DV.values[wqr_std_shift])
        sb_sedimentation_velocity_ice(&Gr.dims, &PV.values[nisi_std_shift], &PV.values[qisi_std_shift], &Ref.rho0_half[0], 
            &DV.values[wnisi_std_shift], &DV.values[wqisi_std_shift])

        if self.cloud_sedimentation:
            wqt_std_shift = DV.get_varshift(Gr, 'w_qt_std')
            if self.stokes_sedimentation:
                microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_SI.ccn, &DV.values[ql_shift], &DV.values[wqt_std_shift])
            else:
                sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], Micro_SB_SI.ccn, &DV.values[ql_shift], &DV.values[wqt_std_shift])
        return 

    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
            ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        iso_stats_io_Base(Gr, PV, DV, Ref, NS, Pa)
        return

cpdef iso_stats_io_Base(Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
            ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
    cdef:
        Py_ssize_t imin    = 0
        Py_ssize_t jmin    = 0
        Py_ssize_t kmin    = 0
        Py_ssize_t imax    = Gr.dims.nlg[0]
        Py_ssize_t jmax    = Gr.dims.nlg[1]
        Py_ssize_t kmax    = Gr.dims.nlg[2]
        Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        Py_ssize_t jstride = Gr.dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k, iter = 0
        Py_ssize_t ql_shift     = DV.get_varshift(Gr, 'ql')
        Py_ssize_t qt_std_shift = PV.get_varshift(Gr,'qt_std')
        Py_ssize_t qv_std_shift = PV.get_varshift(Gr,'qv_std')
        Py_ssize_t ql_std_shift = PV.get_varshift(Gr,'ql_std')
        Py_ssize_t qt_iso_O18_shift = PV.get_varshift(Gr,'qt_iso_O18')
        Py_ssize_t qv_iso_O18_shift = PV.get_varshift(Gr,'qv_iso_O18')
        Py_ssize_t ql_iso_O18_shift = PV.get_varshift(Gr,'ql_iso_O18')
        double[:] tmp
        double[:] delta_qv       = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        double[:] delta_qt       = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        double[:] delta_ql_cloud = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        double[:] cloud_mask     = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        double[:] no_cloud_mask  = np.zeros(Gr.dims.npg, dtype=np.double, order='c')

    with nogil:
        with cython.boundscheck(False):
            for i in xrange(imin,imax):
                ishift = i*istride
                for j in xrange(jmin,jmax):
                    jshift = j*jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        if DV.values[ql_shift + ijk] > 0.0:
                            cloud_mask[ijk] = 1.0
                            delta_ql_cloud[ijk] = q_2_delta(PV.values[ql_iso_O18_shift + ijk], PV.values[ql_std_shift + ijk])
                        else:
                            no_cloud_mask[ijk] = 1.0 
                        delta_qv[ijk] = q_2_delta(PV.values[qv_iso_O18_shift + ijk], PV.values[qv_std_shift + ijk])
                        delta_qt[ijk] = q_2_delta(PV.values[qt_iso_O18_shift + ijk], PV.values[qt_std_shift + ijk])


    tmp = Pa.HorizontalMean(Gr, &PV.values[qt_std_shift])
    NS.write_profile('qt_std', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
    
    tmp = Pa.HorizontalMean(Gr, &DV.values[qv_std_shift])
    NS.write_profile('qv_std', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
    
    tmp = Pa.HorizontalMean(Gr, &DV.values[ql_std_shift])
    NS.write_profile('ql_std', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

    tmp = Pa.HorizontalMean(Gr, &PV.values[qt_iso_O18_shift])
    statsIO_isotope_scaling_magnitude(&Gr.dims, &tmp[0]) # scaling back to correct magnitude
    NS.write_profile('qt_iso_O18', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

    tmp = Pa.HorizontalMean(Gr, &PV.values[ql_iso_O18_shift])
    statsIO_isotope_scaling_magnitude(&Gr.dims, &tmp[0]) # scaling back to correct magnitude
    NS.write_profile('ql_iso_O18', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

    tmp = Pa.HorizontalMean(Gr, &PV.values[qv_iso_O18_shift])
    statsIO_isotope_scaling_magnitude(&Gr.dims, &tmp[0]) # scaling back to correct magnitude
    NS.write_profile('qv_iso_O18', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
    
    tmp = Pa.HorizontalMean(Gr, &delta_qv[0])
    NS.write_profile('delta_qv', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)   

    tmp = Pa.HorizontalMean(Gr, &delta_qt[0])
    NS.write_profile('delta_qt', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)   
    
    tmp = Pa.HorizontalMeanConditional(Gr, &delta_ql_cloud[0], &cloud_mask[0])
    NS.write_profile('delta_ql_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)   
        
    tmp = Pa.HorizontalMeanConditional(Gr, &delta_qv[0], &cloud_mask[0])
    NS.write_profile('delta_qv_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)   

    tmp = Pa.HorizontalMeanConditional(Gr, &delta_qv[0], &no_cloud_mask[0])
    NS.write_profile('delta_qv_Nocloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)   
    return
cpdef initialize_NS_base(NetCDFIO_Stats NS, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):
    NS.add_profile('qt_std', Gr, Pa, 'kg/kg', '', 'stander water tracer total')
    NS.add_profile('qv_std', Gr, Pa, 'kg/kg', '', 'stander water tracer vapor')
    NS.add_profile('ql_std', Gr, Pa, 'kg/kg', '', 'stander water tracer liquid')
    NS.add_profile('qt_iso_O18', Gr, Pa, 'kg/kg', '', 'Finial result of total water isotopic specific humidity')
    NS.add_profile('qv_iso_O18', Gr, Pa, 'kg/kg', '', 'Finial result of vapor isotopic specific humidity')
    NS.add_profile('ql_iso_O18', Gr, Pa, 'kg/kg', '', 'Finial result of liquid isotopic sepcific humidity')
    
    NS.add_profile('qr_iso_O18_evap_tendency', Gr, Pa, 'kg/kg', '', '')
    NS.add_profile('qr_std_evap_tendency', Gr, Pa, 'kg/kg', '', '')
    NS.add_profile('qr_std_auto_tendency', Gr, Pa, 'kg/kg', '', '')
    NS.add_profile('qr_std_accre_tendency', Gr, Pa, 'kg/kg', '', '')
    NS.add_profile('qr_tend_micro_diff', Gr, Pa, 'kg/kg', '', '')
    NS.add_profile('qt_tendencies_diff', Gr, Pa, 'kg/kg', '', '')
    NS.add_profile('qr_std_sedimentation_flux', Gr, Pa, 'kg/kg', '', '')

    NS.add_profile('delta_qt', Gr, Pa, 'permil', '', 'delta of qt, calculated by qt_iso_O18/qt_std during fractioantion')
    NS.add_profile('delta_qv', Gr, Pa, 'permil', '', 'delta of qv, calculated by qt_iso_O18/qt_std during fractioantion')
    NS.add_profile('delta_ql_cloud', Gr, Pa, 'permil', '', 'delta of ql in cloud, calculated by qt_iso_O18/qt_std during fractioantion')
    NS.add_profile('delta_qv_cloud', Gr, Pa, 'permil', '', 'delta of qv in cloud, calculated by qt_iso_O18/qt_std during fractioantion')
    NS.add_profile('delta_qv_Nocloud', Gr, Pa, 'permil', '', 'delta of qv in cloud, calculated by qt_iso_O18/qt_std during fractioantion')
    return
