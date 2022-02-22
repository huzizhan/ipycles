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
        double *qt_std, double *qv_std, double *ql_std,
        double *qt_iso, double *qv_iso, double *ql_iso,
        double *qv_DV, double *ql_DV) nogil
    void delta_isotopologue(Grid.DimStruct *dims, double *q_iso, double *q_std, double *delta, int index) nogil
    void compute_sedimentaion(Grid.DimStruct *dims, double *w_q, double *w_q_iso, double *w_q_std) nogil
    void tracer_constrain_NoMicro(Grid.DimStruct *dims, double *ql, double *ql_std, double *ql_iso, double *qv_std, double *qv_iso, double *qt_std, double *qt_iso) nogil
cdef extern from "scalar_advection.h":
    void compute_advective_fluxes_a(Grid.DimStruct *dims, double *rho0, double *rho0_half, double *velocity, double *scalar, double* flux, int d, int scheme) nogil
cdef extern from "isotope_functions.h":
    double q_2_delta(double q_iso, double q_std) nogil
def IsotopeTracersFactory(namelist):
    try:
        micro_scheme = namelist['microphysics']['scheme']
        if micro_scheme == 'None_SA':
            return IsotopeTracers_NoMicrophysics(namelist)
        elif micro_scheme == 'SB_Liquid':
            return IsotopeTracers_SB_Liquid(namelist)
    except:
        return IsotopeTracersNone()

cdef class IsotopeTracersNone:
    def __init__(self):
        return
    cpdef initialize(self, namelist, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        Pa.root_print('initialized with IsotopeNone')
        return
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState RS, 
					ThermodynamicsSA.ThermodynamicsSA Th_sa, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        return
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   ReferenceState.ReferenceState RS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return

cdef class IsotopeTracers_NoMicrophysics:
    def __init__(self, namelist):
        
        return
    cpdef initialize(self, namelist, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
                    DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        Pa.root_print('initialized with IsotopeTracer with No Microphysics')

        # Prognostic variable: standerd water std of qt, ql and qv, which are totally same as qt, ql and qv 
        PV.add_variable('qt_std', 'kg/kg','qt_std','Total water std specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_std', 'kg/kg','qv_std','Vapor water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_std', 'kg/kg','ql_std','Cloud liquid water std specific humidity','sym', 'scalar', Pa)
        
        # Prognostic variable: qt_iso, total water isotopic specific humidity, defined as the ratio of isotopic mass of H2O18 to moist air.
        PV.add_variable('qt_iso', 'kg/kg','qt_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_iso', 'kg/kg','qv_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_iso', 'kg/kg','ql_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)

        initialize_NS_base(NS, Gr, Pa)
        # finial output results after selection and scaling
        return
        
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState RS, ThermodynamicsSA.ThermodynamicsSA Th_sa,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t t_shift = DV.get_varshift(Gr,'temperature')
            Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')
            Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t s_shift = PV.get_varshift(Gr,'s')
            Py_ssize_t qt_std_shift = PV.get_varshift(Gr,'qt_std')
            Py_ssize_t qv_std_shift = PV.get_varshift(Gr,'qv_std')
            Py_ssize_t ql_std_shift = PV.get_varshift(Gr,'ql_std')
            Py_ssize_t qt_iso_shift = PV.get_varshift(Gr,'qt_iso')
            Py_ssize_t qv_iso_shift = PV.get_varshift(Gr,'qv_iso')
            Py_ssize_t ql_iso_shift = PV.get_varshift(Gr,'ql_iso')
            double[:] qv_std_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] ql_std_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qv_iso_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] ql_iso_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

        iso_equilibrium_fractionation_No_Microphysics(&Gr.dims, &DV.values[t_shift],
                &PV.values[qt_std_shift], &PV.values[qv_std_shift], &PV.values[ql_std_shift], 
                &PV.values[qt_iso_shift], &PV.values[qv_iso_shift], &PV.values[ql_iso_shift], 
                &DV.values[qv_shift], &DV.values[ql_shift])
        return

    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   ReferenceState.ReferenceState RS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        iso_stats_io_Base(Gr, PV, DV, RS, NS, Pa)
        return

cdef class IsotopeTracers_SB_Liquid:
    def __init__(self, namelist):
        return
    cpdef initialize(self, namelist, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
                    DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        Pa.root_print('initialized with IsotopeTracer with SB_Liquid scheme')
        
        # Prognostic variable: q_iso, isotopic specific humidity of qt, qv, ql and qr, defined as the ratio of isotopic mass of H2O18 to moist air.
        PV.add_variable('qt_iso', 'kg/kg','qt_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_iso', 'kg/kg','qv_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_iso', 'kg/kg','ql_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qr_iso', 'kg/kg','qr_isotope','Rain droplets water isotopic specific humidity','sym', 'scalar', Pa)

        PV.add_variable('qt_std', 'kg/kg','qt_std','Total water std specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_std', 'kg/kg','qv_std','Vapor water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_std', 'kg/kg','ql_std','Cloud liquid water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qr_std', 'kg/kg','ql_std','Rain water std specific humidity','sym', 'scalar', Pa)
        PV.add_variable('nr_std', 'kg/kg','ql_std','Rain water std specific humidity','sym', 'scalar', Pa)
        
        # sedimentation velocity of qt_iso(w_qt_iso) and qr_iso(w_qr_iso), which should be same as qt and qr, as DVs w_qt, w_qr 
        try:
            self.cloud_sedimentation = namelist['microphysics']['cloud_sedimentation']
        except:
            self.cloud_sedimentation = False
        
        if self.cloud_sedimentation:
            DV.add_variables('w_qt_iso', 'm/s', r'w_{qt_iso}', 'cloud liquid water isotopic sedimentation velocity', 'sym', Pa)
            DV.add_variables('w_qt_std', 'm/s', r'w_{qt_iso}', 'cloud liquid water std sedimentation velocity', 'sym', Pa)
            NS.add_profile('qt_std_sedimentation_flux', Gr, Pa, 'kg/kg', '', '')
        DV.add_variables('w_qr_iso', 'm/s', r'w_{qr_iso}', 'rain mass isotopic sedimentation veloctiy', 'sym', Pa)
        DV.add_variables('w_qr_std', 'm/s', r'w_{qr_iso}', 'rain std mass sedimentation veloctiy', 'sym', Pa)
        DV.add_variables('w_nr_std', 'm/s', r'w_{qr_iso}', 'rain std mass sedimentation veloctiy', 'sym', Pa)

        NS.add_profile('qr_std', Gr, Pa, 'kg/kg', '', 'stander water tarcer rain')
        NS.add_profile('qr_iso', Gr, Pa, 'kg/kg', '', 'Finial result of rain isotopic sepcific humidity')

        initialize_NS_base(NS, Gr, Pa)
        return
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState RS, ThermodynamicsSA.ThermodynamicsSA Th_sa,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t t_shift = DV.get_varshift(Gr,'temperature')
            Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')
            Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t qr_shift = PV.get_varshift(Gr,'qr')
            Py_ssize_t nr_shift = PV.get_varshift(Gr,'nr')
            Py_ssize_t s_shift = PV.get_varshift(Gr,'s')
            Py_ssize_t alpha_shift = DV.get_varshift(Gr, 'alpha')
            Py_ssize_t qt_std_shift = PV.get_varshift(Gr,'qt_std')
            Py_ssize_t qv_std_shift = PV.get_varshift(Gr,'qv_std')
            Py_ssize_t ql_std_shift = PV.get_varshift(Gr,'ql_std')
            Py_ssize_t qr_std_shift = PV.get_varshift(Gr,'qr_std')
            Py_ssize_t nr_std_shift = PV.get_varshift(Gr,'nr_std')
            Py_ssize_t qt_iso_shift = PV.get_varshift(Gr,'qt_iso')
            Py_ssize_t qv_iso_shift = PV.get_varshift(Gr,'qv_iso')
            Py_ssize_t ql_iso_shift = PV.get_varshift(Gr,'ql_iso')
            Py_ssize_t qr_iso_shift = PV.get_varshift(Gr,'qr_iso')
            double[:] qv_std_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] ql_std_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qv_iso_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] ql_iso_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

            double[:] qr_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qt_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qv_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] ql_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] nr_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] nr_tend_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

            double[:] qr_iso_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qt_iso_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qv_iso_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] ql_iso_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

        iso_equilibrium_fractionation_No_Microphysics(&Gr.dims, &DV.values[t_shift],
                &PV.values[qt_std_shift], &PV.values[qv_std_shift], &PV.values[ql_std_shift], 
                &PV.values[qt_iso_shift], &PV.values[qv_iso_shift], &PV.values[ql_iso_shift], 
                &DV.values[qv_shift], &DV.values[ql_shift])
        return 

    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   ReferenceState.ReferenceState RS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        iso_stats_io_Base(Gr, PV, DV, RS, NS, Pa)
        return

cpdef iso_stats_io_Base(Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
            ReferenceState.ReferenceState RS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
    cdef:
        Py_ssize_t imin = 0
        Py_ssize_t jmin = 0
        Py_ssize_t kmin = 0
        Py_ssize_t imax = Gr.dims.nlg[0]
        Py_ssize_t jmax = Gr.dims.nlg[1]
        Py_ssize_t kmax = Gr.dims.nlg[2]
        Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        Py_ssize_t jstride = Gr.dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k, iter = 0
        double [:] tmp
        Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
        Py_ssize_t qt_std_shift = PV.get_varshift(Gr,'qt_std')
        Py_ssize_t qv_std_shift = PV.get_varshift(Gr,'qv_std')
        Py_ssize_t ql_std_shift = PV.get_varshift(Gr,'ql_std')
        Py_ssize_t qt_iso_shift = PV.get_varshift(Gr,'qt_iso')
        Py_ssize_t qv_iso_shift = PV.get_varshift(Gr,'qv_iso')
        Py_ssize_t ql_iso_shift = PV.get_varshift(Gr,'ql_iso')
        double[:] delta_qv = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        double[:] delta_qt = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        double[:] delta_ql_cloud = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        double[:] cloud_mask = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        double[:] no_cloud_mask = np.zeros(Gr.dims.npg, dtype=np.double, order='c')

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
                            delta_ql_cloud[ijk] = q_2_delta(PV.values[ql_iso_shift + ijk], PV.values[ql_std_shift + ijk])
                        else:
                            no_cloud_mask[ijk] = 1.0 
                        delta_qv[ijk] = q_2_delta(PV.values[qv_iso_shift + ijk], PV.values[qv_std_shift + ijk])
                        delta_qt[ijk] = q_2_delta(PV.values[qt_iso_shift + ijk], PV.values[qt_std_shift + ijk])

    tmp = Pa.HorizontalMean(Gr, &PV.values[qt_std_shift])
    NS.write_profile('qt_std', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
    
    tmp = Pa.HorizontalMean(Gr, &DV.values[qv_std_shift])
    NS.write_profile('qv_std', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
    
    tmp = Pa.HorizontalMean(Gr, &DV.values[ql_std_shift])
    NS.write_profile('ql_std', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

    tmp = Pa.HorizontalMean(Gr, &PV.values[qt_iso_shift])
    statsIO_isotope_scaling_magnitude(&Gr.dims, &tmp[0]) # scaling back to correct magnitude
    NS.write_profile('qt_iso', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

    tmp = Pa.HorizontalMean(Gr, &PV.values[ql_iso_shift])
    statsIO_isotope_scaling_magnitude(&Gr.dims, &tmp[0]) # scaling back to correct magnitude
    NS.write_profile('ql_iso', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

    tmp = Pa.HorizontalMean(Gr, &PV.values[qv_iso_shift])
    statsIO_isotope_scaling_magnitude(&Gr.dims, &tmp[0]) # scaling back to correct magnitude
    NS.write_profile('qv_iso', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
    
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
        NS.add_profile('qt_iso', Gr, Pa, 'kg/kg', '', 'Finial result of total water isotopic specific humidity')
        NS.add_profile('qv_iso', Gr, Pa, 'kg/kg', '', 'Finial result of vapor isotopic specific humidity')
        NS.add_profile('ql_iso', Gr, Pa, 'kg/kg', '', 'Finial result of liquid isotopic sepcific humidity')
        
        NS.add_profile('qr_iso_evap_tendency', Gr, Pa, 'kg/kg', '', '')
        NS.add_profile('qr_std_evap_tendency', Gr, Pa, 'kg/kg', '', '')
        NS.add_profile('qr_std_auto_tendency', Gr, Pa, 'kg/kg', '', '')
        NS.add_profile('qr_std_accre_tendency', Gr, Pa, 'kg/kg', '', '')
        NS.add_profile('qr_tend_micro_diff', Gr, Pa, 'kg/kg', '', '')
        NS.add_profile('qt_tendencies_diff', Gr, Pa, 'kg/kg', '', '')
        NS.add_profile('qr_std_sedimentation_flux', Gr, Pa, 'kg/kg', '', '')

        NS.add_profile('delta_qt', Gr, Pa, 'permil', '', 'delta of qt, calculated by qt_iso/qt_std during fractioantion')
        NS.add_profile('delta_qv', Gr, Pa, 'permil', '', 'delta of qv, calculated by qt_iso/qt_std during fractioantion')
        NS.add_profile('delta_ql_cloud', Gr, Pa, 'permil', '', 'delta of ql in cloud, calculated by qt_iso/qt_std during fractioantion')
        NS.add_profile('delta_qv_cloud', Gr, Pa, 'permil', '', 'delta of qv in cloud, calculated by qt_iso/qt_std during fractioantion')
        NS.add_profile('delta_qv_Nocloud', Gr, Pa, 'permil', '', 'delta of qv in cloud, calculated by qt_iso/qt_std during fractioantion')