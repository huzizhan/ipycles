#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport numpy as np
import numpy as np
cimport Lookup
cimport ParallelMPI
from Microphysics_Arctic_1M cimport Microphysics_Arctic_1M
from Microphysics_SB_Liquid cimport Microphysics_SB_Liquid
from Microphysics_SB_SI cimport Microphysics_SB_SI
from Microphysics_SB_2M cimport Microphysics_SB_2M, No_Microphysics_SB
cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
from libc.math cimport fmax, fmin, fabs
include 'parameters.pxi'

cdef extern from "microphysics.h":
    void microphysics_stokes_sedimentation_velocity(Grid.DimStruct *dims, double* density, double ccn, double*  ql, double*  qt_velocity)
cdef extern from "scalar_advection.h":
    void compute_advective_fluxes_a(Grid.DimStruct *dims, double *rho0, double *rho0_half, double *velocity, double *scalar, double* flux, int d, int scheme) nogil
    void compute_qt_sedimentation_s_source(Grid.DimStruct *dims, double *p0_half,  double* rho0_half, double *flux,
        double* qt, double* qv, double* T, double* tendency, double (*lam_fp)(double),
        double (*L_fp)(double, double), double dx, ssize_t d)nogil

cdef extern from "microphysics_sb.h":
    void sb_sedimentation_velocity_liquid(Grid.DimStruct *dims, double*  density, double ccn, double* ql, double* qt_velocity)nogil

cdef class No_Microphysics_Dry:
    def __init__(self, ParallelMPI.ParallelMPI Par, LatentHeat LH, namelist):
        LH.Lambda_fp = lambda_constant
        LH.L_fp = latent_heat_constant
        self.thermodynamics_type = 'dry'
        return
    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS,ParallelMPI.ParallelMPI Pa):
        return
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return

cdef class No_Microphysics_SA:
    def __init__(self, ParallelMPI.ParallelMPI Par, LatentHeat LH, namelist):
        LH.Lambda_fp = lambda_constant
        LH.L_fp = latent_heat_variable
        self.thermodynamics_type = 'SA'
        #also set local versions
        self.Lambda_fp = lambda_constant
        self.L_fp = latent_heat_variable

        # Extract case-specific parameter values from the namelist
        # Get number concentration of cloud condensation nuclei (1/m^3)
        try:
            self.ccn = namelist['microphysics']['ccn']
        except:
            self.ccn = 100.0e6
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

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        if self.cloud_sedimentation:
            DV.add_variables('w_qt', 'm/s', r'w_ql', 'cloud liquid water sedimentation velocity', 'sym', Pa)
            NS.add_profile('qt_sedimentation_flux', Gr, Pa)
            NS.add_profile('s_qt_sedimentation_source',Gr,Pa)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS,ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t wqt_shift
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
        if self.cloud_sedimentation:
            wqt_shift = DV.get_varshift(Gr, 'w_qt')

            if self.stokes_sedimentation:
                microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], self.ccn, &DV.values[ql_shift], &DV.values[wqt_shift])
            else:
                sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], self.ccn, &DV.values[ql_shift], &DV.values[wqt_shift])
        return
        
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, PrognosticVariables.PrognosticVariables PV, 
            DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t gw = Gr.dims.gw
            double[:] dummy =  np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
            Py_ssize_t qv_shift = DV.get_varshift(Gr, 'qv')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t s_shift  = PV.get_varshift(Gr, 's')
            Py_ssize_t wqt_shift
            double[:] s_src =  np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] tmp

        if self.cloud_sedimentation:
            wqt_shift = DV.get_varshift(Gr,'w_qt')

            compute_advective_fluxes_a(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &DV.values[wqt_shift], &DV.values[ql_shift], &dummy[0], 2, self.order)
            tmp       = Pa.HorizontalMean(Gr, &dummy[0])
            NS.write_profile('qt_sedimentation_flux', tmp[gw:-gw], Pa)

            compute_qt_sedimentation_s_source(&Gr.dims, &Ref.p0_half[0], &Ref.rho0_half[0], &dummy[0],
                                    &PV.values[qt_shift], &DV.values[qv_shift],&DV.values[t_shift], &s_src[0], self.Lambda_fp,
                                    self.L_fp, Gr.dims.dx[2], 2)
            tmp       = Pa.HorizontalMean(Gr, &s_src[0])
            NS.write_profile('s_qt_sedimentation_source', tmp[gw:-gw], Pa)
        return


def MicrophysicsFactory(namelist, LatentHeat LH, ParallelMPI.ParallelMPI Par):
    if(namelist['microphysics']['scheme'] == 'None_Dry'):
        return No_Microphysics_Dry(Par, LH, namelist)
    elif(namelist['microphysics']['scheme'] == 'None_SA'):
        return No_Microphysics_SA(Par, LH, namelist)
    elif(namelist['microphysics']['scheme'] == 'None_SB'):
        return No_Microphysics_SB(Par, LH, namelist)
    elif(namelist['microphysics']['scheme'] == 'SB_Liquid'):
        return Microphysics_SB_Liquid(Par, LH, namelist)
    elif(namelist['microphysics']['scheme'] == 'Arctic_1M'):
        return Microphysics_Arctic_1M(Par, LH, namelist)
    elif(namelist['microphysics']['scheme'] == 'SB_SI'):
        return Microphysics_SB_SI(Par, LH, namelist)
    elif(namelist['microphysics']['scheme'] == 'SB_2M'):
        return Microphysics_SB_2M(Par, LH, namelist)
