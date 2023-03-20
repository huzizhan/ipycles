cimport Grid
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ReferenceState
cimport ParallelMPI
cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from Thermodynamics cimport LatentHeat, ClausiusClapeyron

cdef class Microphysics_SB_SI:
    # Make the thermodynamics_type member available from Python-Space
    cdef public:
        str thermodynamics_type

    cdef:
        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil
        ClausiusClapeyron CC

        double (*compute_rain_shape_parameter)(double density, double qr, double Dm) nogil
        double (*compute_droplet_nu)(double density, double ql) nogil

        double ccn
        
        Py_ssize_t order
        bint cloud_sedimentation
        bint stokes_sedimentation
        
        double [:] evap_rate
        double [:] precip_rate
        double [:] melt_rate

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)

cdef inline double lambda_constant(double T) nogil:
    return 1.0

cdef inline double latent_heat_variable(double T, double Lambda) nogil:
    cdef:
        double TC = T - 273.15
    return (2500.8 - 2.36 * TC + 0.0016 * TC *
            TC - 0.00006 * TC * TC * TC) * 1000.0

# adopte from arctic-1m scheme 
cdef inline double latent_heat_Arctic(double T, double Lambda) nogil:
    cdef:
        double Lv = 2.501e6
        double Ls = 2.8334e6
    return (Lv * Lambda) + (Ls * (1.0 - Lambda))

# adopte from arctic-1m scheme 
cdef inline double latent_heat_variable_Arctic(double T, double Lambda) nogil:
    cdef:
        double TC = T - 273.15
    return (2500.8 - 2.36 * TC + 0.0016 * TC *
            TC - 0.00006 * TC * TC * TC) * 1000.0
