cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport Microphysics
cimport Microphysics_SB_Liquid
cimport Microphysics_Arctic_1M
cimport Microphysics_SB_SI
cimport ParallelMPI
cimport TimeStepping
cimport ReferenceState
cimport Lookup
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
cimport ThermodynamicsSA
cimport TimeStepping

from libc.math cimport log, exp
from NetCDFIO cimport NetCDFIO_Stats
import cython

cimport numpy as np
import numpy as np
include "parameters.pxi"

import cython

cdef class IsotopeTracersNone:

    cpdef initialize(self, namelist, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV,
            DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref,
            Microphysics.No_Microphysics_Dry Micro_Dry, ThermodynamicsSA.ThermodynamicsSA Th_sa, 
            DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
            ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class IsotopeTracers_NoMicrophysics:

    cpdef initialize(namelist, self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
            DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref, 
            Microphysics.No_Microphysics_SA Micro_SA, ThermodynamicsSA.ThermodynamicsSA Th_sa, 
            DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, 
            ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class IsotopeTracers_SB_Liquid:
    
    cdef:
        bint cloud_sedimentation

    cpdef initialize(namelist, self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
            DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref, 
            Microphysics_SB_Liquid.Microphysics_SB_Liquid Micro_SB_Liquid, ThermodynamicsSA.ThermodynamicsSA Th_sa, 
            DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, 
            ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class IsotopeTracers_Arctic_1M:

    cdef:
        bint cloud_sedimentation

    cpdef initialize(namelist, self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
            DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref, 
            Microphysics_Arctic_1M.Microphysics_Arctic_1M Micro_Arctic_1M, ThermodynamicsSA.ThermodynamicsSA Th_sa, 
            DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, 
            ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class IsotopeTracers_SBSI:
    
    cdef:
        bint cloud_sedimentation

    cpdef initialize(namelist, self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
            DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref, 
            Microphysics_SB_SI.Microphysics_SB_SI Micro_SB_SI, ThermodynamicsSA.ThermodynamicsSA Th_sa, 
            DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, 
            ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
