cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport Microphysics
cimport Microphysics_SB_Liquid
cimport Microphysics_Arctic_1M
cimport Microphysics_SB_SI
cimport Microphysics_SB_2M
cimport ParallelMPI
cimport TimeStepping
cimport ReferenceState
cimport Lookup
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
cimport ThermodynamicsSA
cimport ThermodynamicsSB
cimport TimeStepping

from libc.math cimport log, exp
from NetCDFIO cimport NetCDFIO_Stats
import cython

cimport numpy as np
import numpy as np
include "parameters.pxi"

import cython

cdef class IsotopeTracersNone:
    cdef public bint isotope_tracer

    cpdef initialize(self, namelist, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV,
            DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self)
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
            ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class IsotopeTracers_NoMicrophysics:
    cdef public bint isotope_tracer

    cpdef initialize(namelist, self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
            DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref, 
            Microphysics.No_Microphysics_SA Micro_SA, ThermodynamicsSA.ThermodynamicsSA Th_sa, 
            DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, 
            ReferenceState.ReferenceState Ref, Microphysics.No_Microphysics_SA Micro_SA, 
            TimeStepping.TimeStepping TS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class IsotopeTracers_SB_Liquid:
    cdef public bint isotope_tracer
    
    cdef:
        bint cloud_sedimentation

    cpdef initialize(namelist, self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
            DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref, 
            Microphysics_SB_Liquid.Microphysics_SB_Liquid Micro_SB_Liquid, ThermodynamicsSA.ThermodynamicsSA Th_sa, 
            DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, 
            ReferenceState.ReferenceState Ref, Microphysics_SB_Liquid.Microphysics_SB_Liquid Micro_SB_Liquid, 
            TimeStepping.TimeStepping TS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class IsotopeTracers_Arctic_1M:

    cdef:
        bint cloud_sedimentation
        public bint isotope_tracer

    cpdef initialize(namelist, self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
            DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref, 
            Microphysics_Arctic_1M.Microphysics_Arctic_1M Micro_Arctic_1M, ThermodynamicsSA.ThermodynamicsSA Th_sa, 
            DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, 
            ReferenceState.ReferenceState Ref, Microphysics_Arctic_1M.Microphysics_Arctic_1M Micro_Arctic_1M, 
            TimeStepping.TimeStepping TS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class IsotopeTracers_SBSI:
    
    cdef:
        double (*compute_rain_shape_parameter)(double density, double qr, double Dm) nogil
        double (*compute_droplet_nu)(double density, double ql) nogil

        double ccn

        bint cloud_sedimentation
        bint stokes_sedimentation
        Py_ssize_t order

        double [:] NI_Mayer
        double [:] NI_Flecher
        double [:] NI_Copper
        double [:] NI_Phillips
        double [:] NI_contact_Young
        double [:] NI_contact_Mayer

        public bint isotope_tracer

    cpdef initialize(namelist, self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
            DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref, 
            Microphysics_Arctic_1M.Microphysics_Arctic_1M Micro_Arctic_1M, ThermodynamicsSA.ThermodynamicsSA Th_sa, 
            DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, 
            ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)


cdef class IsotopeTracers_SB_Ice:
    
    cdef:
        double (*compute_rain_shape_parameter)(double density, double qr, double Dm) nogil
        double (*compute_droplet_nu)(double density, double ql) nogil

        double ccn
        double ice_nucl

        bint cloud_sedimentation
        bint stokes_sedimentation
        Py_ssize_t order

        public bint isotope_tracer

        double [:] Dm 
        double [:] mass 
        double [:] diagnose_1 
        double [:] diagnose_2 
        double [:] diagnose_3 
        double [:] diagnose_4 
        double [:] diagnose_5 

    cpdef initialize(namelist, self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
            DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref, 
        Microphysics_SB_2M.Microphysics_SB_2M Micro_SB_2M, ThermodynamicsSB.ThermodynamicsSB Th_sb, 
        DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, 
            ReferenceState.ReferenceState Ref,Microphysics_SB_2M.Microphysics_SB_2M Micro_SB_2M,
            TimeStepping.TimeStepping TS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
