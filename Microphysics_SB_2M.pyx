
#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport numpy as np 
import numpy as np 
cimport Lookup
cimport ParallelMPI
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
    void microphysics_stokes_sedimentation_velocity(Grid.DimStruct *dims, double* density, double CCN, double*  ql, 
            double*  qt_velocity) nogil
    void microphysics_wetbulb_temperature(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double* p0, double* s,
            double* qt,  double* T, double* Twet )nogil

cdef extern from "scalar_advection.h":
    void compute_advective_fluxes_a(Grid.DimStruct *dims, double *rho0, double *rho0_half, double *velocity, 
            double *scalar, double* flux, int d, int scheme) nogil
    void compute_qt_sedimentation_s_source(Grid.DimStruct *dims, double *p0_half,  double* rho0_half, double *flux,
            double* qt, double* qv, double* T, double* tendency, double (*lam_fp)(double),
            double (*L_fp)(double, double), double dx, ssize_t d)nogil

cdef extern from "microphysics_sb.h":
    double sb_rain_shape_parameter_0(double density, double qr, double Dm) nogil
    double sb_rain_shape_parameter_1(double density, double qr, double Dm) nogil
    double sb_rain_shape_parameter_2(double density, double qr, double Dm) nogil
    double sb_rain_shape_parameter_4(double density, double qr, double Dm) nogil
    double sb_droplet_nu_0(double density, double ql) nogil
    double sb_droplet_nu_1(double density, double ql) nogil
    double sb_droplet_nu_2(double density, double ql) nogil
    void sb_sedimentation_velocity_liquid(Grid.DimStruct *dims, double*  density, double CCN, double* ql, double* qt_velocity)nogil
    void sb_sedimentation_velocity_rain(Grid.DimStruct *dims, double (*rain_mu)(double,double,double),
            double* density, double* nr, double* qr, double* nr_velocity, double* qr_velocity) nogil
    void sb_autoconversion_rain_wrapper(Grid.DimStruct *dims,  double (*droplet_nu)(double,double), double* density,
            double CCN, double* ql, double* qr, double*  nr_tendency, double* qr_tendency) nogil
    void sb_accretion_rain_wrapper(Grid.DimStruct *dims, double* density, double*  ql, double* qr, double* qr_tendency)nogil
    void sb_selfcollection_breakup_rain_wrapper(Grid.DimStruct *dims, double (*rain_mu)(double,double,double),
            double* density, double* nr, double* qr, double*  nr_tendency)nogil
    void sb_evaporation_rain_wrapper(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
            double (*rain_mu)(double,double,double),  double* density, double* p0,  double* temperature,  double* qt,
            double* ql, double* nr, double* qr, double* nr_tendency, double* qr_tendency)nogil

cdef extern from "microphysics_sb_ice.h":
    void sb_ice_microphysics_sources(Grid.DimStruct *dims, 
        Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
        double* density, double* p0, double dt, 
        double CCN, double IN, 
        double* temperature, double* w,
        double*S, double* qt,
        double* nl, double* ql,
        double* ni, double* qi,
        double* nr, double* qr, 
        double* ns, double* qs, 
        double* Dm, double* mass,
        double* ice_self_col, double* snow_ice_col,
        double* snow_riming, double* snow_dep, double* snow_sub,
        double* nl_tend, double* ql_tend,
        double* ni_tend, double* qi_tend,
        double* nr_tend_micro, double* qr_tend_micro,
        double* nr_tend, double* qr_tend,
        double* ns_tend_micro, double* qs_tend_micro,
        double* ns_tend, double* qs_tend,
        double* precip_rate, double* evap_rate, double* melt_rate)nogil

    void sb_2m_entropy_source(Grid.DimStruct *dims, 
        Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        double* p0, double* temperature, double* Twet, double* w, 
        double* qt, double* qv, 
        double* qr, double* w_qr, 
        double* qs, double* w_qs,
        double* precip_rate, double* evap_rate, double* melt_rate,
        double* sp, double* se, double* sd, 
        double* sm, double* sq, double* sw, 
        double* s_tend)nogil
    
    void saturation_ratio(Grid.DimStruct *dims,  
        Lookup.LookupStruct *LT, double* p0, 
        double* temperature,  double* qt, 
        double* S_lookup, double* S_liq, double* S_ice)

    void sb_sedimentation_velocity_snow(Grid.DimStruct *dims,
        double* ns, double* qs, double* ns_velocity, double* qs_velocity)nogil

    void sb_2m_qt_source_formation(Grid.DimStruct *dims, 
        double* qt_tendency, double* precip_rate, double* evap_rate) nogil
    void sb_2m_qt_source_debug(Grid.DimStruct *dims, 
        double* qt_tendency, double* qr_tend, double* qs_tend) nogil

cdef class No_Microphysics_SB:
    def __init__(self, ParallelMPI.ParallelMPI Par, LatentHeat LH, namelist):
        self.thermodynamics_type = 'SB'
        
        LH.Lambda_fp = lambda_constant
        self.Lambda_fp = lambda_constant
        LH.L_fp = latent_heat_variable
        self.L_fp = latent_heat_variable

        # Extract case-specific parameter values from the namelist
        # Get number concentration of cloud condensation nuclei (1/m^3)
        try:
            self.CCN = namelist['microphysics']['CCN']
        except:
            self.CCN = 100.0e6
        try:
            self.order = namelist['scalar_transport']['order_sedimentation']
        except:
            self.order = namelist['scalar_transport']['order']

        try:
            self.cloud_sedimentation = namelist['microphysics']['cloud_sedimentation']
        except:
            self.cloud_sedimentation = False

        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        if self.cloud_sedimentation:
            DV.add_variables('w_qt', 'm/s', r'w_ql', 'cloud liquid water sedimentation velocity', 'sym', Pa)
            NS.add_profile('qt_sedimentation_flux', Gr, Pa)
            NS.add_profile('s_qt_sedimentation_source',Gr,Pa)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, PrognosticVariables.PrognosticVariables PV, 
            DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS,ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t wqt_shift
            Py_ssize_t ql_shift = PV.get_varshift(Gr,'ql')
        if self.cloud_sedimentation:
            wqt_shift = DV.get_varshift(Gr, 'w_qt')

            if self.stokes_sedimentation:
                microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], self.CCN, &DV.values[ql_shift], &DV.values[wqt_shift])
            else:
                sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], self.CCN, &PV.values[ql_shift], &DV.values[wqt_shift])
        return
        
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, PrognosticVariables.PrognosticVariables PV, 
            DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):
        return

cdef class Microphysics_SB_2M:
    def __init__(self, ParallelMPI.ParallelMPI Par, LatentHeat LH, namelist):

        self.thermodynamics_type = 'SB'

        # Create the appropriate linkages to the bulk thermodynamics
        # use the saturation adjustment only on cloud droplet
        LH.Lambda_fp = lambda_constant
        self.Lambda_fp = lambda_constant
        LH.L_fp = latent_heat_variable
        self.L_fp = latent_heat_variable

        Par.root_print('Using Arctic specific liquid fraction by Kaul et al. 2015!')

        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist, LH, Par)

        # Extract case-specific parameter values from the namelist
        # Set the number concentration of cloud condensation nuclei (1/m^3)
        # First set a default value, then set a case specific value, 
        # which can then be overwritten using namelist options
        self.CCN = 100.0e6
        try:
            self.CCN = namelist['microphysics']['CCN']
        except:
            pass
        
        self.ice_nucl = 2.0e2 # unit: L^-1, Cotton assumption of contact nucleation.
        try:
            self.ice_nucl = namelist['isotopetracers']['ice_nuclei']
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
                Par.root_print("SB_Liquid mu_rain option not recognized, defaulting to option 1")
                self.compute_rain_shape_parameter = sb_rain_shape_parameter_1
        except:
            Par.root_print("SB_Liquid mu_rain option not selected, defaulting to option 1")
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
                Par.root_print("SB_Liquid nu_droplet_option not recognized, defaulting to option 0")
                self.compute_droplet_nu = sb_droplet_nu_0
        except:
            Par.root_print("SB_Liquid nu_droplet_option not selected, defaulting to option 0")
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

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
            DiagnosticVariables.DiagnosticVariables DV, 
            NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        
        # add prognostic variables for mass and number of rain
        PV.add_variable('nr', '1/kg', r'n_r', 'rain droplet number concentration','sym','scalar',Pa)
        PV.add_variable('qr', 'kg/kg', r'q_r', 'rain water specific humidity','sym','scalar',Pa)

        PV.add_variable('ns', '1/kg', r'n_i', 'snow droplet number concentration','sym','scalar',Pa)
        PV.add_variable('qs', 'kg/kg', r'q_i', 'snow water specific humidity','sym','scalar',Pa)

        # add sedimentation velocities as diagnostic variables
        DV.add_variables('w_nr', 'm/s', r'w_{nr}', 'rain number sedimentation velocity', 'sym', Pa)
        DV.add_variables('w_qr', 'm/s', r'w_{qr}', 'rain mass sedimentation veloctiy', 'sym', Pa)
        DV.add_variables('w_ns', 'm/s', r'w_{nisi}', 'snow number sedimentation velocity', 'sym', Pa)
        DV.add_variables('w_qs', 'm/s', r'w_{qisi}', 'snow mass sedimentation veloctiy', 'sym', Pa)

        if self.cloud_sedimentation:
            DV.add_variables('w_qt', 'm/s', r'w_ql', 'cloud liquid water sedimentation velocity', 'sym', Pa)
            NS.add_profile('qt_sedimentation_flux', Gr, Pa)
            NS.add_profile('s_qt_sedimentation_source',Gr,Pa)

        # add wet bulb temperature 
        DV.add_variables('temperature_wb', 'K', r'T_{wb}','wet bulb temperature','sym', Pa) 
        
        NS.add_profile('wqs_mean', Gr, Pa, 'unit', '', 'wqs_mean')
        NS.add_profile('wqr_mean', Gr, Pa, 'unit', '', 'wqr_mean')
        
        # Define the precip_rate, evap_rate and melt_rate for entropy source calculation. 
        self.precip_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c') 
        self.evap_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c') 
        self.melt_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c') 

        NS.add_profile('precip_rate', Gr, Pa, '','','')
        NS.add_profile('evap_rate', Gr, Pa, '','','')
        NS.add_profile('melt_rate', Gr, Pa, '','','')

        # Define the snow diagnosed variables
        self.Dm            = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.mass          = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.ice_self_col  = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.snow_ice_col = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.snow_riming   = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.snow_dep      = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.snow_sub      = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.S_lookup = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.S_liq = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.S_ice = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        
        NS.add_profile('Dm', Gr, Pa, '','','')
        NS.add_profile('mass', Gr, Pa, '','','')
        NS.add_profile('ice_self_col', Gr, Pa, '','','')
        NS.add_profile('snow_ice_col', Gr, Pa, '','','')
        NS.add_profile('snow_riming', Gr, Pa, '','','')
        NS.add_profile('snow_dep', Gr, Pa, '','','')
        NS.add_profile('snow_sub', Gr, Pa, '','','')

        # Define the entropy source diagnosed variables
        self.sp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.se = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.sd = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.sm = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.sq = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.sw = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

        NS.add_profile('sp', Gr, Pa, '', '', '')
        NS.add_profile('se', Gr, Pa, '', '', '')
        NS.add_profile('sd', Gr, Pa, '', '', '')
        NS.add_profile('sm', Gr, Pa, '', '', '')
        NS.add_profile('sq', Gr, Pa, '', '', '')
        NS.add_profile('sw', Gr, Pa, '', '', '')
        
        NS.add_profile('S_lookup', Gr, Pa, '', '', '')
        NS.add_profile('S_liq', Gr, Pa, '', '', '')
        NS.add_profile('S_ice', Gr, Pa, '', '', '')

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, PrognosticVariables.PrognosticVariables PV, 
                 DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t ql_shift = PV.get_varshift(Gr,'ql')
            Py_ssize_t nl_shift = PV.get_varshift(Gr,'nl')
            Py_ssize_t qi_shift = PV.get_varshift(Gr,'qi')
            Py_ssize_t ni_shift = PV.get_varshift(Gr,'ni')
            Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')
            Py_ssize_t nr_shift = PV.get_varshift(Gr, 'nr')
            Py_ssize_t qr_shift = PV.get_varshift(Gr, 'qr')
            Py_ssize_t ns_shift = PV.get_varshift(Gr, 'ns')
            Py_ssize_t qs_shift = PV.get_varshift(Gr, 'qs')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')

            Py_ssize_t w_shift = PV.get_varshift(Gr, 'w')
            Py_ssize_t wqr_shift = DV.get_varshift(Gr, 'w_qr')
            Py_ssize_t wnr_shift = DV.get_varshift(Gr, 'w_nr')
            Py_ssize_t wqs_shift = DV.get_varshift(Gr, 'w_qs')
            Py_ssize_t wns_shift = DV.get_varshift(Gr, 'w_ns')
            Py_ssize_t wqt_shift

            double dt = TS.dt
            double[:] qr_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] nr_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qs_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] ns_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            # double[:] S_ratio = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

        saturation_ratio(&Gr.dims, 
            &self.CC.LT.LookupStructC,
            &Ref.p0_half[0], &DV.values[t_shift], 
            &PV.values[qt_shift],
            &self.S_lookup[0], &self.S_liq[0], &self.S_ice[0])

        # SB 2 moment microphysics source calculation
        sb_ice_microphysics_sources(&Gr.dims, 
            # thermodynamics setting
            &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp,
            # two moment rain droplet mu variable setting
            self.compute_rain_shape_parameter, self.compute_droplet_nu, 
            # INPUT ARRAY INDEX
            &Ref.rho0_half[0],  &Ref.p0_half[0], TS.dt,
            self.CCN, self.ice_nucl,
            &DV.values[t_shift], &PV.values[w_shift],
            &self.S_lookup[0], &PV.values[qt_shift], 
            &PV.values[nl_shift], &PV.values[ql_shift],
            &PV.values[ni_shift], &PV.values[qi_shift],
            &PV.values[nr_shift], &PV.values[qr_shift],
            &PV.values[ns_shift], &PV.values[qs_shift], 
            # ------ DIAGNOSED VARIABLES ---------
            &self.Dm[0], &self.mass[0],
            &self.ice_self_col[0], &self.snow_ice_col[0],
            &self.snow_riming[0], &self.snow_dep[0], &self.snow_sub[0],
            # ------------------------------------
            &PV.tendencies[nl_shift], &PV.tendencies[ql_shift],
            &PV.tendencies[ni_shift], &PV.tendencies[qi_shift],
            &nr_tend_micro[0], &qr_tend_micro[0], &PV.tendencies[nr_shift], &PV.tendencies[qr_shift],
            &ns_tend_micro[0], &qs_tend_micro[0], &PV.tendencies[ns_shift], &PV.tendencies[qs_shift],
            &self.precip_rate[0], &self.evap_rate[0], &self.melt_rate[0])
        
        sb_2m_qt_source_debug(&Gr.dims, &PV.tendencies[qt_shift], 
            &qr_tend_micro[0], &qs_tend_micro[0])
        
        # sedimentation processes of rain and single_ice: w_qr and w_qs
        sb_sedimentation_velocity_rain(&Gr.dims, self.compute_rain_shape_parameter, 
            &Ref.rho0_half[0], &PV.values[nr_shift], &PV.values[qr_shift], 
            &DV.values[wnr_shift], &DV.values[wqr_shift])
        sb_sedimentation_velocity_snow(&Gr.dims,
            &PV.values[ns_shift], &PV.values[qs_shift],  
            &DV.values[wns_shift], &DV.values[wqs_shift])

        if self.cloud_sedimentation:
            wqt_shift = DV.get_varshift(Gr, 'w_qt')

            if self.stokes_sedimentation:
                microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], 
                    self.CCN, &PV.values[ql_shift], &DV.values[wqt_shift])
            else:
                sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], 
                    self.CCN, &PV.values[ql_shift], &DV.values[wqt_shift])
            
        # ========= ENTROPY SOURCE OF MICROPHYSICS PROCESSES ============== 
        cdef:
            Py_ssize_t tw_shift = DV.get_varshift(Gr, 'temperature_wb')
        # Get wetbuble temperature
        microphysics_wetbulb_temperature(&Gr.dims, &self.CC.LT.LookupStructC, 
            &Ref.p0_half[0], &PV.values[s_shift], &PV.values[qt_shift], 
            &DV.values[t_shift], &DV.values[tw_shift])

        sb_2m_entropy_source(&Gr.dims,
            &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, # thermodynamics setting
            &Ref.p0_half[0], &DV.values[t_shift], &DV.values[tw_shift], &PV.values[w_shift], 
            &PV.values[qt_shift], &DV.values[qv_shift],
            &PV.values[qr_shift], &DV.values[wqr_shift], 
            &PV.values[qs_shift], &DV.values[wqs_shift],
            &self.precip_rate[0], &self.evap_rate[0], &self.melt_rate[0],
            # ------ DIAGNOSED VARIABLES ---------
            &self.sp[0], &self.se[0], &self.sd[0], 
            &self.sm[0], &self.sq[0], &self.sw[0], 
            # ------------------------------------
            &PV.tendencies[s_shift])

        return
    
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, 
            PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, 
            NetCDFIO_Stats NS, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        cdef:
            double[:] tmp
        
        tmp = Pa.HorizontalMean(Gr, &self.precip_rate[0])
        NS.write_profile('precip_rate', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.evap_rate[0])
        NS.write_profile('evap_rate', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.melt_rate[0])
        NS.write_profile('melt_rate', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.Dm[0])
        NS.write_profile('Dm', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.mass[0])
        NS.write_profile('mass', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.ice_self_col[0])
        NS.write_profile('ice_self_col', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.snow_ice_col[0])
        NS.write_profile('snow_ice_col', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.snow_riming[0])
        NS.write_profile('snow_riming', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.snow_dep[0])
        NS.write_profile('snow_dep', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.snow_sub[0])
        NS.write_profile('snow_sub', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.sp[0])
        NS.write_profile('sp', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.se[0])
        NS.write_profile('se', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.sd[0])
        NS.write_profile('sd', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.sm[0])
        NS.write_profile('sm', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.sq[0])
        NS.write_profile('sq', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.sw[0])
        NS.write_profile('sw', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)       
        
        tmp = Pa.HorizontalMean(Gr, &self.S_lookup[0])
        NS.write_profile('S_lookup', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)       
        tmp = Pa.HorizontalMean(Gr, &self.S_liq[0])
        NS.write_profile('S_liq', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)       
        tmp = Pa.HorizontalMean(Gr, &self.S_ice[0])
        NS.write_profile('S_ice', tmp[Gr.dims.gw: -Gr.dims.gw], Pa)       

        return
