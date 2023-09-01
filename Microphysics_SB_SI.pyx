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
    void microphysics_stokes_sedimentation_velocity(Grid.DimStruct *dims, double* density, double ccn, double*  ql, 
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
    void sb_sedimentation_velocity_liquid(Grid.DimStruct *dims, double*  density, double ccn, double* ql, double* qt_velocity)nogil
    void sb_sedimentation_velocity_rain(Grid.DimStruct *dims, double (*rain_mu)(double,double,double),
            double* density, double* nr, double* qr, double* nr_velocity, double* qr_velocity) nogil
    void sb_autoconversion_rain_wrapper(Grid.DimStruct *dims,  double (*droplet_nu)(double,double), double* density,
            double ccn, double* ql, double* qr, double*  nr_tendency, double* qr_tendency) nogil
    void sb_accretion_rain_wrapper(Grid.DimStruct *dims, double* density, double*  ql, double* qr, double* qr_tendency)nogil
    void sb_selfcollection_breakup_rain_wrapper(Grid.DimStruct *dims, double (*rain_mu)(double,double,double),
            double* density, double* nr, double* qr, double*  nr_tendency)nogil
    void sb_evaporation_rain_wrapper(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
            double (*rain_mu)(double,double,double),  double* density, double* p0,  double* temperature,  double* qt,
            double* ql, double* nr, double* qr, double* nr_tendency, double* qr_tendency)nogil

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
    void sb_si_entropy_source_heating_rain(Grid.DimStruct *dims, double* T, double* Twet, double* qr,
            double* w_qr, double* w,  double* entropy_tendency) nogil
    void sb_si_entropy_source_heating_ice(Grid.DimStruct *dims, double* T, double* Twet, double* qisi,
            double* w_qs, double* w,  double* entropy_tendency) nogil
    void sb_si_entropy_source_drag(Grid.DimStruct *dims, double* T, double* qprec, double* w_qprec,
            double* entropy_tendency) nogil
    void sb_si_entropy_source_evaporation(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double),
            double (*L_fp)(double, double), double* p0, double* temperature,
            double* Twet, double* qt, double* qv, double* qr_tend, double* evap_rate, double* entropy_tendency)
    void sb_si_entropy_source_precipitation(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double),
            double (*L_fp)(double, double), double* p0, double* temperature,
            double* qt, double* qv, double* qr_tend, double* precip_rate, double* entropy_tendency)
    void sb_si_entropy_source_melt(Grid.DimStruct *dims, double* temperature, double* melt_rate, double* entropy_tendency)
    void sb_nucleation_ice_wrapper(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double* temperature, double* ni, double* qi, double* p0, 
            double* qt, double dt, double* ni_tendency, double* qi_tendency)nogil
    void sb_deposition_ice_wrapper(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
            double* temperature, double* p0, double* qt, double* ni, double* qi, double* qi_tendency)nogil
    void sb_sublimation_ice_wrapper(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
            double* temperature, double* p0, double* qt, double* ni, double* qi, double* qi_tendency)nogil
    void sb_freezing_ice_wrapper(Grid.DimStruct *dims, double (*droplet_nu)(double,double), double ccn, double* temperature, double* density,
            double* ql, double* qr, double* nr, double* qi_tendency, double* ni_tendency)nogil
    void sb_melting_ice_wrapper(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
            double* temperature, double* qv, double* qi, double* ni, double* qi_tendency, double* ni_tendency)nogil
    void sb_accretion_cloud_ice_wrapper(Grid.DimStruct *dims, double ccn, double* density, double* ql, double* qi, double* ni, double* qi_tendency)nogil

cdef extern from "isotope.h":
    void tracer_sb_si_microphysics_sources(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
            double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
            double* density, double* p0, double* temperature,  double* qt, double ccn,
            double* ql, double* nr, double* qr, double* qisi, double* nisi, double dt, double* ql_std,
            double* nr_tendency_micro, double* qr_tendency_micro, double* nr_tendency, double* qr_tendency, 
            double* nisi_tendency_micro, double* qisi_tendency_micro, double* nisi_tendency, double* qisi_tendency,
            double* precip_rate, double* evap_rate, double* melt_rate, 
            double* qt_O18, double* ql_O18, double* qr_O18, double* qisi_O18, 
            double* qr_O18_tendency, double* qisi_O18_tendency, double* qr_O18_tend_micro, double* qisi_O18_tend_micro,
            double* qt_HDO, double* ql_HDO, double* qr_HDO, double* qisi_HDO, 
            double* qr_HDO_tendency, double* qisi_HDO_tendency, double* qr_HDO_tend_micro, double* qisi_HDO_tend_micro) nogil

cdef class Microphysics_SB_SI:
    def __init__(self, ParallelMPI.ParallelMPI Par, LatentHeat LH, namelist):
        # Create the appropriate linkages to the bulk thermodynamics
        LH.Lambda_fp = lambda_constant
        LH.L_fp = latent_heat_variable
        self.thermodynamics_type = 'SA'
        #also set local versions
        self.Lambda_fp = lambda_constant
        self.L_fp = latent_heat_variable
        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist, LH, Par)

        # Extract case-specific parameter values from the namelist
        # Set the number concentration of cloud condensation nuclei (1/m^3)
        # First set a default value, then set a case specific value, which can then be overwritten using namelist options
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

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, 
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        # Define the precip_rate, evap_rate and melt_rate for entropy source calculation. 
        self.precip_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c') 
        self.evap_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c') 
        self.melt_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c') 
        
        # add prognostic variables for mass and number of rain
        PV.add_variable('nr', '1/kg', r'n_r', 'rain droplet number concentration','sym','scalar',Pa)
        PV.add_variable('qr', 'kg/kg', r'q_r', 'rain water specific humidity','sym','scalar',Pa)

        PV.add_variable('nisi', '1/kg', r'n_i', 'total ice droplet number concentration','sym','scalar',Pa)
        PV.add_variable('qisi', 'kg/kg', r'q_i', 'total ice water specific humidity','sym','scalar',Pa)

        # add sedimentation velocities as diagnostic variables
        DV.add_variables('w_qr', 'm/s', r'w_{qr}', 'rain mass sedimentation veloctiy', 'sym', Pa)
        DV.add_variables('w_nr', 'm/s', r'w_{nr}', 'rain number sedimentation velocity', 'sym', Pa)
        DV.add_variables('w_qisi', 'm/s', r'w_{qisi}', 'total ice mass sedimentation veloctiy', 'sym', Pa)
        DV.add_variables('w_nisi', 'm/s', r'w_{nisi}', 'total ice number sedimentation velocity', 'sym', Pa)
        if self.cloud_sedimentation:
            DV.add_variables('w_qt', 'm/s', r'w_ql', 'cloud liquid water sedimentation velocity', 'sym', Pa)
            NS.add_profile('qt_sedimentation_flux', Gr, Pa)
            NS.add_profile('s_qt_sedimentation_source',Gr,Pa)
        # add wet bulb temperature 
        DV.add_variables('temperature_wb', 'K', r'T_{wb}','wet bulb temperature','sym', Pa) 
        
        # add statistical output for the class
        # output for rain variables: qr and nr
        NS.add_profile('nr_sedimentation_flux', Gr, Pa) 
        NS.add_profile('qr_sedimentation_flux', Gr, Pa) 
        NS.add_profile('qr_autoconversion', Gr, Pa) 
        NS.add_profile('nr_autoconversion', Gr, Pa) 
        NS.add_profile('s_autoconversion', Gr, Pa)
        NS.add_profile('nr_selfcollection', Gr, Pa)
        NS.add_profile('qr_accretion', Gr, Pa)
        NS.add_profile('s_accretion', Gr, Pa)
        NS.add_profile('nr_evaporation', Gr, Pa)
        NS.add_profile('qr_evaporation', Gr,Pa)
        NS.add_profile('s_evaporation', Gr,Pa)
        NS.add_profile('s_precip_heating', Gr, Pa)
        NS.add_profile('s_precip_drag', Gr, Pa)
        # output for rain variables: qisi and nisi
        NS.add_profile('qisi_sedimentation_flux', Gr, Pa) 
        NS.add_profile('nisi_sedimentation_flux', Gr, Pa) 
        NS.add_profile('qisi_nucleation', Gr, Pa) 
        NS.add_profile('nisi_nucleation', Gr, Pa) 
        NS.add_profile('qisi_deposition', Gr, Pa) 
        NS.add_profile('qisi_sublimation', Gr, Pa) 
        NS.add_profile('qisi_freezing', Gr, Pa) 
        NS.add_profile('nisi_freezing', Gr, Pa) 
        NS.add_profile('qisi_melting', Gr, Pa) 
        NS.add_profile('nisi_melting', Gr, Pa) 
        NS.add_profile('qisi_accretion_cloud_ice', Gr, Pa)
        NS.add_profile('wqisi_mean', Gr, Pa, 'unit', '', 'wqisi_mean')
        NS.add_profile('wqr_mean', Gr, Pa, 'unit', '', 'wqr_mean')
        
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, PrognosticVariables.PrognosticVariables PV, 
                 DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        # cdef:
        #     Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
        #     Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
        #     Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')
        #     Py_ssize_t nr_shift = PV.get_varshift(Gr, 'nr')
        #     Py_ssize_t qr_shift = PV.get_varshift(Gr, 'qr')
        #     Py_ssize_t nisi_shift = PV.get_varshift(Gr, 'nisi')
        #     Py_ssize_t qisi_shift = PV.get_varshift(Gr, 'qisi')
        #     Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
        #     Py_ssize_t w_shift = PV.get_varshift(Gr, 'w')
        #     double dt = TS.dt
        #     Py_ssize_t wqr_shift = DV.get_varshift(Gr, 'w_qr')
        #     Py_ssize_t wnr_shift = DV.get_varshift(Gr, 'w_nr')
        #     Py_ssize_t wqisi_shift = DV.get_varshift(Gr, 'w_qisi')
        #     Py_ssize_t wnisi_shift = DV.get_varshift(Gr, 'w_nisi')
        #     Py_ssize_t wqt_shift
        #     double[:] qr_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        #     double[:] nr_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        #     double[:] qisi_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        #     double[:] nisi_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        #
        # sb_si_microphysics_sources(&Gr.dims, 
        #     self.compute_rain_shape_parameter, self.compute_droplet_nu, 
        #     &Ref.rho0_half[0],  &Ref.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift], self.ccn, 
        #     &DV.values[ql_shift], &PV.values[nr_shift], &PV.values[qr_shift], &PV.values[qisi_shift], &PV.values[nisi_shift], dt,   
        #     &nr_tend_micro[0], &qr_tend_micro[0], &PV.tendencies[nr_shift], &PV.tendencies[qr_shift],
        #     &nisi_tend_micro[0], &qisi_tend_micro[0], &PV.tendencies[nisi_shift], &PV.tendencies[qisi_shift],
        #     &self.precip_rate[0], &self.evap_rate[0], &self.melt_rate[0])
        #
        # sb_si_qt_source_formation(&Gr.dims, &qisi_tend_micro[0], &qr_tend_micro[0], &PV.tendencies[qt_shift])
        #
        # # sedimentation processes of rain and single_ice: w_qr and w_qisi
        # sb_sedimentation_velocity_rain(&Gr.dims, self.compute_rain_shape_parameter, &Ref.rho0_half[0], &PV.values[nr_shift],
        #     &PV.values[qr_shift], &DV.values[wnr_shift], &DV.values[wqr_shift])
        # sb_sedimentation_velocity_ice(&Gr.dims, &PV.values[nisi_shift], &PV.values[qisi_shift], &Ref.rho0_half[0], 
        #     &DV.values[wnisi_shift], &DV.values[wqisi_shift])
        # if self.cloud_sedimentation:
        #     wqt_shift = DV.get_varshift(Gr, 'w_qt')
        #
        #     if self.stokes_sedimentation:
        #         microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], self.ccn, &DV.values[ql_shift], &DV.values[wqt_shift])
        #     else:
        #         sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], self.ccn, &DV.values[ql_shift], &DV.values[wqt_shift])

        # # tracer micro-source formation
        # cdef:
        #     # std-tracers indexes defination
        #     Py_ssize_t ql_std_shift
        #     Py_ssize_t qr_std_shift
        #     Py_ssize_t qisi_std_shift
        #     Py_ssize_t nisi_std_shift
        #     Py_ssize_t nr_std_shift
        #     Py_ssize_t qt_std_shift
        #     Py_ssize_t wqt_std_shift
        #     Py_ssize_t wqr_std_shift
        #     Py_ssize_t wnr_std_shift
        #     Py_ssize_t wqisi_std_shift
        #     Py_ssize_t wnisi_std_shift
        #     double[:] qr_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        #     double[:] nr_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        #     double[:] qisi_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        #     double[:] nisi_std_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        #
        #     # iso-tracers indexes defination
        #     Py_ssize_t ql_O18_shift
        #     Py_ssize_t qv_O18_shift
        #     Py_ssize_t qr_O18_shift
        #     Py_ssize_t qt_O18_shift
        #     Py_ssize_t wqt_O18_shift
        #     Py_ssize_t wqr_O18_shift
        #     Py_ssize_t wqisi_O18_shift
        #
        #     Py_ssize_t ql_HDO_shift
        #     Py_ssize_t qv_HDO_shift
        #     Py_ssize_t qr_HDO_shift
        #     Py_ssize_t qt_HDO_shift
        #     Py_ssize_t wqt_HDO_shift
        #     Py_ssize_t wqr_HDO_shift
        #     Py_ssize_t wqisi_HDO_shift
        #     
        #     Py_ssize_t wnr_iso_shift
        #     Py_ssize_t wnisi_iso_shift
        #
        #     double[:] wnisi_iso_tmp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        #
        #     double[:] qr_O18_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        #     double[:] qisi_O18_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        #     double[:] qr_HDO_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        #     double[:] qisi_HDO_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        #
        #     double[:] precip_rate_std = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        #     double[:] evap_rate_std = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        #     double[:] melt_rate_std = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        #
        # if self.isotope_tracers:
        #     qr_std_shift  = PV.get_varshift(Gr, 'qr_std')
        #     nr_std_shift  = PV.get_varshift(Gr, 'nr_std')
        #     ql_std_shift  = PV.get_varshift(Gr, 'ql_std')
        #     qt_std_shift  = PV.get_varshift(Gr, 'qt_std')
        #     wqr_std_shift = DV.get_varshift(Gr, 'w_qr_std')
        #     wnr_std_shift = DV.get_varshift(Gr, 'w_nr_std')
        #     qisi_std_shift = PV.get_varshift(Gr, 'qisi_std')
        #     nisi_std_shift = PV.get_varshift(Gr, 'nisi_std')
        #     wqisi_std_shift = DV.get_varshift(Gr, 'w_qisi_std')
        #     wnisi_std_shift = DV.get_varshift(Gr, 'w_nisi_std')
        #     
        #     ql_O18_shift  = PV.get_varshift(Gr,'ql_O18')
        #     qv_O18_shift  = PV.get_varshift(Gr,'qv_O18')
        #     qr_O18_shift  = PV.get_varshift(Gr, 'qr_O18')
        #     qt_O18_shift  = PV.get_varshift(Gr, 'qt_O18')
        #     qisi_O18_shift  = PV.get_varshift(Gr, 'qisi_O18')
        #     wqr_O18_shift = DV.get_varshift(Gr, 'w_qr_O18')
        #     wqisi_O18_shift = DV.get_varshift(Gr, 'w_qisi_O18')
        #     
        #     ql_HDO_shift  = PV.get_varshift(Gr,'ql_HDO')
        #     qv_HDO_shift  = PV.get_varshift(Gr,'qv_HDO')
        #     qr_HDO_shift  = PV.get_varshift(Gr, 'qr_HDO')
        #     qt_HDO_shift  = PV.get_varshift(Gr, 'qt_HDO')
        #     qisi_HDO_shift  = PV.get_varshift(Gr, 'qisi_HDO')
        #     wqr_HDO_shift = DV.get_varshift(Gr, 'w_qr_HDO')
        #     wqisi_HDO_shift = DV.get_varshift(Gr, 'w_qisi_HDO')
        #
        #     wnr_iso_shift = DV.get_varshift(Gr, 'w_nr_iso')
        #     wnisi_iso_shift = DV.get_varshift(Gr, 'w_nisi_iso')
        #
        #     tracer_sb_si_microphysics_sources(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, 
        #         self.compute_rain_shape_parameter, self.compute_droplet_nu,
        #         &Ref.rho0_half[0],  &Ref.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift], self.ccn, &DV.values[ql_shift],
        #         &PV.values[nr_std_shift], &PV.values[qr_std_shift], &PV.values[qisi_std_shift], &PV.values[nisi_std_shift], dt,
        #         &PV.values[ql_std_shift], &nr_std_tend_micro[0], &qr_std_tend_micro[0], &PV.tendencies[nr_std_shift],
        #         &PV.tendencies[qr_std_shift], &nisi_std_tend_micro[0],  &qisi_std_tend_micro[0], &PV.tendencies[nisi_std_shift],
        #         &PV.tendencies[qisi_std_shift], &precip_rate_std[0], &evap_rate_std[0], &melt_rate_std[0], 
        #         &PV.values[qt_O18_shift], &PV.values[ql_O18_shift], &PV.values[qr_O18_shift], 
        #         &PV.values[qisi_O18_shift], &PV.tendencies[qr_O18_shift], &PV.tendencies[qisi_O18_shift], 
        #         &qr_O18_tend_micro[0], &qisi_O18_tend_micro[0],
        #         &PV.values[qt_HDO_shift], &PV.values[ql_HDO_shift], &PV.values[qr_HDO_shift], 
        #         &PV.values[qisi_HDO_shift], &PV.tendencies[qr_HDO_shift], &PV.tendencies[qisi_HDO_shift], 
        #         &qr_HDO_tend_micro[0], &qisi_HDO_tend_micro[0])
        #     
        #     sb_sedimentation_velocity_rain(&Gr.dims, self.compute_rain_shape_parameter, &Ref.rho0_half[0], &PV.values[nr_shift],
        #         &PV.values[qr_shift], &DV.values[wnr_std_shift], &DV.values[wqr_std_shift])
        #     sb_sedimentation_velocity_rain(&Gr.dims, self.compute_rain_shape_parameter, &Ref.rho0_half[0], &PV.values[nr_shift],
        #         &PV.values[qr_shift], &DV.values[wnr_iso_shift], &DV.values[wqr_O18_shift])
        #     sb_sedimentation_velocity_rain(&Gr.dims, self.compute_rain_shape_parameter, &Ref.rho0_half[0], &PV.values[nr_shift],
        #         &PV.values[qr_shift], &DV.values[wnr_iso_shift], &DV.values[wqr_HDO_shift])
        #     sb_sedimentation_velocity_ice(&Gr.dims, &PV.values[nisi_shift], &PV.values[qisi_shift], &Ref.rho0_half[0], 
        #         &DV.values[wnisi_std_shift], &DV.values[wqisi_std_shift])
        #     sb_sedimentation_velocity_ice(&Gr.dims, &PV.values[nisi_shift], &PV.values[qisi_shift], &Ref.rho0_half[0], 
        #         &DV.values[wnisi_iso_shift], &DV.values[wqisi_O18_shift])
        #     sb_sedimentation_velocity_ice(&Gr.dims, &PV.values[nisi_shift], &PV.values[qisi_shift], &Ref.rho0_half[0], 
        #         &DV.values[wnisi_iso_shift], &DV.values[wqisi_HDO_shift])
        #     
        #     if self.cloud_sedimentation:
        #         wqt_std_shift = DV.get_varshift(Gr, 'w_qt_std')
        #         wqt_O18_shift = DV.get_varshift(Gr, 'w_qt_O18')
        #         wqt_HDO_shift = DV.get_varshift(Gr, 'w_qt_HDO')
        #
        #         if self.stokes_sedimentation:
        #             microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], self.ccn, &DV.values[ql_shift], &DV.values[wqt_std_shift])
        #             microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], self.ccn, &DV.values[ql_shift], &DV.values[wqt_O18_shift])
        #             microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], self.ccn, &DV.values[ql_shift], &DV.values[wqt_HDO_shift])
        #         else:
        #             sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], self.ccn, &DV.values[ql_shift], &DV.values[wqt_std_shift])
        #             sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], self.ccn, &DV.values[ql_shift], &DV.values[wqt_O18_shift])
        #             sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], self.ccn, &DV.values[ql_shift], &DV.values[wqt_HDO_shift])
        #
        #     sb_si_qt_source_formation(&Gr.dims, &qisi_std_tend_micro[0], &qr_std_tend_micro[0], &PV.tendencies[qt_std_shift]);
        #     sb_si_qt_source_formation(&Gr.dims, &qisi_O18_tend_micro[0], &qr_O18_tend_micro[0], &PV.tendencies[qt_O18_shift]);
        #     sb_si_qt_source_formation(&Gr.dims, &qisi_HDO_tend_micro[0], &qr_HDO_tend_micro[0], &PV.tendencies[qt_HDO_shift]);
            
        # entropy source for microphysics processes
        # cdef:
        #     Py_ssize_t tw_shift = DV.get_varshift(Gr, 'temperature_wb')
        #     Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
        #
        # microphysics_wetbulb_temperature(&Gr.dims, &self.CC.LT.LookupStructC, &Ref.p0_half[0], &PV.values[s_shift],
        #                                   &PV.values[qt_shift], &DV.values[t_shift], &DV.values[tw_shift])
        #
        # sb_si_entropy_source_precipitation(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &Ref.p0_half[0],
        #                              &DV.values[t_shift], &PV.values[qt_shift], &DV.values[qv_shift], &qr_tend_micro[0],
        #                              &self.precip_rate[0], &PV.tendencies[s_shift])
        #
        # sb_si_entropy_source_evaporation(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &Ref.p0_half[0],
        #                            &DV.values[t_shift], &DV.values[tw_shift], &PV.values[qt_shift], &DV.values[qv_shift], 
        #                            &qr_tend_micro[0], &self.evap_rate[0], &PV.tendencies[s_shift])
        #
        # sb_si_entropy_source_melt(&Gr.dims, &DV.values[t_shift], &self.melt_rate[0], &PV.tendencies[s_shift])
        # 
        # sb_si_entropy_source_heating_rain(&Gr.dims, &DV.values[t_shift], &DV.values[tw_shift], &PV.values[qr_shift],
        #                           &DV.values[wqr_shift],  &PV.values[w_shift], &PV.tendencies[s_shift])
        #
        # sb_si_entropy_source_heating_ice(&Gr.dims, &DV.values[t_shift], &DV.values[tw_shift], &PV.values[qisi_shift],
        #                           &DV.values[wqisi_shift],  &PV.values[w_shift], &PV.tendencies[s_shift])
        # # entropy from rain drag
        # sb_si_entropy_source_drag(&Gr.dims, &DV.values[t_shift], &PV.values[qr_shift], &DV.values[wqr_shift],
        #                     &PV.tendencies[s_shift])
        # # entropy from ice drag
        # sb_si_entropy_source_drag(&Gr.dims, &DV.values[t_shift], &PV.values[qisi_shift], &DV.values[wqisi_shift],
        #                     &PV.tendencies[s_shift])

        return
    
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, 
                   NetCDFIO_Stats NS, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):
        # cdef:
        #     Py_ssize_t i, j, k, ijk
        #     Py_ssize_t gw = Gr.dims.gw
        #     Py_ssize_t imax = Gr.dims.nlg[0]
        #     Py_ssize_t jmax = Gr.dims.nlg[1]
        #     Py_ssize_t kmax = Gr.dims.nlg[2]
        #     Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        #     Py_ssize_t jstride = Gr.dims.nlg[2]
        #     Py_ssize_t ishift, jshift
        #
        #     Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
        #     Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
        #     Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')
        #     Py_ssize_t nr_shift = PV.get_varshift(Gr, 'nr')
        #     Py_ssize_t qr_shift = PV.get_varshift(Gr, 'qr')
        #     Py_ssize_t nisi_shift = PV.get_varshift(Gr, 'nisi')
        #     Py_ssize_t qisi_shift = PV.get_varshift(Gr, 'qisi')
        #     Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
        #     Py_ssize_t w_shift = PV.get_varshift(Gr, 'w')
        #     double dt = TS.dt
        #     double[:] dummy =  np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        #     Py_ssize_t wqr_shift = DV.get_varshift(Gr, 'w_qr')
        #     Py_ssize_t wnr_shift = DV.get_varshift(Gr, 'w_nr')
        #     Py_ssize_t wqisi_shift = DV.get_varshift(Gr, 'w_qisi')
        #     Py_ssize_t wnisi_shift = DV.get_varshift(Gr, 'w_nisi')
        #     Py_ssize_t wqt_shift
        #     double[:] qr_tendency = np.empty((Gr.dims.npg,), dtype=np.double, order='c')
        #     double[:] nr_tendency = np.empty((Gr.dims.npg,), dtype=np.double, order='c')
        #
        # cdef double[:] s_src =  np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        # if self.cloud_sedimentation:
        #     wqt_shift = DV.get_varshift(Gr,'w_qt')
        #
        #     compute_advective_fluxes_a(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &DV.values[wqt_shift], &DV.values[ql_shift], &dummy[0], 2, self.order)
        #     tmp = Pa.HorizontalMean(Gr, &dummy[0])
        #     NS.write_profile('qt_sedimentation_flux', tmp[gw:-gw], Pa)
        #
        #     compute_qt_sedimentation_s_source(&Gr.dims, &Ref.p0_half[0], &Ref.rho0_half[0], &dummy[0],
        #                             &PV.values[qt_shift], &DV.values[qv_shift],&DV.values[t_shift], &s_src[0], self.Lambda_fp,
        #                             self.L_fp, Gr.dims.dx[2], 2)
        #     tmp = Pa.HorizontalMean(Gr, &s_src[0])
        #     NS.write_profile('s_qt_sedimentation_source', tmp[gw:-gw], Pa)
        #
        # #compute sedimentation flux only of nr
        # compute_advective_fluxes_a(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &DV.values[wnr_shift], &PV.values[nr_shift], &dummy[0], 2, self.order)
        # tmp = Pa.HorizontalMean(Gr, &dummy[0])
        # NS.write_profile('nr_sedimentation_flux', tmp[gw:-gw], Pa)
        # 
        # #compute sedimentation flux only of qr
        # compute_advective_fluxes_a(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &DV.values[wqr_shift], &PV.values[qr_shift], &dummy[0], 2, self.order)
        # tmp = Pa.HorizontalMean(Gr, &dummy[0])
        # NS.write_profile('qr_sedimentation_flux', tmp[gw:-gw], Pa)
        # 
        # #compute sedimentation flux only of nisi
        # compute_advective_fluxes_a(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &DV.values[wnisi_shift], &PV.values[nisi_shift], &dummy[0], 2, self.order)
        # tmp = Pa.HorizontalMean(Gr, &dummy[0])
        # NS.write_profile('nisi_sedimentation_flux', tmp[gw:-gw], Pa)
        #
        # #compute sedimentation flux only of qisi
        # compute_advective_fluxes_a(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &DV.values[wqisi_shift], &PV.values[qisi_shift], &dummy[0], 2, self.order)
        # tmp = Pa.HorizontalMean(Gr, &dummy[0])
        # NS.write_profile('qisi_sedimentation_flux', tmp[gw:-gw], Pa)
        # 
        # #note we can re-use nr_tendency and qr_tendency because they are overwritten in each function
        # #must have a zero array to pass as entropy tendency and need to send a dummy variable for qt tendency
        #
        # # Autoconversion tendencies of qr, nr, s
        # # comment the entropy source computation for temp purpose;
        # sb_autoconversion_rain_wrapper(&Gr.dims,  self.compute_droplet_nu, &Ref.rho0_half[0], self.ccn,
        #                                &DV.values[ql_shift], &PV.values[qr_shift], &nr_tendency[0], &qr_tendency[0])
        # tmp = Pa.HorizontalMean(Gr, &nr_tendency[0])
        # NS.write_profile('nr_autoconversion', tmp[gw:-gw], Pa)
        # tmp = Pa.HorizontalMean(Gr, &qr_tendency[0])
        # NS.write_profile('qr_autoconversion', tmp[gw:-gw], Pa)
        # # cdef double[:] s_auto = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        # # sb_entropy_source_formation(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &Ref.p0_half[0],
        # #                           &DV.values[t_shift], &DV.values[tw_shift],&PV.values[qt_shift], &DV.values[qv_shift],
        # #                           &qr_tendency[0], &s_auto[0])
        # #
        # # tmp = Pa.HorizontalMean(Gr, &s_auto[0])
        # # NS.write_profile('s_autoconversion', tmp[gw:-gw], Pa)
        #
        #
        # # Accretion tendencies of qr, s
        # # comment the entropy source computation for temp purpose;
        # sb_accretion_rain_wrapper(&Gr.dims, &Ref.rho0_half[0], &DV.values[ql_shift], &PV.values[qr_shift], &qr_tendency[0])
        # tmp = Pa.HorizontalMean(Gr, &qr_tendency[0])
        # NS.write_profile('qr_accretion', tmp[gw:-gw], Pa)
        # # cdef double[:] s_accr = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        # # sb_entropy_source_formation(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &Ref.p0_half[0],
        # #                           &DV.values[t_shift], &DV.values[tw_shift],&PV.values[qt_shift], &DV.values[qv_shift],
        # #                           &qr_tendency[0], &s_accr[0])
        # # tmp = Pa.HorizontalMean(Gr, &s_accr[0])
        # # NS.write_profile('s_accretion', tmp[gw:-gw], Pa)
        #
        # # Self-collection and breakup tendencies (lumped) of nr
        # sb_selfcollection_breakup_rain_wrapper(&Gr.dims, self.compute_rain_shape_parameter, &Ref.rho0_half[0],
        #                                        &PV.values[nr_shift], &PV.values[qr_shift], &nr_tendency[0])
        # tmp = Pa.HorizontalMean(Gr, &nr_tendency[0])
        # NS.write_profile('nr_selfcollection', tmp[gw:-gw], Pa)
        #
        # # Evaporation tendencies of qr, nr, s
        # # comment the entropy source computation for temp purpose;
        # sb_evaporation_rain_wrapper(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp,
        #                             self.compute_rain_shape_parameter, &Ref.rho0_half[0], &Ref.p0_half[0],
        #                             &DV.values[t_shift], &PV.values[qt_shift], &DV.values[ql_shift],
        #                             &PV.values[nr_shift], &PV.values[qr_shift], &nr_tendency[0], &qr_tendency[0])
        #
        # tmp = Pa.HorizontalMean(Gr, &nr_tendency[0])
        # NS.write_profile('nr_evaporation', tmp[gw:-gw], Pa)
        # tmp = Pa.HorizontalMean(Gr, &qr_tendency[0])
        # NS.write_profile('qr_evaporation', tmp[gw:-gw], Pa)
        # # cdef double[:] s_evp = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        # # sb_entropy_source_formation(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &Ref.p0_half[0],
        # #                           &DV.values[t_shift], &DV.values[tw_shift],&PV.values[qt_shift], &DV.values[qv_shift],
        # #                           &qr_tendency[0], &s_evp[0])
        # # tmp = Pa.HorizontalMean(Gr, &s_evp[0])
        # # NS.write_profile('s_evaporation', tmp[gw:-gw], Pa)
        #
        # # cdef double[:] s_heat = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        # # sb_entropy_source_heating(&Gr.dims, &DV.values[t_shift], &DV.values[tw_shift], &PV.values[qr_shift],
        # #                           &DV.values[wqr_shift],  &PV.values[w_shift], &s_heat[0])
        # # tmp = Pa.HorizontalMean(Gr, &s_heat[0])
        # # NS.write_profile('s_precip_heating', tmp[gw:-gw], Pa)
        #
        # # cdef double[:] s_drag = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        # # sb_entropy_source_drag(&Gr.dims, &DV.values[t_shift], &PV.values[qr_shift], &DV.values[wqr_shift], &s_drag[0])
        # # tmp = Pa.HorizontalMean(Gr, &s_drag[0])
        # # NS.write_profile('s_precip_drag', tmp[gw:-gw], Pa)
        #
        # # ================single_ice output section =================
        # cdef:
        #     double[:] qisi_tendency = np.empty((Gr.dims.npg,), dtype=np.double, order='c')
        #     double[:] nisi_tendency = np.empty((Gr.dims.npg,), dtype=np.double, order='c')
        # 
        # # nucleation output of qisi and nisi;
        # sb_nucleation_ice_wrapper(&Gr.dims, &self.CC.LT.LookupStructC, &DV.values[t_shift], &PV.values[nisi_shift], 
        #                         &PV.values[qisi_shift], &Ref.p0_half[0], &PV.values[qt_shift], dt, 
        #                         &nisi_tendency[0], &qisi_tendency[0])
        # tmp = Pa.HorizontalMean(Gr, &nisi_tendency[0])
        # NS.write_profile('nisi_nucleation', tmp[gw:-gw], Pa)
        # tmp = Pa.HorizontalMean(Gr, &qisi_tendency[0])
        # NS.write_profile('qisi_nucleation', tmp[gw:-gw], Pa)
        #
        # # deposition of qisi
        # sb_deposition_ice_wrapper(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &DV.values[t_shift], 
        #                         &Ref.p0_half[0], &PV.values[qt_shift], &PV.values[nisi_shift], &PV.values[qisi_shift], 
        #                         &qisi_tendency[0])
        # tmp = Pa.HorizontalMean(Gr, &qisi_tendency[0])
        # NS.write_profile('qisi_deposition', tmp[gw:-gw], Pa)
        #
        # # sublimation of qisi
        # sb_sublimation_ice_wrapper(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &DV.values[t_shift], 
        #                         &Ref.p0_half[0], &PV.values[qt_shift], &PV.values[nisi_shift], &PV.values[qisi_shift], 
        #                         &qisi_tendency[0])
        # tmp = Pa.HorizontalMean(Gr, &qisi_tendency[0])
        # NS.write_profile('qisi_sublimation', tmp[gw:-gw], Pa)
        #
        # # freezing of qisi and nisi
        # sb_freezing_ice_wrapper(&Gr.dims, self.compute_droplet_nu, self.ccn, &DV.values[t_shift], &Ref.rho0[0], 
        #                         &DV.values[ql_shift], &PV.values[qr_shift], &PV.values[nr_shift],
        #                         &qisi_tendency[0], &nisi_tendency[0])
        # tmp = Pa.HorizontalMean(Gr, &nisi_tendency[0])
        # NS.write_profile('nisi_freezing', tmp[gw:-gw], Pa)
        # tmp = Pa.HorizontalMean(Gr, &qisi_tendency[0])
        # NS.write_profile('qisi_freezing', tmp[gw:-gw], Pa)
        # 
        # # accretion of cloud ice on qisi;
        # sb_accretion_cloud_ice_wrapper(&Gr.dims, self.ccn, &Ref.rho0[0], &DV.values[ql_shift], &PV.values[qisi_shift], 
        #                         &PV.values[nisi_shift], &qisi_tendency[0])
        # tmp = Pa.HorizontalMean(Gr, &qisi_tendency[0])
        # NS.write_profile('qisi_accretion_cloud_ice', tmp[gw:-gw], Pa)
        #
        # # melting effect on qisi and nisi;
        # sb_melting_ice_wrapper(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &DV.values[t_shift], 
        #                         &DV.values[qv_shift], &PV.values[qisi_shift], &PV.values[nisi_shift], &qisi_tendency[0],
        #                         &nisi_tendency[0])
        # tmp = Pa.HorizontalMean(Gr, &nisi_tendency[0])
        # NS.write_profile('nisi_melting', tmp[gw:-gw], Pa)
        # tmp = Pa.HorizontalMean(Gr, &qisi_tendency[0])
        # NS.write_profile('qisi_melting', tmp[gw:-gw], Pa)
        #
        # tmp = Pa.HorizontalMean(Gr, &DV.values[wqisi_shift])
        # NS.write_profile('wqisi_mean', tmp[gw:-gw], Pa)
        # tmp = Pa.HorizontalMean(Gr, &DV.values[wqr_shift])
        # NS.write_profile('wqr_mean', tmp[gw:-gw], Pa)
        return
