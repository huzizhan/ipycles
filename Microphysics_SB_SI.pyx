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
    void sb_si_microphysics_sources(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                                    double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
                                    double* density, double* p0, double* temperature,  double* qt, double ccn,
                                    double* ql, double* nr, double* qr, double* qi, double* ni, double dt, 
                                    double* nr_tendency_micro, double* qr_tendency_micro, double* nr_tendency, double* qr_tendency, 
                                    double* ni_tendency_micro, double* qi_tendency_micro, double* ni_tendency, double* qi_tendency,
                                    double* precip_rate, double* evap_rate, double* melt_rate) nogil
    void sb_si_qt_source_formation(Grid.DimStruct *dims, double* qi_tendency, double* qr_tendency, double* qt_tendency)nogil
    void sb_sedimentation_velocity_ice(Grid.DimStruct *dims, double* ni, double* qi, double* ni_velocity, double* qi_velocity) nogil
    void sb_si_entropy_source_heating_rain(Grid.DimStruct *dims, double* T, double* Twet, double* qr,
                                    double* w_qr, double* w,  double* entropy_tendency) nogil
    void sb_si_entropy_source_heating_snow(Grid.DimStruct *dims, double* T, double* Twet, double* qi,
                                    double* w_qs, double* w,  double* entropy_tendency) nogil
    void sb_si_entropy_source_drag(Grid.DimStruct *dims, double* T, double* qprec, double* w_qprec,
                                    double* entropy_tendency) nogil
    void sb_si_entropy_source_evaporation(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double),
                                    double (*L_fp)(double, double), double* p0, double* temperature,
                                    double* Twet, double* qt, double* qv, double* evap_rate, double* entropy_tendency)
    void sb_si_entropy_source_precipitation(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double),
                                    double (*L_fp)(double, double), double* p0, double* temperature,
                                    double* qt, double* qv, double* precip_rate, double* entropy_tendency)
    void sb_si_entropy_source_melt(Grid.DimStruct *dims, double* temperature, double* melt_rate, double* entropy_tendency)

cdef class Microphysics_SB_Liquid:
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

        # define the isotope tracers components 
        try:
            if namelist['isotopetracers']['use_tracers']:
                self.isotope_tracers = True
            else:
                self.isotope_tracers = False
        except:
            self.isotope_tracers = False

        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, 
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        # add prognostic variables for mass and number of rain
        PV.add_variable('nr', '1/kg', r'n_r', 'rain droplet number concentration','sym','scalar',Pa)
        PV.add_variable('qr', 'kg/kg', r'q_r', 'rain water specific humidity','sym','scalar',Pa)

        PV.add_variable('ni', '1/kg', r'n_i', 'total ice droplet number concentration','sym','scalar',Pa)
        PV.add_variable('qi', 'kg/kg', r'q_i', 'total ice water specific humidity','sym','scalar',Pa)

        # add sedimentation velocities as diagnostic variables
        DV.add_variables('w_qr', 'm/s', r'w_{qr}', 'rain mass sedimentation veloctiy', 'sym', Pa)
        DV.add_variables('w_nr', 'm/s', r'w_{nr}', 'rain number sedimentation velocity', 'sym', Pa)
        DV.add_variables('w_qi', 'm/s', r'w_{qi}', 'total ice mass sedimentation veloctiy', 'sym', Pa)
        DV.add_variables('w_ni', 'm/s', r'w_{ni}', 'total ice number sedimentation velocity', 'sym', Pa)
        if self.cloud_sedimentation:
            DV.add_variables('w_qt', 'm/s', r'w_ql', 'cloud liquid water sedimentation velocity', 'sym', Pa)
            NS.add_profile('qt_sedimentation_flux', Gr, Pa)
            NS.add_profile('s_qt_sedimentation_source',Gr,Pa)
        # add wet bulb temperature DV.add_variables('temperature_wb', 'K', r'T_{wb}','wet bulb temperature','sym', Pa) Define the precip_rate, evap_rate and melt_rate for entropy source calculation. self.precip_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c') self.evap_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c') self.melt_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c') add statistical output for the class NS.add_profile('qr_sedimentation_flux', Gr, Pa) NS.add_profile('nr_sedimentation_flux', Gr, Pa) NS.add_profile('qr_autoconversion', Gr, Pa) NS.add_profile('nr_autoconversion', Gr, Pa) NS.add_profile('qi_sedimentation_flux', Gr, Pa) NS.add_profile('ni_sedimentation_flux', Gr, Pa)
        NS.add_profile('qi_autoconversion', Gr, Pa)
        NS.add_profile('ni_autoconversion', Gr, Pa)

        NS.add_profile('s_autoconversion', Gr, Pa)
        NS.add_profile('nr_selfcollection', Gr, Pa)
        NS.add_profile('qr_accretion', Gr, Pa)
        NS.add_profile('s_accretion', Gr, Pa)
        NS.add_profile('nr_evaporation', Gr, Pa)
        NS.add_profile('qr_evaporation', Gr,Pa)
        NS.add_profile('s_evaporation', Gr,Pa)
        NS.add_profile('s_precip_heating', Gr, Pa)
        NS.add_profile('s_precip_drag', Gr, Pa)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, PrognosticVariables.PrognosticVariables PV, 
                 DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')
            Py_ssize_t nr_shift = PV.get_varshift(Gr, 'nr')
            Py_ssize_t qr_shift = PV.get_varshift(Gr, 'qr')
            Py_ssize_t ni_shift = PV.get_varshift(Gr, 'ni')
            Py_ssize_t qi_shift = PV.get_varshift(Gr, 'qi')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t w_shift = PV.get_varshift(Gr, 'w')
            double dt = TS.dt
            Py_ssize_t wqr_shift = DV.get_varshift(Gr, 'w_qr')
            Py_ssize_t wnr_shift = DV.get_varshift(Gr, 'w_nr')
            Py_ssize_t wqi_shift = DV.get_varshift(Gr, 'w_qi')
            Py_ssize_t wni_shift = DV.get_varshift(Gr, 'w_ni')
            Py_ssize_t wqt_shift
            double[:] qr_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] nr_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] qi_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] ni_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        
        sb_si_microphysics_sources(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, 
                                self.compute_rain_shape_parameter, self.compute_droplet_nu, 
                                &Ref.rho0_half[0],  &Ref.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift], self.ccn, 
                                &DV.values[ql_shift], &PV.values[nr_shift], &PV.values[qr_shift], &PV.values[qi_shift], &PV.values[ni_shift], dt,   
                                &nr_tend_micro[0], &qr_tend_micro[0], &PV.tendencies[nr_shift], &PV.tendencies[qr_shift],
                                &ni_tend_micro[0], &qi_tend_micro[0], &PV.tendencies[ni_shift], &PV.tendencies[qi_shift],
                                &self.precip_rate[0], &self.evap_rate[0], &self.melt_rate[0])
        sb_si_qt_source_formation(&Gr.dims, &qi_tend_micro[0], &qr_tend_micro[0], &PV.tendencies[qt_shift])

        # sedimentation processes of rain and single_ice: w_qr and w_qi
        sb_sedimentation_velocity_rain(&Gr.dims, self.compute_rain_shape_parameter, &Ref.rho0_half[0], &PV.values[nr_shift], &PV.values[qr_shift],
                                       &DV.values[wnr_shift], &DV.values[wqr_shift])
        sb_sedimentation_velocity_ice(&Gr.dims, &PV.values[ni_shift], &PV.values[qi_shift], &DV.values[wni_shift], &DV.values[wqi_shift])
        if self.cloud_sedimentation:
            wqt_shift = DV.get_varshift(Gr, 'w_qt')

            if self.stokes_sedimentation:
                microphysics_stokes_sedimentation_velocity(&Gr.dims,  &Ref.rho0_half[0], self.ccn, &DV.values[ql_shift], &DV.values[wqt_shift])
            else:
                sb_sedimentation_velocity_liquid(&Gr.dims,  &Ref.rho0_half[0], self.ccn, &DV.values[ql_shift], &DV.values[wqt_shift])

        # entropy source for microphysics processes
        cdef:
            Py_ssize_t tw_shift = DV.get_varshift(Gr, 'temperature_wb')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')

        microphysics_wetbulb_temperature(&Gr.dims, &self.CC.LT.LookupStructC, &Ref.p0_half[0], &PV.values[s_shift],
                                          &PV.values[qt_shift], &DV.values[t_shift], &DV.values[tw_shift])

        sb_si_entropy_source_precipitation(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &Ref.p0_half[0],
                                     &DV.values[t_shift], &PV.values[qt_shift], &DV.values[qv_shift],
                                     &self.precip_rate[0], &PV.tendencies[s_shift])

        sb_si_entropy_source_evaporation(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &Ref.p0_half[0],
                                   &DV.values[t_shift], &DV.values[tw_shift], &PV.values[qt_shift],
                                   &DV.values[qv_shift], &self.evap_rate[0], &PV.tendencies[s_shift])

        sb_si_entropy_source_melt(&Gr.dims, &DV.values[t_shift], &self.melt_rate[0], &PV.tendencies[s_shift])

        sb_si_entropy_source_heating_rain(&Gr.dims, &DV.values[t_shift], &DV.values[tw_shift], &PV.values[qr_shift],
                                  &DV.values[wqr_shift],  &PV.values[w_shift], &PV.tendencies[s_shift])

        sb_si_entropy_source_heating_snow(&Gr.dims, &DV.values[t_shift], &DV.values[tw_shift], &PV.values[qi_shift],
                                  &DV.values[wqi_shift],  &PV.values[w_shift], &PV.tendencies[s_shift])
        # entropy from rain drag
        sb_si_entropy_source_drag(&Gr.dims, &DV.values[t_shift], &PV.values[qr_shift], &DV.values[wqr_shift],
                            &PV.tendencies[s_shift])
        # entropy from ice drag
        sb_si_entropy_source_drag(&Gr.dims, &DV.values[t_shift], &PV.values[qi_shift], &DV.values[wqi_shift],
                            &PV.tendencies[s_shift])

        return
    
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, 
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
    #     cdef:
    #         Py_ssize_t i, j, k, ijk
    #         Py_ssize_t gw = Gr.dims.gw
    #         Py_ssize_t imax = Gr.dims.nlg[0]
    #         Py_ssize_t jmax = Gr.dims.nlg[1]
    #         Py_ssize_t kmax = Gr.dims.nlg[2]
    #         Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
    #         Py_ssize_t jstride = Gr.dims.nlg[2]
    #         Py_ssize_t ishift, jshift
    #
    #         Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature') Py_ssize_t tw_shift = DV.get_varshift(Gr, 'temperature_wb') Py_ssize_t qv_shift = DV.get_varshift(Gr, 'qv') Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql') Py_ssize_t nr_shift = PV.get_varshift(Gr, 'nr') Py_ssize_t qr_shift = PV.get_varshift(Gr, 'qr') Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt') Py_ssize_t w_shift = PV.get_varshift(Gr, 'w') double[:] qr_tendency = np.empty((Gr.dims.npg,), dtype=np.double, order='c') double[:] nr_tendency = np.empty((Gr.dims.npg,), dtype=np.double, order='c') double[:] tmp
    #         double[:] dummy =  np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
    #         Py_ssize_t wqr_shift = DV.get_varshift(Gr, 'w_qr')
    #         Py_ssize_t wnr_shift = DV.get_varshift(Gr, 'w_nr')
    #         Py_ssize_t wqt_shift
    #
    #     cdef double[:] s_src =  np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
    #     if self.cloud_sedimentation:
    #         wqt_shift = DV.get_varshift(Gr,'w_qt')
    #
    #         compute_advective_fluxes_a(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &DV.values[wqt_shift], &DV.values[ql_shift], &dummy[0], 2, self.order)
    #         tmp = Pa.HorizontalMean(Gr, &dummy[0])
    #         NS.write_profile('qt_sedimentation_flux', tmp[gw:-gw], Pa)
    #
    #         compute_qt_sedimentation_s_source(&Gr.dims, &Ref.p0_half[0], &Ref.rho0_half[0], &dummy[0],
    #                                 &PV.values[qt_shift], &DV.values[qv_shift],&DV.values[t_shift], &s_src[0], self.Lambda_fp,
    #                                 self.L_fp, Gr.dims.dx[2], 2)
    #         tmp = Pa.HorizontalMean(Gr, &s_src[0])
    #         NS.write_profile('s_qt_sedimentation_source', tmp[gw:-gw], Pa)
    #
    #     #compute sedimentation flux only of nr
    #     compute_advective_fluxes_a(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &DV.values[wnr_shift], &PV.values[nr_shift], &dummy[0], 2, self.order)
    #     tmp = Pa.HorizontalMean(Gr, &dummy[0])
    #     NS.write_profile('nr_sedimentation_flux', tmp[gw:-gw], Pa)
    #
    #     #compute sedimentation flux only of qr
    #     compute_advective_fluxes_a(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &DV.values[wqr_shift], &PV.values[qr_shift], &dummy[0], 2, self.order)
    #     tmp = Pa.HorizontalMean(Gr, &dummy[0])
    #     NS.write_profile('qr_sedimentation_flux', tmp[gw:-gw], Pa)
    #
    #     #note we can re-use nr_tendency and qr_tendency because they are overwritten in each function
    #     #must have a zero array to pass as entropy tendency and need to send a dummy variable for qt tendency
    #
    #     # Autoconversion tendencies of qr, nr, s
    #     sb_autoconversion_rain_wrapper(&Gr.dims,  self.compute_droplet_nu, &Ref.rho0_half[0], self.ccn,
    #                                    &DV.values[ql_shift], &PV.values[qr_shift], &nr_tendency[0], &qr_tendency[0])
    #     tmp = Pa.HorizontalMean(Gr, &nr_tendency[0])
    #     NS.write_profile('nr_autoconversion', tmp[gw:-gw], Pa)
    #     tmp = Pa.HorizontalMean(Gr, &qr_tendency[0])
    #     NS.write_profile('qr_autoconversion', tmp[gw:-gw], Pa)
    #     cdef double[:] s_auto =  np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
    #     sb_entropy_source_formation(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &Ref.p0_half[0],
    #                               &DV.values[t_shift], &DV.values[tw_shift],&PV.values[qt_shift], &DV.values[qv_shift],
    #                               &qr_tendency[0], &s_auto[0])
    #
    #     tmp = Pa.HorizontalMean(Gr, &s_auto[0])
    #     NS.write_profile('s_autoconversion', tmp[gw:-gw], Pa)
    #
    #
    #     # Accretion tendencies of qr, s
    #     sb_accretion_rain_wrapper(&Gr.dims, &Ref.rho0_half[0], &DV.values[ql_shift], &PV.values[qr_shift], &qr_tendency[0])
    #     tmp = Pa.HorizontalMean(Gr, &qr_tendency[0])
    #     NS.write_profile('qr_accretion', tmp[gw:-gw], Pa)
    #     cdef double[:] s_accr =  np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
    #     sb_entropy_source_formation(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &Ref.p0_half[0],
    #                               &DV.values[t_shift], &DV.values[tw_shift],&PV.values[qt_shift], &DV.values[qv_shift],
    #                               &qr_tendency[0], &s_accr[0])
    #     tmp = Pa.HorizontalMean(Gr, &s_accr[0])
    #     NS.write_profile('s_accretion', tmp[gw:-gw], Pa)
    #
    #     # Self-collection and breakup tendencies (lumped) of nr
    #     sb_selfcollection_breakup_rain_wrapper(&Gr.dims, self.compute_rain_shape_parameter, &Ref.rho0_half[0],
    #                                            &PV.values[nr_shift], &PV.values[qr_shift], &nr_tendency[0])
    #     tmp = Pa.HorizontalMean(Gr, &nr_tendency[0])
    #     NS.write_profile('nr_selfcollection', tmp[gw:-gw], Pa)
    #
    #     # Evaporation tendencies of qr, nr, s
    #     sb_evaporation_rain_wrapper(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp,
    #                                 self.compute_rain_shape_parameter, &Ref.rho0_half[0], &Ref.p0_half[0],
    #                                 &DV.values[t_shift], &PV.values[qt_shift], &DV.values[ql_shift],
    #                                 &PV.values[nr_shift], &PV.values[qr_shift], &nr_tendency[0], &qr_tendency[0])
    #
    #     tmp = Pa.HorizontalMean(Gr, &nr_tendency[0])
    #     NS.write_profile('nr_evaporation', tmp[gw:-gw], Pa)
    #     tmp = Pa.HorizontalMean(Gr, &qr_tendency[0])
    #     NS.write_profile('qr_evaporation', tmp[gw:-gw], Pa)
    #     cdef double[:] s_evp =  np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
    #     sb_entropy_source_formation(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &Ref.p0_half[0],
    #                               &DV.values[t_shift], &DV.values[tw_shift],&PV.values[qt_shift], &DV.values[qv_shift],
    #                               &qr_tendency[0], &s_evp[0])
    #     tmp = Pa.HorizontalMean(Gr, &s_evp[0])
    #     NS.write_profile('s_evaporation', tmp[gw:-gw], Pa)
    #
    #
    #     cdef double[:] s_heat =  np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
    #     sb_entropy_source_heating(&Gr.dims, &DV.values[t_shift], &DV.values[tw_shift], &PV.values[qr_shift],
    #                               &DV.values[wqr_shift],  &PV.values[w_shift], &s_heat[0])
    #     tmp = Pa.HorizontalMean(Gr, &s_heat[0])
    #     NS.write_profile('s_precip_heating', tmp[gw:-gw], Pa)
    #
    #     cdef double[:] s_drag =  np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
    #     sb_entropy_source_drag(&Gr.dims, &DV.values[t_shift], &PV.values[qr_shift], &DV.values[wqr_shift], &s_drag[0])
    #     tmp = Pa.HorizontalMean(Gr, &s_drag[0])
    #     NS.write_profile('s_precip_drag', tmp[gw:-gw], Pa)

        return
