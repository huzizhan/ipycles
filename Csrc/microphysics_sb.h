#pragma once
#include "lookup.h"
#include "parameters.h"
#include "parameters_micro_sb.h"
#include "microphysics.h"
#include "advection_interpolation.h"
#include "entropies.h"
#include "thermodynamic_functions.h"
// #include <cmath>
// #include <cmath>
#include <math.h>

// #define MAX_ITER  15 //maximum substep loops in source term computation
// #define RAIN_MAX_MASS  5.2e-7 //kg; DALES: 5.0e-6 kg
// #define RAIN_MIN_MASS  2.6e-10 //kg
// #define DROPLET_MIN_MASS 4.20e-15 // kg
// #define DROPLET_MAX_MASS  2.6e-10 //1.0e-11  // kg
// #define DENSITY_SB  1.225 // kg/m^3; a reference density used in Seifert & Beheng 2006, DALES
// #define XSTAR  2.6e-10
// #define KCC  10.58e9 // Constant in cloud-cloud kernel, m^3 kg^{-2} s^{-1}: Using Value in DALES; also, 9.44e9 (SB01, SS08), 4.44e9 (SB06)
// #define KCR  5.25   // Constant in cloud-rain kernel, m^3 kg^{-1} s^{-1}: Using Value in DALES and SB06;  KCR = kr = 5.78 (SB01, SS08)
// #define KRR  7.12   // Constant in rain-rain kernel,  m^3 kg^{-1} s^{-1}: Using Value in DALES and SB06; KRR = kr = 5.78 (SB01, SS08); KRR = 4.33 (S08)
// #define KAPRR  60.7  // Raindrop typical mass (4.471*10^{-6} kg to the -1/3 power), kg^{-1/3}; = 0.0 (SB01, SS08)
// #define KAPBR  2.3e3 // m^{-1} - Only used in SB06 break-up
// #define D_EQ  0.9e-3 // equilibrium raindrop diameter, m, used for SB-breakup
// #define D_EQ_MU  1.1e-3 // equilibrium raindrop diameter, m, used for SB-mu, opt=4
// #define A_RAIN_SED  9.65    // m s^{-1}
// #define B_RAIN_SED  9.796 // 10.3    # m s^{-1}
// #define C_RAIN_SED  600.0   // m^{-1}
// #define C_LIQUID_SED 702780.63036 //1.19e8 *(3.0/(4.0*pi*rho_liq))**(2.0/3.0)*np.exp(5.0*np.log(1.34)**2.0)
// #define A_VENT_RAIN  0.78
// #define B_VENT_RAIN  0.308
// #define NSC_3  0.892112 //cbrt(0.71) // Schmidt number to the 1/3 power
// #define KIN_VISC_AIR  1.4086e-5 //m^2/s kinematic viscosity of air
// #define A_NU_SQ sqrt(A_RAIN_SED/KIN_VISC_AIR)
// #define SB_EPS  1.0e-13 //small value
// #define LIQUID_DM_PREFACTOR  1.0 
// #define LIQUID_DM_EXPONENT  1.0 

// // ===========<<< ventilation parameters >>> ============
// #define A_VR  0.78 // constant ventilation coefficient for raindrops aᵥᵣ
// #define A_VI  (0.78+0.86)/2 // constant ventilation coefficient for ice aᵥᵢ
// #define B_VR  0.308 // constant ventilation coefficient for raindrops bᵥᵣ
// #define B_VI  (0.28+0.308)/2 // constant ventilation coefficient for raindrops bᵥᵢ

// // ===========<<< ice-particle parameters >>> ============
// #define ICE_MAX_MASS  5.2e-7 //kg; DALES: 5.0e-6 kg
// #define ICE_MIN_MASS  2.6e-10 //kg
// #define N_M92  1e3 // m^{-3}
// #define A_M92  -0.639
// #define B_M92  12.96
// #define X_ICE_NUC  1e-12 // kg
// #define L_MELTING  0.333e6 // J/kg latent heat of melting
// #define L_SUBLIMATION  2.834e6 // J/kg latent heat of sublimation
// #define D_L0  1.5e-5 // 15μm 
// #define D_L1  4e-5 // 40μm 
// #define D_I0  1.5e-4 // 150μm 
// #define SIGMA_ICE  0.2 // m/s

// ===========<<< Reference about SB_Liquid microphysics scheme >>> ============
// SB06: Seifert & Beheng 2006: A two-moment cloud microphysics parameterization for mixed-phase clouds. Part 1: Model description
// AS08: Axel Seifert 2008: On the Parameterization of Evaporation of Raindrops as Simulated by a One-Dimensional Rainshaft Model 
// SS08: Seifert & Stevens 2008: Understanding macrophysical outcomes of microphysical choices in simulations of shallow cumulus convection
//
// ===========<<< Keys need to know about SB_Liquid microphysics scheme >>> ============
// Unless specified otherwise, Diameter = Dm not Dp
// Note: All sb_shape_parameter_X functions must have same signature

// ===========<<< Function section of variables definition>> ============

double sb_rain_shape_parameter_0(double density, double qr, double Dm){
    //Seifert & Beheng 2001 and Seifert & Beheng 2006
    double shape_parameter = 0.0;
    return shape_parameter;
}

double sb_rain_shape_parameter_1(double density, double qr, double Dm ){
    //qr: rain specific humidity kg/kg
    //Dm is mass-weighted mean diameter
    //Seifert and Stevens 2008 and DALES v 3.1.1
    double shape_parameter = 10.0 * (1.0 + tanh( 1200.0 * (Dm - 1.4e-3) ));   // Note: UCLA-LES uses 1.5e-3
    return shape_parameter;
}

double sb_rain_shape_parameter_2(double density, double qr, double Dm){
    //qr: rain specific humidity kg/kg
    //DALES v3.2, v4.0
    double shape_parameter = fmin(30.0, -1.0+0.008*pow(qr*density, -0.6));
    return shape_parameter;
}

double sb_rain_shape_parameter_4(double density, double qr, double Dm ){
    //qr: rain specific humidity kg/kg
    //Dm: mass-weighted mean diameter
    //Seifert 2008
    double shape_parameter;
    if(Dm <= D_EQ_MU){
        shape_parameter = 6.0 * tanh((4000.0*(Dm - D_EQ_MU))*(4000.0*(Dm - D_EQ_MU))) + 1.0;
    }
    else{
        shape_parameter = 30.0 * tanh((1000.0*(Dm - D_EQ_MU))*(1000.0*(Dm - D_EQ_MU))) + 1.0;
    }
 return shape_parameter;
}

double sb_droplet_nu_0(double density, double ql){
    // ql: cloud liquid droplet specific humidity kg/kg
    // density: kg/m^3
    // Seifert & Beheng 2001, Seifert and Stevens 2008
    double nu = 0.0;
    return nu;
}

double sb_droplet_nu_1(double density, double ql){
    // ql: cloud liquid droplet specific humidity kg/kg
    // density: kg/m^3
    // Seifert & Beheng 2006
    double nu=1.0;
    return nu;
}

double sb_droplet_nu_2(double density, double ql){
    // ql: cloud liquid specific humidity kg/kg
    // density: kg/m^3
    // DALES
    double nu = 1.58 * (1000.0 * density * ql) - 0.28;
    return nu;
}

double sb_Dp(double Dm, double mu){
    // Dm: mass-weighted mean diameter
    // mu: sb_rain_shape_parameter defined based on functions above (different cases)
    // Dp: sb_rain_size_parameter defined in SS08, equ (5)
    // Here tgamma(mu+1.0)/tgamma(mu + 4.0) is same as 1/(mu + 3.0)*(mu+2.0)*(mu+1.0) in equ (5)
    double Dp = Dm * cbrt(tgamma(mu + 1.0) / tgamma(mu + 4.0));
    return Dp;
}

// ===========<<< Function section of All related microphysics processes>> ============

void sb_autoconversion_rain(double (*droplet_nu)(double,double), double density, 
        double nl, double ql, double qr, double* nr_tendency, double* qr_tendency){
    // Computation of rain specific humidity and number source terms from autoconversion of cloud liquid to rain
    double nu, phi, tau, tau_pow, droplet_mass;

    if(ql < SB_EPS){
        // if liquid specific humidity is negligibly small, set source terms to zero
        *qr_tendency = 0.0;
        *nr_tendency = 0.0;
    }
    else{
        nu = droplet_nu(density, ql);
        tau = fmin(fmax(1.0 - ql/(ql + qr), 0.0), 0.99);

        // Formulation used by DALES and Seifert & Beheng 2006
        tau_pow = pow(tau,0.7);
        phi = 400.0 * tau_pow * (1.0 - tau_pow) * (1.0 - tau_pow) * (1.0 - tau_pow);
        // Formulation used by Seifert & Beheng 2001, Seifert & Stevens 2008
        // tau_pow = pow(tau, 0.68);
        // phi = 600.0 * tau_pow * (1.0 - tau_pow)* (1.0 - tau_pow)* (1.0 - tau_pow)
        droplet_mass = microphysics_mean_mass(nl, ql, DROPLET_MIN_MASS, DROPLET_MAX_MASS);
        *qr_tendency = (KCC / (20.0 * XSTAR) * (nu + 2.0) * (nu + 4.0)/(nu * nu + 2.0 * nu + 1.0)
                        * ql * ql * droplet_mass * droplet_mass * (1.0 + phi/(1.0 - 2.0 * tau + tau * tau)) * DENSITY_SB);
        *nr_tendency = (*qr_tendency)/XSTAR;
    }
    // if prognostic cloud liquid is used, the following tendencies would also need to be computed
    //ql_tendency = -qr_tendency
    //nl_tendency = -2.0 * nr_tendency
    return;
}

void sb_accretion_rain(double density, double ql, double qr, double* qr_tendency){
    //Computation of tendency of rain specific humidity due to accretion of cloud liquid droplets
    double tau, phi;
    if(ql < SB_EPS || qr < SB_EPS){
        *qr_tendency = 0.0;
    }
    else{
        tau = fmin(fmax(1.0 - ql/(ql + qr), 0.0), 0.99);
        phi = pow((tau / (tau + 5.0e-5)), 4.0);      // - DALES, and SB06
        // phi = pow((tau/(tau+5.0e-4)),4.0);   // - SB01, SS08
        
        //SB06, DALES formulation of effective density
        *qr_tendency = KCR * ql * qr * phi * sqrt(DENSITY_SB * density); 
    }
    // if prognostic cloud liquid is used, the following tendencies would also need to be computed
    //ql_tendency = -qr_tendency
    //droplet_mass = microphysics_mean_mass(nl, ql, DROPLET_MIN_MASS, DROPLET_MAX_MASS);
    //nl_tendency = ql_tendency/droplet_mass;
    return;
}

void sb_selfcollection_breakup_rain(double density, double nr, double qr, double mu, double rain_mass, double Dm, double* nr_tendency){
    //this function gives the net tendency breakup + selfcollection: nr_tendency = -phi*nr_tendency_sc
    double lambda_rain, phi_sc, phi_bk = 0.0;
    double nr_tendency_sc;

    if(qr < SB_EPS || nr < SB_EPS){
        *nr_tendency = 0.0;
    }
    else{
        lambda_rain    = 1.0/cbrt(rain_mass * tgamma(mu + 1.0)/ tgamma(mu + 4.0));
        phi_sc         = pow((1.0 + KAPRR/lambda_rain), -9.0); //Seifert & Beheng 2006, DALES
        // phi_sc      = 1.0; //Seifert & Beheng 2001, Seifert & Stevens 2008, Seifert 2008
        nr_tendency_sc = -KRR * nr * qr * phi_sc * sqrt(DENSITY_SB*density);
        // Seifert & Stevens 2008, Seifert 2008, DALES
        if(Dm > 0.3e-3){
            phi_bk = 1000.0 * Dm - 0.1;
        }
        *nr_tendency = -phi_bk * nr_tendency_sc;

    }
    return;
}

void sb_evaporation_rain( double g_therm, double sat_ratio, double nr, double qr, double mu, double rain_mass, double Dp,
    double Dm, double* nr_tendency, double* qr_tendency){
    double gamma, dpfv, phi_v;
    const double bova      = B_RAIN_SED/A_RAIN_SED;
    const double cdp       = C_RAIN_SED * Dp;
    const double mupow     = mu + 2.5;
    const double mup2      = mu + 2.0;
    double qr_tendency_tmp = 0.0;

    if(qr < SB_EPS || nr < SB_EPS){
        *nr_tendency = 0.0;
        *qr_tendency = 0.0;
    }
    else if(sat_ratio >= 0.0){
        *nr_tendency = 0.0;
        *qr_tendency = 0.0;
    }
    else{
        gamma           = 0.7; // gamma = 0.7 is used by DALES ; 
                               // alternative expression gamma= d_eq/Dm * exp(-0.2*mu) is used by AS08, Equa23
    
        // AS08: Equ A7
        phi_v           = 1.0 - (0.5  * bova * pow(1.0 +  cdp, -mupow) + 0.125 * bova * bova * pow(1.0 + 2.0*cdp, -mupow)
                          + 0.0625 * bova * bova * bova * pow(1.0 +3.0*cdp, -mupow) + 0.0390625 * bova * bova * bova * bova * pow(1.0 + 4.0*cdp, -mupow));
        dpfv            = A_VENT_RAIN * tgamma(mup2) * pow(Dp, mup2) + B_VENT_RAIN * NSC_3 * A_NU_SQ * tgamma(mupow) * pow(Dp, mupow) * phi_v;
        // following expression comes from cmkaul <cmkaul@gmail.com>, the default PyCLES expression.
        // dpfv = (A_VENT_RAIN * tgamma(mu + 2.0) * Dp + B_VENT_RAIN * NSC_3 * A_NU_SQ * tgamma(mupow) * pow(Dp, 1.5) * phi_v)/tgamma(mu + 1.0);
        
        qr_tendency_tmp = 2.0 * pi * g_therm * sat_ratio* nr * dpfv;
        *qr_tendency    = qr_tendency_tmp;

        // Defined in AS08, Equ(22): ∂Nᵣ/∂t = γ*Nᵣ/Lᵣ*∂Lᵣ/∂t
        *nr_tendency    = gamma /rain_mass * qr_tendency_tmp; 
    }
    return;
}

void sb_nucleation_ice(double temperature, double S_i, double dt, double ni, double* qi_tendency, double* ni_tendency){

    double n_in = N_M92*exp(A_M92 + B_M92*S_i);
    double ni_tend_tmp;

    // if (temperature > T_ICE || S_i < 0.0 || ni >= n_in || ni < SB_EPS){
    if (S_i < 0.0 || ni >= n_in){
        ni_tend_tmp = 0.0;
    }
    else{
        ni_tend_tmp = (n_in - ni)/dt;
    }
    *ni_tendency = ni_tend_tmp;
    *qi_tendency = X_ICE_NUC*ni_tend_tmp;
}

void sb_deposition_ice(struct LookupStruct *LT,  double (*lam_fp)(double), double (*L_fp)(double, double),
        double temperature, double Dm_i, double S_i, double ice_mass, double fall_vel,
        double qi, double ni, double* qi_tendency, double* ni_tendency){
    
    // if(temperature > T_ICE || qi < SB_EPS || ni < SB_EPS){
    if(qi < SB_EPS || ni < SB_EPS || ice_mass < SB_EPS){
        *ni_tendency = 0.0;
        *qi_tendency = 0.0;
    }
    else if(S_i <= 0.0){
        *ni_tendency = 0.0;
        *qi_tendency = 0.0;
    }
    else{
        // double G_iv  = microphysics_g(LT, lam_fp, L_fp, temperature);
        
        double pv_sat = lookup(LT, temperature);
        double G_iv = 1.0/(Rv*temperature/DVAPOR/pv_sat + L_IV/KT/temperature * (L_IV/Rv/temperature - 1.0));

        double F_v_mass  = microphysics_ventilation_coefficient_ice(Dm_i, fall_vel, ice_mass, 1);
        double gamma = 1.0; // following same statement in rain evaporation.

        double qi_tendency_tmp  = 4 * G_iv * Dm_i * F_v_mass * S_i;

        *qi_tendency = qi_tendency_tmp;
        *ni_tendency = gamma/ice_mass * qi_tendency_tmp;
    }
    return;
}

void sb_freezing_ice(double (*droplet_nu)(double,double), double density, double temperature, 
        double liquid_mass, double rain_mass, double ql, double nl, double qr, double nr, 
        double* ql_tendency, double* qr_tendency, double* nr_tendency, double* qi_tendency, double* ni_tendency){

    // ================================================
    // ToDo: ToDo give a conditional settings of the threshold of freezing
    // ================================================
    if(qr < SB_EPS || nr < SB_EPS || rain_mass < SB_EPS){
        // if liquid specific humidity is negligibly small, set source terms to zero
        *ql_tendency = 0.0;
        *qr_tendency = 0.0;
        *nr_tendency = 0.0;
        *qi_tendency = 0.0;
        *ni_tendency = 0.0;
    }
    else{
        double ql_hom, nl_hom, qr_het, nr_het;
        double nu    = droplet_nu(density, ql);
        double J_hom = microphysics_homogenous_freezing_rate(temperature);
        double J_het = microphysics_heterogenous_freezing_rate(temperature);
        
        ql_hom = ((nu + 2)/(nu + 1)) * ql * liquid_mass * J_hom;
        nl_hom = nl * liquid_mass * J_hom;
        ql_hom = 0.0;
        nl_hom = 0.0;
        qr_het = 20 * qr * rain_mass * J_het;
        nr_het = nr * rain_mass * J_het;
        
        *ql_tendency = ql_hom;
        *qr_tendency = qr_het;
        *nr_tendency = nr_het;
        *qi_tendency = -(ql_hom + qr_het);
        *ni_tendency = -(nl_hom + nr_het);
    }
    return;
}

// ===========<<< SB06 accretion of ice and cloud droplets parameters >>> ============
// adopted in sb_accretion_cloud_ice()
// Seifert & Beheng 2006: Equ

double microphysics_sb_E_il(double Dm_l, double Dm_i){
    double E_l, E_i;
    // calculation of E_l
    if(Dm_l < 1.5e-5){
        E_l = 0.0;
    }
    else if(Dm_l <= 4.0e-5){
        E_l = (Dm_l - 1.5e-5)/2.5e-5;
    }
    else{
        E_l = 1.0;
    }
    // calculation of E_i
    if(Dm_i <= 1.5e-4){
        E_i = 0.0;
    }
    else{
        E_i = 0.8;
    }
    return E_l*E_i;
}

void microphysics_sb_collision_parameters(double sb_a_ice, double sb_b_ice, double sb_beta_ice, double k,
        double* delta_li, double* delta_l, double* delta_i, double* vartheta_l, double* vartheta_li){
    // k: k-th moment
    double ice_mu_        = 3.0; // 1/mu_ice, and mu_ice =1/3.0
    double liquid_mu_     = 1.0; // liquid_mu_ = 1.0
    double nu             = 1.0; // both ice and cloud droplets
    double var_ice_1      = gamma(6.0); // Γ((nu+1)/ice_mu)
    double var_ice_2      = gamma(9.0); // Γ((nu+2)/ice_mu)
    double var_ice_3      = var_ice_1/var_ice_2;
    double var_ice_4      = (2*sb_b_ice + nu + 1.0 + k) * ice_mu_;
    double var_ice_5      = (sb_b_ice + nu + 1.0 +k) * ice_mu_;
    double var_ice_6      = (sb_beta_ice + sb_b_ice + nu + 1.0 + k) * ice_mu_;
    double var_ice_7      = (sb_b_ice + nu + 1.0 + k) * ice_mu_;

    double var_liquid_1   = gamma(2.0); // Γ((nu+1)/liquid_mu)
    double var_liquid_2   = gamma(3.0); // Γ((nu+2)/liquid_mu)
    double var_liquid_3   = var_liquid_1/var_liquid_2;
    double var_liquid_4   = 8.0 + 3.0*k; // (2*sb_b_liquid + nu + 1.0 + k)*ice_mu_
    double sb_b_liquid    = 1.0/3.0;
    double sb_beta_liquid = 2.0/3.0;
    double var_liquid_5   = 7.0 + 3.0*k; // (sb_b_liquid + nu + 1.0 + k)*ice_mu_
    double var_liquid_6   = 12.0 + 3.0*k; // (2*sb_beta_liquid + 2*sb_b_liquid + nu + 1.0 + k)*ice_mu_
   
    *delta_i     = gamma(var_ice_4)/var_ice_1 * pow(var_ice_3, (2*sb_b_ice+k));
    *delta_l     = gamma(var_liquid_4)/var_liquid_1 * pow(var_liquid_3, (2*sb_b_liquid+k));
    *delta_li    = 2.0 * gamma(var_ice_5)/var_ice_1 * gamma(7.0)/var_liquid_1 * pow(var_ice_3, (sb_b_ice+k)) * cbrt(var_liquid_3);
    
    *vartheta_l  = gamma(var_liquid_6)/gamma(var_liquid_4)*pow(var_liquid_3, 2*sb_beta_liquid);
    *vartheta_li = 2.0 * gamma(var_ice_6)/gamma(var_ice_7) * gamma(9.0)/gamma(7.0) * pow(var_ice_3, sb_beta_ice) * pow(var_liquid_3, sb_beta_liquid);
}

void sb_accretion_cloud_ice(double liquid_mass, double Dm_l, double ice_mass, double Dm_i, 
        double velocity_i, double nl, double ql, double ni, double qi,
        double sb_a_ice, double sb_b_ice, double sb_beta_ice,
        double* nl_tendency, double* qi_tendency){
    if(ql < SB_EPS || qi < SB_EPS || ni < SB_EPS){
        *qi_tendency = 0.0;
    }
    else{
        double delta_il, delta_l, delta_i, vartheta_l, vartheta_il;
        double E_il = microphysics_sb_E_il(Dm_l, Dm_i);
        double n = 1.0; // 1-th moments 
        microphysics_sb_collision_parameters(sb_a_ice, sb_b_ice, sb_beta_ice, n, &delta_il, &delta_l, &delta_i, &vartheta_l, &vartheta_il);

        double velocity_l = LIQUID_DM_EXPONENT*pow(Dm_l, LIQUID_DM_EXPONENT);

        double qi_tendency_tmp, nl_tendency_tmp;
        double qi_var_1 = 1.0*pow(Dm_i,2) + delta_il*Dm_l*Dm_i + delta_l*pow(Dm_l,2);
        double qi_var_2 = 1.0*pow(velocity_i, 2) - vartheta_il*velocity_i*velocity_l + vartheta_l*pow(velocity_l, 2) + SIGMA_ICE;
        double nl_var_1 = pow(Dm_i,2) + Dm_l*Dm_i + pow(Dm_l,2);
        double nl_var_2 = pow(velocity_i, 2) - velocity_i*velocity_l + pow(velocity_l, 2) + SIGMA_ICE;

        qi_tendency_tmp = pi/4 * E_il * ni * ql * qi_var_1 * pow(qi_var_2,0.5);
        nl_tendency_tmp = -pi/4 * E_il * ni * ql * nl_var_1 * pow(nl_var_2,0.5);
        
        *qi_tendency = qi_tendency_tmp;
        *nl_tendency = nl_tendency_tmp;
    }
    return;
}

// Seifert & Beheng 2006: Equ 
double sb_ice_melting_thermo(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        double temperature, double qv){
    double D_T = 1.0;
    double t_3 = 273.15; // J T_3
    double lam = lam_fp(t_3);
    double L = L_fp(t_3,lam);
    double pv_sat = lookup(LT, t_3);
    double pv = 1.0;

    // ToDo: find the right definition of D_T, and pv;
    
    double melt_thermo = (KT*D_T/DVAPOR)*(temperature - t_3) + DVAPOR*L/Rv*(pv/temperature - pv_sat/t_3);
    return melt_thermo;
}

void sb_melting_ice(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        double temperature, double ice_mass, double Dm_i, double qv, double ni, double qi, 
        double* qi_tendency, double* ni_tendency){

    if( qi < SB_EPS || ni < SB_EPS){
        *qi_tendency = 0.0;
        *ni_tendency = 0.0;
    }
    else{
        double F_vl_ni, F_vl_qi;
        double G_melt = sb_ice_melting_thermo(LT, lam_fp, L_fp, temperature, qv);

        double qi_tendency_tmp = 2*pi/L_MELTING * G_melt * ni * Dm_i * pow(ice_mass, 0) * F_vl_qi; 
        double ni_tendency_tmp = 2*pi/L_MELTING * G_melt * ni * Dm_i * pow(ice_mass,-1) * F_vl_ni; 

        *qi_tendency = qi_tendency_tmp;
        *ni_tendency = ni_tendency_tmp;
    }
    return;
}

void sb_sedimentation_velocity_rain(const struct DimStruct *dims, double (*rain_mu)(double,double,double),
    double* restrict density, double* restrict nr, double* restrict qr, double* restrict nr_velocity, double* restrict qr_velocity){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin-1; k<kmax+1; k++){
                const ssize_t ijk = ishift + jshift + k;
                double qr_tmp         = fmax(qr[ijk],0.0);
                double density_factor = sqrt(DENSITY_SB/density[k]);
                double rain_mass      = microphysics_mean_mass(nr[ijk], qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS);
                double Dm             = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi);
                double mu             = rain_mu(density[k], qr_tmp, Dm);
                double Dp             = sb_Dp(Dm, mu);

                // Based on equation 21 in SB06
                // But the relationship between γ and Dp is defined in SB08
                nr_velocity[ijk] = -fmin(fmax( density_factor * (A_RAIN_SED - B_RAIN_SED * pow(1.0 + C_RAIN_SED * Dp, -mu - 1.0)) , 0.0),10.0);
                qr_velocity[ijk] = -fmin(fmax( density_factor * (A_RAIN_SED - B_RAIN_SED * pow(1.0 + C_RAIN_SED * Dp, -mu - 4.0)) , 0.0),10.0);

            }
        }
    }
     for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax-1 ; k++){
                const ssize_t ijk = ishift + jshift + k;
                nr_velocity[ijk] = interp_2(nr_velocity[ijk], nr_velocity[ijk+1]) ;
                qr_velocity[ijk] = interp_2(qr_velocity[ijk], qr_velocity[ijk+1]) ;
            }
        }
    }
    return;
}

void sb_sedimentation_velocity_liquid(const struct DimStruct *dims, double* restrict density, double ccn,
                                      double* restrict ql, double* restrict qt_velocity){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin-1; k<kmax+1; k++){
                const ssize_t ijk = ishift + jshift + k;
                double nl = ccn/density[k];
                double ql_tmp = fmax(ql[ijk],0.0);
                double liquid_mass = microphysics_mean_mass(nl, ql_tmp, DROPLET_MIN_MASS, DROPLET_MAX_MASS);
                qt_velocity[ijk] = -C_LIQUID_SED * cbrt(liquid_mass * liquid_mass);

            }
        }
    }

     for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax-1 ; k++){
                const ssize_t ijk = ishift + jshift + k;

                qt_velocity[ijk] = interp_2(qt_velocity[ijk], qt_velocity[ijk+1]) ;
            }
        }
    }

    return;
}

///==========================To facilitate output=============================

void sb_autoconversion_rain_wrapper(const struct DimStruct *dims,  double (*droplet_nu)(double,double),
                                    double* restrict density,  double ccn, double* restrict ql,  double* restrict qr,
                                    double* restrict nr_tendency, double* restrict qr_tendency){

    //Here we compute the source terms for nr and qr (number and mass of rain)
    //Temporal substepping is used to help ensure boundedness of moments

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                const double nl = ccn/density[k];
                //compute the source terms
                double ql_tmp = fmax(ql[ijk], 0.0);
                double qr_tmp = fmax(qr[ijk], 0.0);
                sb_autoconversion_rain(droplet_nu, density[k], nl, ql_tmp, qr_tmp, &nr_tendency[ijk], &qr_tendency[ijk]);


            }
        }
    }
    return;
}

void sb_accretion_rain_wrapper(const struct DimStruct *dims, double* restrict density,  double* restrict ql,
                               double* restrict qr, double* restrict qr_tendency){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                const double ql_tmp = fmax(ql[ijk], 0.0);
                const double qr_tmp = fmax(qr[ijk], 0.0);
                sb_accretion_rain(density[k], ql_tmp, qr_tmp, &qr_tendency[ijk]);

            }
        }
    }
    return;
}

void sb_selfcollection_breakup_rain_wrapper(const struct DimStruct *dims, double (*rain_mu)(double,double,double),
                                            double* restrict density, double* restrict nr, double* restrict qr, double* restrict nr_tendency){

    //Here we compute the source terms for nr and qr (number and mass of rain)
    //Temporal substepping is used to help ensure boundedness of moments
    double rain_mass, Dm, mu;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
               //obtain some parameters
                const double qr_tmp = fmax(qr[ijk],0.0);
                const double nr_tmp = fmax(fmin(nr[ijk], qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                const double rain_mass = microphysics_mean_mass(nr_tmp, qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS);
                const double Dm = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi);
                const double mu = rain_mu(density[k], qr_tmp, Dm);

                //compute the source terms
                sb_selfcollection_breakup_rain(density[k], nr_tmp, qr_tmp, mu, rain_mass, Dm, &nr_tendency[ijk]);

            }
        }
    }
    return;
}

void sb_evaporation_rain_wrapper(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                             double (*rain_mu)(double,double,double),  double* restrict density, double* restrict p0,  double* restrict temperature,  double* restrict qt,
                             double* restrict ql, double* restrict nr, double* restrict qr, double* restrict nr_tendency, double* restrict qr_tendency){

    //Here we compute the source terms for nr and qr (number and mass of rain)
    //Temporal substepping is used to help ensure boundedness of moments
    double rain_mass, Dm, mu, Dp;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                const double qr_tmp = fmax(qr[ijk],0.0);
                const double nr_tmp = fmax(fmin(nr[ijk], qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                const double qv = qt[ijk] - ql[ijk];
                const double sat_ratio = microphysics_saturation_ratio(LT, temperature[ijk], p0[k], qt[ijk]);
                const double g_therm = microphysics_g(LT, lam_fp, L_fp, temperature[ijk]);
                //obtain some parameters
                const double rain_mass = microphysics_mean_mass(nr_tmp, qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS);
                const double Dm = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi);
                const double mu = rain_mu(density[k], qr_tmp, Dm);
                const double Dp = sb_Dp(Dm, mu);
                //compute the source terms
                sb_evaporation_rain( g_therm, sat_ratio, nr_tmp, qr_tmp, mu, rain_mass, Dp, Dm, &nr_tendency[ijk], &qr_tendency[ijk]);

            }
        }
    }
    return;
}

