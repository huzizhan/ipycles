#pragma once
#include "lookup.h"
#include "microphysics_arctic_1m.h"
#include "parameters.h"
#include "parameters_micro_sb.h"
#include "microphysics.h"
#include "advection_interpolation.h"
#include "entropies.h"
#include "thermodynamic_functions.h"
#include <math.h>

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

double F_v_simple(double diameter, double velocity){
    //-------------------------------------------------------------
    // INPUT VARIABLES
    //-------------------------------------------------------------
    // diameter: diameter(mass average) of particle
    // velocity: falling velocity of particle
    //-------------------------------------------------------------
    // OUTPUT VARIABLES
    //-------------------------------------------------------------
    // ventilation factor 
    //-------------------------------------------------------------
    // Reference equation: Fᵥ = aᵥ + bᵥ Nₛᵥ^1/3 * Nᵣₑ^1/2, Equ 24 in SB06;
    
    double re = diameter*velocity/KIN_VISC_AIR;
    double f_vent = 0.78 + 0.308*NSC_3*sqrt(re);
    return f_vent;
}

// Seifert & Beheng 2006: Equ 41, 85-89
double microphysics_ventilation_coefficient_ice(double Dm, double v_fall, double mass, double n, double sb_b_const, double sb_bete_const){
    //-------------------------------------------------------------
    // INPUT VARIABLES
    //-------------------------------------------------------------
    // Dm: mass weighted diameter;
    // v_fall: mass weighted falling velocity of particle;
    // mass: average mass of specific ice particle;
    // n: n-th power of moment;
    // N_re: the Reynolds number which is a function of mass N_re(mass);
    // sb_b_const: diameter-mass constant b for particle, see definition in SB06;
    // sb_bete_const: diameter-velocity constant β for particle, see definition in SB06;
    //-------------------------------------------------------------
    // OUTPUT VARIABLES
    //-------------------------------------------------------------
    // F_vn: the ventilation coefficient with n-th moment;
    //-------------------------------------------------------------

    double N_re = v_fall*Dm/KIN_VISC_AIR;
    double mu_ = 3.0; // 1/mu_ice, and mu_ice =1/3
    double nu  = 1.0;
    double a_exponent = sb_b_const + n - 1.0; 
    double b_var_tmp  = 1.5*sb_b_const + 0.5*sb_bete_const;
    double b_exponent = b_var_tmp + n - 1.0; 
    double var_const  = tgamma((nu+1.0)*mu_)/tgamma((nu+2.0)*mu_);

    double a_vent_n = A_VI * tgamma((nu+n+sb_b_const)*mu_)/tgamma((nu+1)*mu_) * pow(var_const, a_exponent);
    double b_vent_n = B_VI * tgamma((nu+n+b_var_tmp)*mu_)/tgamma((nu+1)*mu_) * pow(var_const, b_exponent);

    double F_vn = a_vent_n + b_vent_n * NSC_3 * sqrt(N_re);
    return F_vn;
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

void sb_evaporation_rain_debug(
        struct LookupStruct *LT, double (*lam_fp)(double), 
        double (*L_fp)(double, double), 
        double temperature,
        double sat_ratio, 
        double nr, 
        double qr, 
        double mu, 
        double rain_mass, 
        double Dp,
        double Dm,
        double* dpfv_stats,
        double* nr_tendency, 
        double* qr_tendency
    ){
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
        // dpfv            = A_VENT_RAIN * tgamma(mup2) * pow(Dp, mup2) + B_VENT_RAIN * NSC_3 * A_NU_SQ * tgamma(mupow) * pow(Dp, mupow) * phi_v;
        // following expression comes from cmkaul <cmkaul@gmail.com>, the default PyCLES expression.
        dpfv = (A_VENT_RAIN * tgamma(mu + 2.0) * Dp + B_VENT_RAIN * NSC_3 * A_NU_SQ * tgamma(mupow) * pow(Dp, 1.5) * phi_v)/tgamma(mu + 1.0);
        
        double g_therm = microphysics_g(LT, lam_fp, L_fp, temperature);
        qr_tendency_tmp = 2.0 * pi * g_therm * sat_ratio* nr * dpfv;

        *qr_tendency    = qr_tendency_tmp;
        *dpfv_stats = dpfv;

        // Defined in AS08, Equ(22): ∂Nᵣ/∂t = γ*Nᵣ/Lᵣ*∂Lᵣ/∂t
        *nr_tendency    = gamma /rain_mass * qr_tendency_tmp; 
    }
    return;
}

void sb_evaporation_rain(double g_therm, 
        double sat_ratio, 
        double nr, 
        double qr, 
        double mu, 
        double rain_mass, 
        double Dp,
        double Dm, 
        double* nr_tendency, 
        double* qr_tendency){
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
        // dpfv            = A_VENT_RAIN * tgamma(mup2) * pow(Dp, mup2) + B_VENT_RAIN * NSC_3 * A_NU_SQ * tgamma(mupow) * pow(Dp, mupow) * phi_v;
        // following expression comes from cmkaul <cmkaul@gmail.com>, the default PyCLES expression.
        dpfv = (A_VENT_RAIN * tgamma(mu + 2.0) * Dp + B_VENT_RAIN * NSC_3 * A_NU_SQ * tgamma(mupow) * pow(Dp, 1.5) * phi_v)/tgamma(mu + 1.0);
        
        qr_tendency_tmp = 2.0 * pi * g_therm * sat_ratio* nr * dpfv;
        *qr_tendency    = qr_tendency_tmp;

        // Defined in AS08, Equ(22): ∂Nᵣ/∂t = γ*Nᵣ/Lᵣ*∂Lᵣ/∂t
        *nr_tendency    = gamma /rain_mass * qr_tendency_tmp; 
    }
    return;
}

void sb_nucleation_ice(
        double temperature, // air temperature
        double S_i, // supper-saturation ratio over ice, POSITIVE when supper saturated, NEGATIVE when under saturated;
        double dt, // time step
        double ni, // number density of ice
        double density, //aire density
        // OUTPUT
        double* qi_tendency, 
        double* ni_tendency
    ){

    double NI_cond_immer, NI_contact;

    if (S_i >= 0.0){
        // double N_nc = 1.0e-2 * exp(0.6*(273.15 - fmax(temperature, 246.0))); // scheme from RR98;
        // double N_nc = 0.005 * exp(0.304*(273.15 - temperature)) * 1e3; // scheme from MS08(Coper62);
        // double N_nc = exp(-2.8 + 0.262*(273.15 - temperature)); // scheme from MY92;
        // double N_dn = 1.0e3 * exp(-0.639 + 12.96*S_i); // scheme adopted from SB06, Equ 36, adopted from MY92, unit m^3
        //
        double NI_cond_immer = microphysics_ice_nuclei_cond_immer_Mayer(temperature, S_i); // unit is L^-1
        double NI_contact = microphysics_ice_nuclei_contact_Young(temperature); // unit is L^-1
        double N_in = (NI_cond_immer + NI_contact) * 1000.0 / density; // convert L^-1 to m^3, then to kg^-1

        N_in = fmax(N_in, 0.0);

        if (N_in > ni && ni > SB_EPS){
            double ni_tend_tmp = (N_in - ni)/dt;
            *ni_tendency = ni_tend_tmp;
            *qi_tendency = X_ICE_NUC*ni_tend_tmp;
        }
        else{
            *qi_tendency = 0.0;
            *ni_tendency = 0.0;
        }
    }
    else{
        *qi_tendency = 0.0;
        *ni_tendency = 0.0;
    }
    return;
}

void sb_deposition_ice(
        double g_therm_ice, // thermodynamic factor
        double temperature, 
        double Dm_i, // mass-weighted diameter of ice;
        double S_i, // supper saturation over ice , POSITIVE when supper saturated, NEGATIVE when under saturated;
        double ice_mass, // ice mass 
        double velocity_ice, // falling velocity of ice
        double qi, // ice mixing ratio
        double ni, // ice number 
        double sb_b_ice,  // SB parameters for ice
        double sb_beta_ice,  // SB parameters for ice 
        // OUTPUT
        double* qi_tendency
    ){
    
    if(qi > 1e-12 && ni > 1e-12 && S_i >= 0.0){
        double F_v_mass  = microphysics_ventilation_coefficient_ice(Dm_i, 
                velocity_ice, ice_mass, 1, sb_b_ice, sb_beta_ice);
        double qi_tendency_tmp  = 4 * g_therm_ice * Dm_i * F_v_mass * S_i;
        *qi_tendency = qi_tendency_tmp;
    }
    else{
        *qi_tendency = 0.0;
    }
    return;
}

void sb_sublimation_ice(double g_therm_ice, 
        double temperature, 
        double Dm_i, 
        double S_i, 
        double ice_mass, 
        double fall_vel,
        double qi, 
        double ni, 
        double sb_b_ice, 
        double sb_beta_ice, 
        double* qi_tendency){
    // ========IN PUT ================
    // Dm_i: mass-weighted diameter of ice;
    // ice_mass: average mass of ice;
    // S_i: supper saturation over ice , POSITIVE when supper saturated, NEGATIVE when under saturated;
    // velocity_ice: falling velocity of ice;
    // sb_b_const: diameter-mass constant b for ice;
    // sb_bete_const: diameter-velocity constant β for ice;
    // ========OUT PUT================
    // qi_tendency: ∂qᵢ/∂t of sublimation, based on Equ 42 in SB06

    if(qi > 1e-11 && ni > 1e-11 && S_i < 0.0){
        double F_v_mass  = microphysics_ventilation_coefficient_ice(Dm_i, fall_vel, ice_mass, 1, sb_b_ice, sb_beta_ice);
        double qi_tendency_tmp  = 4 * g_therm_ice * Dm_i * F_v_mass * S_i;
        *qi_tendency = qi_tendency_tmp;
    }
    else{
        *qi_tendency = 0.0;
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

void microphysics_sb_collision_parameters_liquid(double sb_a_ice, double sb_b_ice, double sb_beta_ice, double k,
        double* delta_li, double* delta_l, double* vartheta_l, double* vartheta_li){
    // k: k-th moment
    double ice_mu_        = 3.0; // 1/mu_ice, and mu_ice =1/3.0
    double liquid_mu_     = 1.0; // liquid_mu_ = 1.0
    double nu             = 1.0; // both ice and cloud droplets
    double var_ice_1      = tgamma(6.0); // Γ((nu+1)/ice_mu)
    double var_ice_2      = tgamma(9.0); // Γ((nu+2)/ice_mu)
    double var_ice_3      = var_ice_1/var_ice_2;
    double var_ice_4      = (2.0*sb_b_ice + nu + 1.0 + k) * ice_mu_;
    double var_ice_5      = (sb_b_ice + nu + 1.0 +k) * ice_mu_;
    double var_ice_6      = (sb_beta_ice + sb_b_ice + nu + 1.0 + k) * ice_mu_;
    double var_ice_7      = (sb_b_ice + nu + 1.0 + k) * ice_mu_;

    double var_liquid_1   = tgamma(2.0); // Γ((nu+1)/liquid_mu)
    double var_liquid_2   = tgamma(3.0); // Γ((nu+2)/liquid_mu)
    double var_liquid_3   = var_liquid_1/var_liquid_2;
    double var_liquid_4   = 8.0 + 3.0*k; // (2*sb_b_liquid + nu + 1.0 + k)*ice_mu_
    double sb_b_liquid    = 1.0/3.0;
    double sb_beta_liquid = 2.0/3.0;
    double var_liquid_5   = 7.0 + 3.0*k; // (sb_b_liquid + nu + 1.0 + k)*ice_mu_
    double var_liquid_6   = 12.0 + 3.0*k; // (2*sb_beta_liquid + 2*sb_b_liquid + nu + 1.0 + k)*ice_mu_
   
    *delta_l     = tgamma(11.0/3.0)/tgamma(2.0) * pow(tgamma(2.0)/tgamma(3.0), 5.0/3.0);
    *delta_li    = 2.0 * tgamma(var_ice_5)/var_ice_1 * tgamma(7.0)/var_liquid_1 * pow(var_ice_3, (sb_b_ice+k)) * cbrt(var_liquid_3);
    
    *vartheta_l  = tgamma(var_liquid_6)/tgamma(var_liquid_4)*pow(tgamma(2.0)/tgamma(3.0), 4.0/3.0);
    *vartheta_li = 2.0 * tgamma(var_ice_6)/tgamma(var_ice_7) * tgamma(9.0)/tgamma(7.0) * pow(tgamma(6.0)/tgamma(9.0), sb_beta_ice) * pow(tgamma(2.0)/tgamma(3.0), sb_beta_liquid);
}

void microphysics_sb_collision_parameters_rain(double sb_a_ice, double sb_b_ice, double sb_beta_ice, double k,
        double* delta_ri, double* delta_r, double* vartheta_r, double* vartheta_ri){
    // k: k-th moment 
    
    double ice_mu_        = 3.0; // 1/mu_ice, and mu_ice =1/3.0
    double rain_mu_       = 3.0; // liquid_mu_ = 1.0
    double ice_nu         = 1.0; // both ice and cloud droplets
    double rain_nu        = -2.0/3.0; // rain_mu

    double var_ice_1      = tgamma(6.0); // Γ((ice_nu+1)/ice_mu)
    double var_ice_2      = tgamma(9.0); // Γ((ice_nu+2)/ice_mu)
    double var_ice_3      = var_ice_1/var_ice_2;
    // double var_ice_4      = (2.0*sb_b_ice + ice_nu + 1.0 + k) * ice_mu_;
    double var_ice_5      = (sb_b_ice + ice_nu + 1.0 +k) * ice_mu_;
    double var_ice_6      = (sb_beta_ice + sb_b_ice + ice_nu + 1.0 + k) * ice_mu_;
    double var_ice_7      = (sb_b_ice + ice_nu + 1.0 + k) * ice_mu_;

    double var_rain_1   = tgamma(1.0); // Γ((rain_nu+1)/rain_mu)
    double var_rain_2   = tgamma(4.0); // Γ((rain_nu+2)/rain_mu)
    double var_rain_3   = var_rain_1/var_rain_2;
    double var_rain_4   = 3.0 + 3.0*k; // (2*sb_b_rain + nu_rain + 1.0 + k)*rain_mu_
    double sb_b_rain    = 1.0/3.0;
    double sb_beta_rain = 0.226;
    double var_rain_5   = 2.0 + 3.0*k; // (sb_b_rain + rain_nu + 1.0 + k)*rain_mu_
    double var_rain_6   = (2*sb_beta_rain + 1.0 + k) * rain_mu_; // (2*sb_beta_rain + 2*sb_b_rain + rain_nu + 1.0 + k)*rain_mu_
    double var_rain_7   = (sb_beta_rain + 2.0/3.0) * rain_mu_;
   
    *delta_r     = tgamma(var_rain_4)/var_rain_1 * pow(var_rain_3, (2.0/3.0 + k));
    *delta_ri    = 2.0 * tgamma(var_ice_5)/var_ice_1 * tgamma(2.0)/var_rain_1 * pow(var_ice_3, (sb_b_ice+k)) * cbrt(var_rain_3);
    
    *vartheta_r  = tgamma(var_rain_6)/tgamma(var_rain_4) * pow(var_rain_3, 0.452);
    *vartheta_ri = 2.0 * tgamma(var_ice_6)/tgamma(var_ice_7) * tgamma(var_rain_7/tgamma(2.0)) * pow(var_ice_3, sb_beta_ice) * pow(var_rain_3, sb_beta_rain);

    return;
}

void sb_accretion_cloud_ice(double liquid_mass, double Dm_l, double velocity_l, 
        double ice_mass, double Dm_i, double velocity_i, double nl, double ql, double ni, double qi,
        double sb_a_ice, double sb_b_ice, double sb_beta_ice, double* qi_tendency){

    if(ql > SB_EPS && qi > SB_EPS && ni > SB_EPS){
        double delta_il, delta_l, vartheta_l, vartheta_il;
        double E_il = microphysics_sb_E_il(Dm_l, Dm_i);
        double n = 1.0; // 1-th moments 
        microphysics_sb_collision_parameters_liquid(sb_a_ice, sb_b_ice, sb_beta_ice, n, &delta_il, &delta_l, &vartheta_l, &vartheta_il);

        double qi_var_1 = 1.0*Dm_i*Dm_i + delta_il*Dm_l*Dm_i + delta_l*Dm_l*Dm_l;
        double qi_var_2 = 1.0*velocity_i*velocity_i - vartheta_il*velocity_i*velocity_l + vartheta_l*velocity_l*velocity_l + SIGMA_ICE;

        double qi_tendency_tmp = pi/4 * E_il * ni * ql * qi_var_1;
        *qi_tendency = qi_tendency_tmp;
    }
    else{
        *qi_tendency = 0.0;
    }
    return;
}

void sb_accretion_rain_ice(double rain_mass, double Dm_r, double velocity_r, 
        double ice_mass, double Dm_i, double velocity_i, double nr, double qr, double ni, double qi,
        double sb_a_ice, double sb_b_ice, double sb_beta_ice, double* qi_tendency){

    if(qr > SB_EPS && qi > SB_EPS && ni > SB_EPS){
        double delta_ir, delta_r, vartheta_r, vartheta_ir;
        double E_ir = 1.0;
        double n = 1.0; // 1-th moments 
        microphysics_sb_collision_parameters_rain(sb_a_ice, sb_b_ice, sb_beta_ice, n, &delta_ir, &delta_r, &vartheta_r, &vartheta_ir);

        double qi_var_1 = 1.0*Dm_i*Dm_i + delta_ir*Dm_r*Dm_i + delta_r*Dm_r*Dm_r;
        double qi_var_2 = 1.0*velocity_i*velocity_i - vartheta_ir*velocity_i*velocity_r + vartheta_r*velocity_r*velocity_r + SIGMA_ICE;

        double qi_tendency_tmp = pi/4 * E_ir * ni * qr * qi_var_1;
        *qi_tendency = qi_tendency_tmp;
    }
    else{
        *qi_tendency = 0.0;
    }
    return;
}

// Seifert & Beheng 2006: Equ 
double sb_ice_melting_thermo(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        double temperature, double qv){
    double t_3 = 273.15; // J T_3
    double lam = lam_fp(t_3);
    double L = L_fp(t_3,lam);
    double pv_sat = lookup(LT, t_3);

    // ToDo: find the right definition of D_T, and pv;
    double D_T = 1.0;
    double pv = 1.0;
    
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
                //compute the source terms
                const double nl = ccn/density[k];
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
                const double qr_tmp    = fmax(qr[ijk],0.0);
                const double nr_tmp    = fmax(fmin(nr[ijk], qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                const double qv        = qt[ijk] - ql[ijk];
                //obtain some parameters
                const double sat_ratio = microphysics_saturation_ratio(LT, temperature[ijk], p0[k], qt[ijk]);
                const double g_therm   = microphysics_g(LT, lam_fp, L_fp, temperature[ijk]);
                const double rain_mass = microphysics_mean_mass(nr_tmp, qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS);
                const double Dm        = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi);
                const double mu        = rain_mu(density[k], qr_tmp, Dm);
                const double Dp        = sb_Dp(Dm, mu);
                //compute the source terms
                sb_evaporation_rain( g_therm, sat_ratio, nr_tmp, qr_tmp, mu, rain_mass, Dp, Dm, &nr_tendency[ijk], &qr_tendency[ijk]);
            }
        }
    }
    return;
}
