#pragma once
#include "parameters.h"
#include "parameters_micro_sb.h"
#include "microphysics.h"
#include "microphysics_sb.h"
#include "advection_interpolation.h"
#include "entropies.h"
#include "thermodynamic_functions.h"
#include "thermodynamics_sa.h"
#include "advection_interpolation.h"
#include <math.h>

// ===========<<< SB Ice phase parameterization >>> ============
// Basic physic parameters definition based on Table 1 in Seifert and Beheng 2006

// ------ Liquid Cloud Droplets coefficients-----------
#define SB_LIQUID_A 0.124 // m kg^-β
#define SB_LIQUID_B 0.333333
#define SB_LIQUID_alpha 3.75e5
#define SB_LIQUID_beta 0.666666
#define SB_LIQUID_lamuda 1.0
#define SB_LIQUID_nu 1.0
#define SB_LIQUID_mu 1.0

// ------ Ice Cloud Particle coefficients-----------
#define SB_ICE_A 0.217 // m kg^-β
#define SB_ICE_B 0.302
#define SB_ICE_alpha 317.0
#define SB_ICE_beta 0.363
#define SB_ICE_lamuda 0.5
#define SB_ICE_nu 1.0
#define SB_ICE_mu 0.333333 // 1/3

// ------ Rain coefficients-----------
#define SB_RAIN_A 0.124 // m kg^-β
#define SB_RAIN_B 0.333333
#define SB_RAIN_alpha 159.0
#define SB_RAIN_beta 0.266
#define SB_RAIN_lamuda 0.5
#define SB_RAIN_nu -0.666666
#define SB_RAIN_mu 0.333333 // 1/3
// TODO: those rain coefficients need to change based on table 1 in SB06

// ------ Snow coefficients-----------
#define SB_SNOW_A 8.156 // m kg^-β
#define SB_SNOW_B 0.526
#define SB_SNOW_alpha 27.7
#define SB_SNOW_beta 0.216
#define SB_SNOW_lamuda 0.5
#define SB_SNOW_nu 1.0
#define SB_SNOW_mu 0.333333 // 1/3
#define SB_SNOW_MIN_MASS 1.73e-9
#define SB_SNOW_MAX_MASS 1.0e-7

// ----- Ice Nucleation Coefficients-----------
// -- In Spring Condition
#define T_NUC 268.15
#define A_IME 1.5684e5
#define B_IME 0.2466
#define C_IME 1.2293
#define A_DEP 1.7836e5
#define B_DEP 0.0075
#define C_DEP 2.0341

// ------Ice self-collection Coefficients-----------
#define ICE_CRIT_II 8.3e-7 // specific humidity threshold for ice_selfcollection 1.0e-6(kg/m3) ÷ 1.204(kg/m3), based on ICON code
#define ICE_D_CRIT_II 5.0e-6 // D-threshold for ice self_collection

// ------Snow self-collection Coefficients----------
// #define SNOW_CRIT_SS 8.3e-10 // specific humidity threshold for snow self_collection 1.0e-

// ------Snow Riming Coefficients------------
#define LIQUID_CRIT_RIMING 8.3e-7 // specific humidity threshold for ice_cloud_riming and snow_cloud_riming
#define LIQUID_D_CRIT_RIMING 15.0e-6 // diameter threshold for ice_cloud_riming and snow_cloud_riming
#define SNOW_D_CRIT_RIMING 150.0-6 // diameter threshold for ice_cloud_riming and snow_cloud_riming
#define SNOW_E_MAX 0.8; // max collision efficiencies for snow to collect other species
#define RAIN_CRIT_RIMING 8.3e-6 // specific humidity threshold for ice_rain_riming and snow_rain_riming
#define RAIN_D_CRIT_RIMING 100.0e-6 // D-threshold for ice_rain_riming and snow_rain_riming
#define D_CONV_SG 200.0e-6 // D-threshold for conversion of snow to graupel
#define D_CONV_IG 200.0e-6 // D-threshold for conversion of ice to graupel

// ------Ice Multiplication Coefficients------------
#define T_MULT_MIN 265.0 // K
#define T_MULT_MAX 270.0 // K
#define T_MULT_OPT 268.0 //
#define C_MULT 3.5e8 // TODO find the defination of this variable

// ------Riming of snow to graupel------------------
#define T_MAX_GR_RIME 270.16 // K
                             //
// ------ Other parameters adopted from Arc1M scheme
#define SB_N_ICE_MIN 1.579437940972532e+17 //
#define SB_N_ICE_MAX 21601762742.634903 //

#define N_MAX_ICE 1.579437940972532e+17
#define N_MIN_ICE 21601762742.634903

#define T_3 273.15 

// ==================== Warm Phase Process of SB scheme ==========================

void sb_cloud_activation_hdcp(
        double p0,
        double qv,
        double ql,
        double nl,
        double w,
        double dt,
        double S,
        // double* diag_1,
        double* ql_tendency,
        double* nl_tendency
    ){
    double n_nuc, q_nuc;

    // if (ql > SB_EPS && w > 0.0){
    if (S > 0.0 && w > 0.0){
        const double A_P = 183230691.161 * atan(0.0001984051994*p0 - 16.2420263911) + 287736034.13;
        const double B_P = 0.10147358938 * atan(4.473190485e-05*p0 - 3.22011836758) + 0.6258809883;
        const double C_P = -0.2922395814 * atan(0.0001843225275*p0 - 13.8499423719) + 0.8907491812;
        const double D_P = 229189886.226 * atan(0.0001986158191*p0 - 16.2461600644) + 360848977.55;

        n_nuc = A_P * atan(B_P * log(w) + C_P) + D_P;
        n_nuc = fmax(fmax(n_nuc, 10.0e-6) - nl, 0.0);

        q_nuc = fmin(n_nuc*LIQUID_MIN_MASS, qv);
        n_nuc = q_nuc/LIQUID_MIN_MASS;

        *ql_tendency = q_nuc/dt;
        *nl_tendency = n_nuc/dt;
    }
    else{
        *ql_tendency = 0.0;
        *nl_tendency = 0.0;
    }
    // *diag_1 = n_nuc;
    return;
}

void sb_ccn(
    double C_ccn,
    double S,
    double dS,
    double dzi,
    double w,
    // double* diag_1,
    // double* diag_2,
    double* ql_tendency,
    double* nl_tendency
    ){
    const double kappa_mar = 0.462; // maritime conditions
    const double kappa_con = 0.462; // continental conditions
    const double S_max = 1.1;
    const double var_1 = dS*dzi*w;
    const double cloud_x_min = 1.0e-12;

    double nc_dt, qc_dt;
    if (S > 0.0 && S < S_max && var_1 > 0.0){
        *nl_tendency += C_ccn * kappa_mar * pow(S,kappa_mar) * var_1;
        *ql_tendency += *nl_tendency * cloud_x_min;
        // *nl_tendency += 1.0;
        // *ql_tendency += 1.0;
    }
    else{
        *nl_tendency += 0.0;
        *ql_tendency += 0.0;
    }
    // *diag_1 = var_1;
    // *diag_2 = *nl_tendency;
    return;
}

void sb_autoconversion_rain_tmp(
        // INPUT
        double (*droplet_nu)(double,double), 
        double density, 
        double nl, 
        double ql, 
        double qr, 
        // OUTPUT
        double* nr_tendency, double* qr_tendency,
        double* nl_tendency, double* ql_tendency){
  // Computation of rain specific humidity and number source terms from
  // autoconversion of cloud liquid to rain
  double nu, phi, tau, tau_pow, droplet_mass;

  if (ql < SB_EPS && nl <1.0e5) {
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
    *ql_tendency = -*qr_tendency;
    *nl_tendency = -2.0 * (*nr_tendency);
    return;
}

void sb_accretion_rain_tmp(
        // INPUT
        double density, 
        double ql, 
        double qr,
        // OUTPUT
        double droplet_mass,
        double* qr_tendency,
        double* ql_tendency,
        double* nl_tendency
    ){
    // Computation of tendency of rain specific humidity due to accretion of
    // cloud liquid droplets
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
    *ql_tendency = - *qr_tendency;
    *nl_tendency = *ql_tendency/droplet_mass;

    return;
}

void sb_evaporation_rain_tmp(
        // INPUT
        double g_therm, 
        double sat_ratio, 
        double nr,
        double qr, 
        double mu, 
        double rain_mass, 
        double Dp,
        double Dm, 
        // OUTPUT
        double* nr_tendency, 
        double* qr_tendency,
        double* qv_tendency
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
        dpfv            = A_VENT_RAIN * tgamma(mup2) * pow(Dp, mup2) + B_VENT_RAIN * NSC_3 * A_NU_SQ * tgamma(mupow) * pow(Dp, mupow) * phi_v;
        // following expression comes from cmkaul <cmkaul@gmail.com>, the default PyCLES expression.
        // dpfv = (A_VENT_RAIN * tgamma(mu + 2.0) * Dp + B_VENT_RAIN * NSC_3 * A_NU_SQ * tgamma(mupow) * pow(Dp, 1.5) * phi_v)/tgamma(mu + 1.0);
        
        qr_tendency_tmp = 2.0 * pi * g_therm * sat_ratio* nr * dpfv;
        *qr_tendency    = qr_tendency_tmp;

        // Defined in AS08, Equ(22): ∂Nᵣ/∂t = γ*Nᵣ/Lᵣ*∂Lᵣ/∂t
        *nr_tendency    = gamma /rain_mass * qr_tendency_tmp; 
    }
    // compute vapor tendency source
    *qv_tendency = - *qr_tendency;

    return;
}

double microphysics_g_sb_ice(
        // thermodynamic settings
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        // INPUT VARIABLES
        double temperature, 
        double dvapor, // vapor diffusivity
        double kappa_t // heat conductivity
        ){

    double lam = lam_fp(temperature);
    double L = L_fp(temperature,lam);
    // double pv_sat_i = lookup(LT, temperature);
    double pv_sat_i = saturation_vapor_pressure_ice(temperature);

    double g_therm_ice = 1.0/(Rv*temperature/dvapor/pv_sat_i + L_IV/kappa_t/temperature * (L_IV/Rv/temperature - 1.0));
    // double g_therm = 1.0/(Rv*temperature/DVAPOR/pv_sat + L/KT/temperature * (L/Rv/temperature - 1.0));
    return g_therm_ice;
}

// see equation 43 and Appendix B in SB06
double sb_ventilation_coefficient(
        double n,        // n-th moment;
        double Dm,       // mass weighted diameter
        double velocity, // falling velocity of particle
        double mass,     // mass
        double sb_b,     // mass-related parameter b of particle
        double sb_beta,  // mass-related parameter beta of particle
        double nu,       // ν of particle
        double mu        // μ of particle
        ){

    // see the definition of N_re in equation 42 in SB06
    double N_re = velocity*Dm/KIN_VISC_AIR;
    
    // next section follows equaiton 88 on a_vent_n, and equation 90 on b_vent_n
    double var_1 = tgamma( (nu + n + sb_b)/mu );
    double var_2 = tgamma( (nu + 1.0)/mu );
    double var_3 = var_2/tgamma( (nu + 2.0)/mu );
    double var_4 = tgamma( (nu + n + 1.5*sb_b + 0.5*sb_beta)/mu );
    double exponent_a = sb_b + n - 1.0;
    double exponent_b = 1.5*sb_b + 0.5*sb_beta + n - 1;

    double a_vent_n = A_VI * var_1/var_2 * pow(var_3, exponent_a);
    double b_vent_n = B_VI * var_4/var_2 * pow(var_3, exponent_b);

    double F_vn = a_vent_n + b_vent_n * NSC_3 * sqrt(N_re);
    return F_vn;
}

// following the equation 90 in SB06
double sb_collection_delta_b(
        double k,    // k-th moment
        double sb_b, // mass-related parameter b of particle
        double nu,   // ν of particle
        double mu    // μ of particle
        ){

    double var_1 = tgamma( (2.0*sb_b + nu + 1.0 + k) / mu );
    double var_2 = tgamma( (nu + 1.0) / mu );
    double var_3 = tgamma( (nu + 2.0) / mu );
    double var_exponent = 2.0*sb_b + k; 

    return (var_1/var_2) * pow( (var_2/var_3), var_exponent );
}

// following the equation 91 in SB06
double sb_collection_delta_ab(
        double k,      // k-th moment
        double sb_b_a, // mass-related parameter b of particle a
        double nu_a,   // ν of particle a
        double mu_a,   // μ of particle a
        double sb_b_b, // mass-related parameter b of particle b
        double nu_b,   // ν of particle b
        double mu_b    // μ of particle b
        ){
    
    double var_a_1 = tgamma( (sb_b_a + nu_a + 1.0 + k) / mu_a );
    double var_a_2 = tgamma( (nu_a + 1.0) / mu_a );
    double var_a_3 = tgamma( (nu_a + 2.0) / mu_a );

    double var_b_1 = tgamma( (sb_b_b + nu_b + 1.0) / mu_b );
    double var_b_2 = tgamma( (nu_b + 1.0) / mu_b );
    double var_b_3 = tgamma( (nu_b + 2.0) / mu_b );

    return 2.0 * (var_a_1/var_a_2) * (var_b_1/var_b_2) * 
           pow((var_a_2/var_a_3), (sb_b_a + k)) * 
           pow((var_b_2/var_b_3), sb_b_b);
}

// following the equation 92 in SB06
double sb_collection_vartheta_b(
        double k,       // k-th moment
        double sb_b,    // mass-related parameter b of particle
        double sb_beta, // mass-related parameter β of particle
        double nu,      // ν of particle
        double mu       // μ of particle
        ){

    double var_1 = tgamma( (2.0*sb_beta + 2.0*sb_b + nu + 1.0 + k) / mu );
    double var_2 = tgamma( (2.0*sb_b + nu + 1.0 + k) / mu  );
    double var_3 = tgamma( (nu + 1.0) / mu ) / tgamma( (nu + 2.0) / mu );
    double exponent = 2.0*sb_beta;

    return (var_1/var_2) * pow(var_3, exponent);
}

// following the equation 93 in SB06
double sb_collection_vartheta_ab(
        double k,         // k-th moment
        double sb_b_a,    // mass-related parameter b of particle a
        double sb_beta_a, // mass-related parameter β of particle a
        double nu_a,      // ν of particle a
        double mu_a,      // μ of particle a
        double sb_b_b,    // mass-related parameter b of particle b
        double sb_beta_b, // mass-related parameter β of particle b
        double nu_b,      // ν of particle b
        double mu_b       // μ of particle b
        ){
    
    double var_a_1 = tgamma( (sb_beta_a + sb_b_a + nu_a + 1.0 + k) / mu_a );
    double var_a_2 = tgamma( (sb_b_a + nu_a + 1.0 + k) / mu_a );
    double var_a_3 = tgamma( (nu_a + 1.0) / mu_a );
    double var_a_4 = tgamma( (nu_a + 2.0) / mu_a );

    double var_b_1 = tgamma( (sb_beta_b + sb_b_b + nu_b + 1.0) / mu_b );
    double var_b_2 = tgamma( (sb_b_b + nu_b + 1.0) / mu_b );
    double var_b_3 = tgamma( (nu_b + 1.0) / mu_b );
    double var_b_4 = tgamma( (nu_b + 2.0) / mu_b );

    return 2.0 * (var_a_1/var_a_2) * (var_b_1/var_b_2) * 
           pow((var_a_3/var_a_4), sb_beta_a) * 
           pow((var_b_3/var_b_4), sb_beta_b);
}

// see equation 67 in SB06
double sticking_efficiencies(double T){
    return fmax(0.1, fmin(exp(0.09*(T-T_3)), 1.0));
}

double cotton_efficiency(double T){
    double base = pow(10.0, (0.035*(T-T_3)-0.7));
    return fmin(base, 0.2);
}
// this component follows the equation 64-66 in SB06
double collection_efficiencies_cloud(double Dm_e, double Dm_l){
    double E_l, E_e;
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
    // calculation of E_e
    if(Dm_e <= 1.5e-4){
        E_e = 0.0;
    }
    else{
        E_e = 0.8;
    }
    return E_l*E_e;
}

// ------------------- Ice Process --------------------

double mayer_dep_immer(double satratio){
    double n_dep_immer;
    n_dep_immer = exp(-0.639 + 12.96*satratio);
    n_dep_immer = n_dep_immer * 1e3;
    return n_dep_immer;
}

double mayer_contact(double T){
    double n_contact;
    n_contact = exp(-2.8 + 0.262*(273.15 - T)); // unit: L^-1
    n_contact = n_contact*1e3; // unit m^3
    return n_contact;
}

double young_contact_simple(double IN, double T){
    double n_contact;
    // double N_a0 = 2.0e2; // unit L^-1
    n_contact = IN * pow((270.16 - T), 1.3);
    n_contact = n_contact*1e3; // unit m^3
    return n_contact;
}

void sb_ice_nucleation_mayer(
        // thermodynamic settings
        struct LookupStruct *LT,
        // INPUT VARIABLES
        double IN,
        double T, 
        double qt,         // total water specific humidity
        double p0,         // air pressure
        double qv,
        double ni,
        double dt,
        // double* diag_1,
        // double* diag_2,
        // double* diag_3,
        double* qi_tendency,
        double* ni_tendency
    ){

    double T_c = T - T_3;
    double n_nuc, q_nuc;

    // calculate the sat_ratio with look up table method 
    const double pv_star = lookup(LT, T);
    const double qv_star = qv_star_c(p0, qt, pv_star);
    const double satratio = qt/qv_star - 1.0;
    // const double satratio = microphysics_saturation_ratio_ice(T, p0, qt);

    double ni_diag = 0.0;

    if (satratio >= 0.0 && T_c >= -30.0 && T_c < -5.0){

        ni_diag += mayer_dep_immer(satratio);
        ni_diag += young_contact_simple(IN, T);

        // *diag_1 = mayer_dep_immer(satratio);
        // *diag_2 = young_contact_simple(T);
        // *diag_3 = mayer_contact(T);
    }
    else{
        ni_diag += 0.0;
    }

    if (ni_diag > ni){
        n_nuc = fmax((ni_diag-ni), 0.0);
        q_nuc = fmin(n_nuc*ICE_MIN_MASS, qv);
        n_nuc = q_nuc/ICE_MIN_MASS;

        *ni_tendency = n_nuc/dt;
        *qi_tendency = q_nuc/dt;
    }
    else{
        *ni_tendency = 0.0;
        *qi_tendency = 0.0;
    }

    return;
}

double hdcp_immersion(double T, double ql){
    double n_immer;
    if(ql > SB_EPS){
        T = fmax(T, 237.1501);
        if (T < 261.15){
            // n_immer = 8.1909e4 * exp( - 0.2290 * pow((T - 237.15), 1.2553)); // year
            // *diag_1 = 2.9694e4 * exp( - 0.2813 * pow((T - 237.15), 1.1778)); // summer 
            // *diag_2 = 4.9920e4 * exp( - 0.2622 * pow((T - 237.15), 1.2044)); // fall
            // *diag_3 = 1.0259e5 * exp( - 0.2073 * pow((T - 237.15), 1.2873)); // winter
            n_immer = A_IME * exp( - B_IME * pow((T - 237.15), C_IME)); // spring
        }
        else{
            n_immer = 0.0;
        }
    }
    else{
        n_immer = 0.0;
    }
    return n_immer;
}

double hdcp_deposition(double T, double satratio){
    double n_dep;
    double c_in, dsf;
    T = fmax(T, 220.001);
    if (T < 253.0){
        c_in = A_DEP * exp( - B_DEP * pow((T - 237.15), C_DEP));
        dsf = 0.27626 * atan(6.21*satratio - 1.3107) + 2.6789;
        n_dep = c_in * dsf;
    }
    else{
        n_dep = 0.0;
    }
    return n_dep;
}

double hdcp_contact(double T){
    double n_contact;
    n_contact = 1.0;
    return n_contact;
}

void sb_ice_nucleation_hdcp(
        // thermodynamic settings
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
        // INPUT VARIABLES
        double T, 
        double qt,         // total water specific humidity
        double p0,         // air pressure
        double qv,
        double ql,
        double ni,
        double dt,
        double S_i,
        // double* diag_1,
        // double* diag_2,
        // double* diag_3,
        double* qi_tendency,
        double* ni_tendency
    ){
    double n_nuc, q_nuc;

    // calculate the sat_ratio with look up table method 
    // const double pv_star = lookup(LT, T);
    // const double qv_star = qv_star_c(p0, qt, pv_star);
    // const double satratio = qt/qv_star - 1.0;
    const double satratio = microphysics_saturation_ratio_ice(T, p0, qt);

    const double ni_het_max = 1.0e4; // number density per liter
                                     //
    double ni_diag = 0.0;
    
    if(T < T_NUC && T > 180.0 && satratio > 0.0 && ni < ni_het_max){
        ni_diag += hdcp_immersion(T, ql);
        ni_diag += hdcp_deposition(T, satratio);
        ni_diag += mayer_contact(T);
    }
    
    if (ni_diag > ni){
        n_nuc = fmax((ni_diag-ni), 0.0);
        q_nuc = fmin(n_nuc*ICE_MIN_MASS, qv);
        n_nuc = q_nuc/ICE_MIN_MASS;

        *ni_tendency = n_nuc/dt;
        *qi_tendency = q_nuc/dt;
    }
    else{
        *ni_tendency = 0.0;
        *qi_tendency = 0.0;
    }

    return;
}

void sb_freezing(double (*droplet_nu)(double,double), 
        double density, 
        double temperature, 
        double liquid_mass, 
        double rain_mass, 
        double ql, 
        double nl, 
        double qr, 
        double nr, 
        double* ql_tendency, 
        double* nl_tendency,
        double* qr_tendency, 
        double* nr_tendency
    ){
    // homogeneous freezing of cloud droplets under -30 C 
    // heterogeneous freezing of rain droplets between -30 ~ 0 C

    if(qr < SB_EPS || nr < SB_EPS || ql < SB_EPS){
        // if liquid specific humidity is negligibly small, set source terms to zero
        *ql_tendency = 0.0;
        *nl_tendency = 0.0;
        *qr_tendency = 0.0;
        *nr_tendency = 0.0;
    }
    else{
        double ql_hom, nl_hom, qr_het, nr_het;
        double nu    = droplet_nu(density, ql);
        // J_hom of cloud droplets under -30 C 
        // follows Equ 12 in Cotton&Field2002
        // Same as the code in ICON
        double J_hom = microphysics_homogenous_freezing_rate(temperature);
        ql_hom = ((nu + 2.0)/(nu + 1.0)) * ql * liquid_mass * J_hom;
        nl_hom = nl * liquid_mass * J_hom;

        // J_het of rain droplets follows Equ 44 in SB06
        double J_het = microphysics_heterogenous_freezing_rate(temperature);
        qr_het = 20.0 * qr * rain_mass * J_het;
        nr_het = nr * rain_mass * J_het;
        
        *ql_tendency = ql_hom;
        *nl_tendency = nl_hom;
        *qr_tendency = qr_het;
        *nr_tendency = nr_het;
    }
    return;
}

void sb_ice_deposition(
        // thermodynamic settings
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
        // INPUT VARIABLES
        double T, 
        double qt,         // total water specific humidity
        double p0,         // air pressure
        double qi,         // ice specific humidity
        double ni,         // ice number density
        double Dm_i,       // mass-weighted diameter of ice
        double mass_i,     // average mass of ice
        double velocity_i, // falling velocity of ice
        double dt, 
        double S_i,
        // OUTPUT VARIABLES INDEX
        double* qi_tendency,
        double* ni_tendency, 
        double* dep_tend,
        double* sub_tend,
        // double* diag_1,
        // double* diag_2,
        double* qv_tendency
    ){

    // The diffusion section includes two main content:
    // - deposition if sat_ratio > 0.0 (over saturated);
    // - sublimation if sat_ratio < 0.0 (under saturated)

    double qi_tendency_dep = 0.0;
    double qi_tendency_sub = 0.0;
    double qi_tendency_diff = 0.0;
    double qi_dep = 0.0;
    double ni_dep = 0.0;
    
    // calculate the sat_ratio with look up table method 
    // const double pv_star = lookup(LT, T);
    // const double qv_star = qv_star_c(p0, qt, pv_star);
    // const double satratio = qt/qv_star - 1.0;
    const double satratio = microphysics_saturation_ratio_ice(T, p0, qt);
    
    // specific setting for ice
    double c_i = pi;

    if(qi > SB_EPS && ni > SB_EPS){

        // calculate g_therm factor during vapor diffusion
        double g_therm_ice = microphysics_g_sb_ice(LT, lam_fp, L_fp, T, DVAPOR, KT);
        double F_v_ice = 0.78 + 0.308*NSC_3*sqrt(Dm_i*velocity_i/KIN_VISC_AIR);

        if(satratio >= 0.0){
            qi_tendency_dep = 4.0*pi/c_i * g_therm_ice * Dm_i * F_v_ice * satratio;
            *dep_tend = qi_tendency_dep;
            *sub_tend = 0.0;
        }
        else{
            qi_tendency_sub = 4.0*pi/c_i * g_therm_ice * Dm_i * F_v_ice * satratio;
            *sub_tend = qi_tendency_sub;
            *dep_tend = 0.0;
        }
        qi_tendency_diff = qi_tendency_dep + qi_tendency_sub;

        qi_dep = fmax(qi_tendency_diff*dt, -qi);
        ni_dep = fmax(ni + fmin(qi_tendency_diff, 0.0)/mass_i/2.0, 0.0);

        *qi_tendency = qi_dep/dt;
        *ni_tendency = ni_dep/dt;
        // TODO: can't be += all there will be a big wrong effect
        *qv_tendency += -qi_dep/dt;

        // *diag_1 = F_v_ice;
        // *diag_2 = g_therm_ice;
    }
    else{
        *qi_tendency = 0.0;
        *ni_tendency = 0.0;
        *dep_tend = 0.0;
        *sub_tend = 0.0;
        *qv_tendency += 0.0;
    }
    return;
}

void sb_ice_self_collection(
    // INPUT variables 
    double T,          // ambient atmosphere temperature
    double qi,         // ice specific content
    double ni,         // ice number density
    double Dm_i,       // ice diameter
    double velocity_i, // ice falling velocity
    double dt,
    // OUTPUT variable indexs
    double* qs_tendency,
    double* ns_tendency,
    double* qi_tendency,
    double* ni_tendency
    ){
    // ice selfcollection and aggration to snow: i+i -> s
    // define local variables
    double q_ii_tend, n_ii_tend; // qi and ni tendency owning to self collection
    double q_ii, n_ii; // tmp qi and ni source (q_tend x dt)
    double E_ii; // collection efficience

    if(ni > 0.0 && qi > ICE_CRIT_II && Dm_i > ICE_D_CRIT_II){
        // delta_0_i: 'varaibles name'_'particle species'_'moment'
        double delta_i_0 = sb_collection_delta_b(0.0, SB_ICE_B, SB_ICE_nu, SB_ICE_mu);
        double delta_ii_0 = sb_collection_delta_ab(0.0, SB_ICE_B, SB_ICE_nu, SB_ICE_mu, SB_ICE_B, SB_ICE_nu, SB_ICE_mu);
        double delta_i_1 = sb_collection_delta_b(1.0, SB_ICE_B, SB_ICE_nu, SB_ICE_mu);
        double delta_ii_1 = sb_collection_delta_ab(1.0, SB_ICE_B, SB_ICE_nu, SB_ICE_mu, SB_ICE_B, SB_ICE_nu, SB_ICE_mu);

        double vartheta_i_0 = sb_collection_vartheta_b(0.0, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu);
        double vartheta_ii_0 = sb_collection_vartheta_ab(0.0, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu);
        double vartheta_i_1 = sb_collection_vartheta_b(1.0, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu);
        double vartheta_ii_1 = sb_collection_vartheta_ab(1.0, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu);
        
        double E_ii = cotton_efficiency(T);
        double epsilon_i = 0.2; // m s^-1
        double x_conv_ii = pow((ICE_D_CRIT_II/SB_SNOW_A),(1.0/SB_SNOW_B));

        q_ii_tend = pi/4.0 * E_ii * ni * qi * (delta_i_0*Dm_i*Dm_i + delta_ii_1*Dm_i*Dm_i + delta_i_1*Dm_i*Dm_i) * 
            sqrt(vartheta_i_0*velocity_i*velocity_i - vartheta_ii_1*velocity_i*velocity_i + vartheta_i_1*velocity_i*velocity_i + 2.0*epsilon_i);

        n_ii_tend = pi/4.0 * E_ii * ni * ni * (delta_i_0*Dm_i*Dm_i + delta_ii_0*Dm_i*Dm_i + delta_i_0*Dm_i*Dm_i) * 
            sqrt(vartheta_i_0*velocity_i*velocity_i - vartheta_ii_0*velocity_i*velocity_i + vartheta_i_0*velocity_i*velocity_i + 2.0*epsilon_i);

        q_ii = fmin(q_ii_tend*dt, qi);
        n_ii = fmin(fmin(n_ii_tend*dt, n_ii_tend*dt/x_conv_ii), ni);

        *qi_tendency = -q_ii/dt;
        *qs_tendency = q_ii/dt;
        *ni_tendency = -n_ii/dt;
        *ns_tendency = n_ii/dt/2.0;
    }
    else{
        *qi_tendency = 0.0;
        *qs_tendency = 0.0;
        *ni_tendency = 0.0;
        *ns_tendency = 0.0;
    }
    return;
}

void sb_snow_self_collection(
    // INPUT variables 
    double T,          // ambient atmosphere temperature
    double qs,         // snow specific humidity
    double ns,         // snow number density
    double Dm_s,       // snow diameter
    double velocity_s, // snow falling velocity
    double dt,
    // OUTPUT variable indexs
    double* ns_tendency
){
    // snow selfcollection effect on snow number density: s+s -> s
    // define local variables
    double n_ss_tend; // qi and ni tendency owning to self collection
    double n_ss; // tmp qi and ni source (q_tend x dt)
    double E_ss; // collection efficience
    

    if(qs > SB_EPS){
        // delta_0_i: 'varaibles name'_'particle species'_'moment'
        double delta_s_0 = sb_collection_delta_b(0.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu);
        double delta_ss_0 = sb_collection_delta_ab(0.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu);

        double vartheta_s_0 = sb_collection_vartheta_b(0.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);
        double vartheta_ss_0 = sb_collection_vartheta_ab(0.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);
        
        double E_ss = sticking_efficiencies(T);
        double epsilon_s = 0.2; // m s^-1

        n_ss_tend = pi/4.0 * E_ss * ns * ns * (delta_s_0*Dm_s*Dm_s + delta_ss_0*Dm_s*Dm_s + delta_s_0*Dm_s*Dm_s) * 
            sqrt(vartheta_s_0*velocity_s*velocity_s - vartheta_ss_0*velocity_s*velocity_s + vartheta_s_0*velocity_s*velocity_s + 2.0*epsilon_s);

        n_ss = fmin(n_ss_tend*dt, ns);

        *ns_tendency = -n_ss/dt;
    }
    else{
        *ns_tendency = 0.0;
    }
    return;
}

void sb_snow_ice_collection(
    double T, 
    double qi,
    double ni,
    double Dm_i,
    double velocity_i,
    double qs,
    double ns,
    double Dm_s,
    double velocity_s,
    double dt,
    double* qs_tendency,
    double* qi_tendency,
    double* ni_tendency
){
    // snow collect ice: s+i -> s
    // define local variables
    double q_si_tend, n_si_tend; // qi and ni tendency owning to self collection
    double q_si, n_si; // tmp qi and ni source (∂q/∂t × ∂t)
    double E_si; // collection efficience
    double epsilon_s, epsilon_i;
 
    if (qi > SB_EPS && qs > SB_EPS){
        double delta_s_0 = sb_collection_delta_b(0.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu);
        double delta_i_0 = sb_collection_delta_b(0.0, SB_ICE_B, SB_ICE_nu, SB_ICE_mu);
        double delta_si_0 = sb_collection_delta_ab(0.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu, SB_ICE_B, SB_ICE_nu, SB_ICE_mu);
        double delta_i_1 = sb_collection_delta_b(1.0, SB_ICE_B, SB_ICE_nu, SB_ICE_mu);
        double delta_si_1 = sb_collection_delta_ab(1.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu, SB_ICE_B, SB_ICE_nu, SB_ICE_mu);

        double vartheta_s_0 = sb_collection_vartheta_b(0.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);
        double vartheta_i_0 = sb_collection_vartheta_b(0.0, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu);
        double vartheta_si_0 = sb_collection_vartheta_ab(0.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu);
        double vartheta_i_1 = sb_collection_vartheta_b(1.0, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu);
        double vartheta_si_1 = sb_collection_vartheta_ab(1.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu);

        E_si = 1.0;
        epsilon_s = 0.2;
        epsilon_i = 0.2;

        n_si_tend = pi/4.0 * E_si * ns * ni * (delta_s_0*Dm_s*Dm_s + delta_si_0*Dm_s*Dm_i + delta_i_0*Dm_i*Dm_i) * 
            sqrt(vartheta_s_0*velocity_s*velocity_s - vartheta_si_0*velocity_s*velocity_i + vartheta_i_0*velocity_i*velocity_i + epsilon_s + epsilon_i);
        q_si_tend = pi/4.0 * E_si * ns * qi * (delta_s_0*Dm_s*Dm_s + delta_si_1*Dm_s*Dm_i + delta_i_1*Dm_i*Dm_i) * 
            sqrt(vartheta_s_0*velocity_s*velocity_s - vartheta_si_1*velocity_s*velocity_i + vartheta_i_1*velocity_i*velocity_i + epsilon_s + epsilon_i);

        n_si = fmin(n_si_tend*dt, ni);
        q_si = fmin(q_si_tend*dt, qi);
        
        *qs_tendency = q_si/dt;
        *qi_tendency = -q_si/dt;
        *ni_tendency = -n_si/dt;
    }
    else{
        *qs_tendency = 0.0;
        *qi_tendency = 0.0;
        *ni_tendency = 0.0;
    }

    return;
}

// Snow Riming process
void sb_snow_cloud_riming(
    double nl,
    double ql,
    double Dm_l,
    double velocity_l,
    double ns,
    double qs,
    double Dm_s,
    double velocity_s,
    double* q_sl_tend,
    double* n_sl_tend
){
    // snow riming cloud droplet: s+l -> s
    // define local variables
    double E_sl; // collection efficience
    double epsilon_s, epsilon_l;

    if (ql > LIQUID_CRIT_RIMING && qs > LIQUID_CRIT_RIMING && Dm_l > LIQUID_D_CRIT_RIMING && Dm_s > LIQUID_D_CRIT_RIMING){
        const double delta_s_0 = sb_collection_delta_b(0.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu);
        const double delta_l_0 = sb_collection_delta_b(0.0, SB_LIQUID_B, SB_LIQUID_nu, SB_LIQUID_mu);
        const double delta_sl_0 = sb_collection_delta_ab(0.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu, SB_LIQUID_B, SB_LIQUID_nu, SB_LIQUID_mu);
        const double delta_l_1 = sb_collection_delta_b(1.0, SB_LIQUID_B, SB_LIQUID_nu, SB_LIQUID_mu);
        const double delta_sl_1 = sb_collection_delta_ab(1.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu, SB_LIQUID_B, SB_LIQUID_nu, SB_LIQUID_mu);

        const double vartheta_s_0 = sb_collection_vartheta_b(0.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);
        const double vartheta_l_0 = sb_collection_vartheta_b(0.0, SB_LIQUID_B, SB_LIQUID_beta, SB_LIQUID_nu, SB_LIQUID_mu);
        const double vartheta_sl_0 = sb_collection_vartheta_ab(0.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu, SB_LIQUID_B, SB_LIQUID_beta, SB_LIQUID_nu, SB_LIQUID_mu);
        const double vartheta_l_1 = sb_collection_vartheta_b(1.0, SB_LIQUID_B, SB_LIQUID_beta, SB_LIQUID_nu, SB_LIQUID_mu);
        const double vartheta_sl_1 = sb_collection_vartheta_ab(1.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu, SB_LIQUID_B, SB_LIQUID_beta, SB_LIQUID_nu, SB_LIQUID_mu);
    
        const double const_0 = 1.0/(40.0e-6 - 15.0e-6);
        const double const_1 = const_0 * SNOW_E_MAX;
        E_sl = fmin(0.8, fmax(const_1*(Dm_l - LIQUID_D_CRIT_RIMING), 0.1));

        epsilon_s = 0.2;
        epsilon_l = 0.0;

        *n_sl_tend = pi/4.0 * E_sl * ns * nl * (delta_s_0*Dm_s*Dm_s + delta_sl_0*Dm_s*Dm_l + delta_l_0*Dm_l*Dm_l) * 
            sqrt(vartheta_s_0*velocity_s*velocity_s - vartheta_sl_0*velocity_s*velocity_l + vartheta_l_0*velocity_l*velocity_l + epsilon_s + epsilon_l);
        *q_sl_tend = pi/4.0 * E_sl * ns * ql * (delta_s_0*Dm_s*Dm_s + delta_sl_1*Dm_s*Dm_l + delta_l_1*Dm_l*Dm_l) * 
            sqrt(vartheta_s_0*velocity_s*velocity_s - vartheta_sl_1*velocity_s*velocity_l + vartheta_l_1*velocity_l*velocity_l + epsilon_s + epsilon_l);
    }
    else{
        *n_sl_tend = 0.0;
        *q_sl_tend = 0.0;
    }
    return;
}

void sb_snow_rain_riming(
    double nr,
    double qr,
    double Dm_r,
    double velocity_r,
    double ns,
    double qs,
    double Dm_s,
    double velocity_s,
    double* q_sr_tend, // snow collect rain
    double* q_rs_tend, // rain collect snow
    double* n_sr_tend 
){
    // snow riming raindrops: s+r -> s (and r+s -> g)
    // define local variables
    double E_sr; // collection efficience
    double epsilon_s, epsilon_r;

    if (qr > SB_EPS && qs > RAIN_CRIT_RIMING && Dm_s > RAIN_D_CRIT_RIMING){
        double delta_s_0 = sb_collection_delta_b(0.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu);
        double delta_r_0 = sb_collection_delta_b(0.0, SB_RAIN_B, SB_RAIN_nu, SB_RAIN_mu);
        double delta_sr_0 = sb_collection_delta_ab(0.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu, SB_RAIN_B, SB_RAIN_nu, SB_RAIN_mu);
        double delta_s_1 = sb_collection_delta_b(1.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu);
        double delta_r_1 = sb_collection_delta_b(1.0, SB_RAIN_B, SB_RAIN_nu, SB_RAIN_mu);
        double delta_sr_1 = sb_collection_delta_ab(1.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu, SB_RAIN_B, SB_RAIN_nu, SB_RAIN_mu);
        double delta_rs_1 = sb_collection_delta_ab(1.0, SB_RAIN_B, SB_RAIN_nu, SB_RAIN_mu, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu);

        double vartheta_s_0 = sb_collection_vartheta_b(0.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);
        double vartheta_r_0 = sb_collection_vartheta_b(0.0, SB_RAIN_B, SB_RAIN_beta, SB_RAIN_nu, SB_RAIN_mu);
        double vartheta_sr_0 = sb_collection_vartheta_ab(0.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu, SB_RAIN_B, SB_RAIN_beta, SB_RAIN_nu, SB_RAIN_mu);
        double vartheta_s_1 = sb_collection_vartheta_b(1.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);
        double vartheta_r_1 = sb_collection_vartheta_b(1.0, SB_RAIN_B, SB_RAIN_beta, SB_RAIN_nu, SB_RAIN_mu);
        double vartheta_sr_1 = sb_collection_vartheta_ab(1.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu, SB_RAIN_B, SB_RAIN_beta, SB_RAIN_nu, SB_RAIN_mu);
        double vartheta_rs_1 = sb_collection_vartheta_ab(1.0, SB_RAIN_B, SB_RAIN_beta, SB_RAIN_nu, SB_RAIN_mu, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);

        E_sr = 1.0;
        epsilon_s = 0.2;
        epsilon_r = 0.0;

        *n_sr_tend = pi/4.0 * E_sr * ns * nr * (delta_s_0*Dm_s*Dm_s + delta_sr_0*Dm_s*Dm_r + delta_r_0*Dm_r*Dm_r) * 
            sqrt(vartheta_s_0*velocity_s*velocity_s - vartheta_sr_0*velocity_s*velocity_r + vartheta_r_0*velocity_r*velocity_r + epsilon_s + epsilon_r);
        *q_sr_tend = pi/4.0 * E_sr * ns * qr * (delta_s_0*Dm_s*Dm_s + delta_sr_1*Dm_s*Dm_r + delta_r_1*Dm_r*Dm_r) * 
            sqrt(vartheta_s_0*velocity_s*velocity_s - vartheta_sr_1*velocity_s*velocity_r + vartheta_r_1*velocity_r*velocity_r + epsilon_s + epsilon_r);
        *q_rs_tend = pi/4.0 * E_sr * nr * qs * (delta_r_0*Dm_r*Dm_r + delta_rs_1*Dm_r*Dm_s + delta_s_1*Dm_s*Dm_s) * 
            sqrt(vartheta_r_0*velocity_r*velocity_r - vartheta_rs_1*velocity_r*velocity_s + vartheta_s_1*velocity_s*velocity_s + epsilon_r + epsilon_s);
    }
    else{
        *n_sr_tend = 0.0;
        *q_sr_tend = 0.0;
        *q_rs_tend = 0.0;
    }
    return;
}

void ice_multiplication(
    double T,
    double q_rime,
    double* q_mult,
    double* n_mult
){
    double q_mult_tmp, n_mult_tmp;
    double mult_const1 = 1.0/(T_MULT_OPT - T_MULT_MIN);
    double mult_const2 = 1.0/(T_MULT_OPT - T_MULT_MAX);
    double mult_1 = (T - T_MULT_MIN)*mult_const1;
    double mult_2 = (T - T_MULT_MAX)*mult_const2;
    mult_1 = fmax(0.0, fmin(mult_1, 1.0));
    mult_2 = fmax(0.0, fmin(mult_2, 1.0));
    n_mult_tmp = C_MULT * mult_1 * mult_2 * q_rime;
    q_mult_tmp = n_mult_tmp * ICE_MIN_MASS;
    q_mult_tmp = fmin(q_rime, q_mult_tmp);
    n_mult_tmp = q_mult_tmp / ICE_MIN_MASS;

    *q_mult = q_mult_tmp;
    *n_mult = n_mult_tmp;

    return;
}

void sb_snow_riming(
    // INPUT
    double T,
    double ql,         // --- cloud droplet variables ---- 
    double nl,
    double Dm_l,
    double velocity_l, // --------------------------------
    double qr,         
    double nr,
    double Dm_r,
    double velocity_r,
    double mass_r,
    double qs,
    double ns,
    double Dm_s,
    double velocity_s,
    double dt,
    double qs_tend_dep,
    // OUTPUT
    double* ql_tendency,
    double* nl_tendency,
    double* qi_tendency,
    double* ni_tendency,
    double* qr_tendency,
    double* nr_tendency,
    double* qs_tendency,
    double* ns_tendency
){
    // ---------------------------------------------------------------------------
    // Riming of snow with cloud droplet and rain drops. First the process rates
    // are calculated in
    // - sb_snow_cloud_riming ()
    // - sb_snow_rain_riming ()
    // using those rates and the previously calculated and stored deposition
    // rate the conversion of snow to graupel and rain is done.
    // ---------------------------------------------------------------------------
    
    // define local variables
    double q_sl_tend, n_sl_tend;
    double q_sl, n_sl;
    double q_sr_tend, n_sr_tend;
    double q_rs_tend;
    double q_sr, n_sr, q_rs;
    double q_mult, n_mult;
    double q_rime_all;

    // first do the riming core calculation for snow of cloud droplet and snow
    // same treatment like collection
    sb_snow_cloud_riming(nl, ql, Dm_l, velocity_l, ns, qs, Dm_s, velocity_s, &q_sl_tend, &n_sl_tend);
    sb_snow_rain_riming(nr, qr, Dm_r, velocity_r, ns, qs, Dm_s, velocity_s, &q_sr_tend, &q_rs_tend, &n_sr_tend);

    q_rime_all = q_sl_tend + q_sr_tend;

    // Depositional growth is stronger than riming growth, therefore snow stays snow:
    if (qs_tend_dep > 0.0 && qs_tend_dep > q_rime_all){
        // snow cloud riming
        if (q_sl_tend > 0.0){ 
            q_sl = fmin(ql, q_sl_tend*dt);
            n_sl = fmin(nl, n_sl_tend*dt);

            *qs_tendency += q_sl/dt;
            *ql_tendency += -q_sl/dt;
            *nl_tendency += -n_sl/dt;

            // ice multiplication
            q_mult = 0.0;
            if(T < T_3){
                ice_multiplication(T, q_sl, &q_mult, &n_mult);

                *ni_tendency += n_mult/dt;
                *qi_tendency += q_mult/dt;
                *qs_tendency += -q_mult;
            }
        }
        // snow rain riming
        if (q_sr_tend > 0.0){
            q_sr = fmin(qr, q_sr_tend*dt);
            n_sr = fmin(nr, n_sr_tend*dt);
            *qs_tendency += q_sr/dt;
            *qr_tendency += q_sr/dt;
            *nr_tendency += n_sr/dt;
            
            // ice multiplication
            q_mult = 0.0;
            if(T < T_3){
                ice_multiplication(T, q_sr, &q_mult, &n_mult);

                *ni_tendency += n_mult/dt;
                *qi_tendency += q_mult/dt;
                *qs_tendency += -q_mult;
            }
        }
    }
    // Depositional growth is negative or smaller than riming growth, 
    // therefore snow is allowed to convert to graupel and / or hail:
    else{
        // snow cloud riming
        if (q_sl_tend > 0.0){ 
            q_sl = fmin(ql, q_sl_tend*dt);
            n_sl = fmin(nl, n_sl_tend*dt);

            *qs_tendency += q_sl/dt;
            *ql_tendency += -q_sl/dt;
            *nl_tendency += -n_sl/dt;

            // ice multiplication
            q_mult = 0.0;
            if(T < T_3){
                ice_multiplication(T, q_sl, &q_mult, &n_mult);

                *ni_tendency += n_mult/dt;
                *qi_tendency += q_mult/dt;
                *qs_tendency += -q_mult;
            }

            // conversion of snow to graupel, depends on diameter and temperature
            // if (){
            //
            // }
        }

        // snow rain riming
        if (q_rs_tend > 0.0){
            q_rs = fmin(qs, q_rs_tend*dt);
            q_sr = fmin(qr, q_sr_tend*dt);
            double n = fmin(fmin(nr, n_sr_tend*dt), ns);

            *ns_tendency += -n/dt;
            *nr_tendency += -n/dt;
            *qs_tendency += -q_rs/dt;
            *qr_tendency += -q_sr/dt;
            
            // ice multiplication
            q_mult = 0.0;
            if(T < T_3){
                ice_multiplication(T, q_rs, &q_mult, &n_mult);
            }

            if(T >= T_3){
                *ns_tendency += n/dt;
                *nr_tendency += q_sr/mass_r/dt;
                *qs_tendency += q_rs/dt;
                *qr_tendency += q_sr/dt;
            }
            else{
                // new ice particles from multiplication
                *ni_tendency += n_mult/dt;
                *qi_tendency += q_mult/dt;
                
                // riming to graupel
                if (T < T_MAX_GR_RIME){
                    // 
                }
                // snow + frozen liquid stays snow
                else{
                    *ns_tendency += n/dt;
                    *qs_tendency += (q_sr + q_rs - q_mult)/dt;
                }
            }
        }
    }
    return;
}

void sb_snow_deposition(
        // thermodynamic settings
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
        // INPUT VARIABLES
        double T, 
        double qt,         // total water specific humidity
        double p0,         // air pressure
        double qs,         // snow specific humidity
        double ns,         // snow number density
        double Dm_s,       // mass-weighted diameter of snow
        double mass_s,     // average mass of snow
        double velocity_s, // falling velocity of snow
        double dt, 
        double S_i,
        // OUTPUT VARIABLES INDEX
        double* qs_tendency,
        double* ns_tendency, 
        double* dep_tend,
        double* sub_tend,
        // double* diag_1,
        // double* diag_2,
        double* qv_tendency
    ){

    // The diffusion section includes two main content:
    // - deposition if sat_ratio > 0.0 (over saturated);
    // - sublimation if sat_ratio < 0.0 (under saturated)

    double qs_tendency_dep = 0.0;
    double qs_tendency_sub = 0.0;
    double qs_tendency_diff = 0.0;
    double qs_dep = 0.0;
    double ns_dep = 0.0;
    
    // calculate the sat_ratio 
    // calculate the sat_ratio with look up table method 
    // const double pv_star = lookup(LT, T);
    // const double qv_star = qv_star_c(p0, qt, pv_star);
    // const double satratio = qt/qv_star - 1.0;
    const double satratio = microphysics_saturation_ratio_ice(T, p0, qt);
    
    // specific setting for snow
    double c_i = 2.0;

    if(qs > SB_EPS && ns > SB_EPS){

        // calculate g_therm factor during vapor diffusion
        double g_therm_ice = microphysics_g_sb_ice(LT, lam_fp, L_fp, T, DVAPOR, KT);
        double F_v_snow = 0.78 + 0.308*NSC_3*sqrt(Dm_s*velocity_s/KIN_VISC_AIR);

        if(satratio >= 0.0){
            qs_tendency_dep = 4.0*pi/c_i * g_therm_ice * Dm_s * F_v_snow * satratio;
            *dep_tend += qs_tendency_dep;
        }
        else{
            qs_tendency_sub = 4.0*pi/c_i * g_therm_ice * Dm_s * F_v_snow * satratio;
            *sub_tend += qs_tendency_sub;
        }
        qs_tendency_diff = qs_tendency_dep + qs_tendency_sub;

        qs_dep = fmax(qs_tendency_diff*dt, -qs);
        ns_dep = fmax(ns + fmin(qs_tendency_diff, 0.0)/mass_s/2.0, 0.0);

        *qs_tendency = qs_dep/dt;
        *ns_tendency = ns_dep/dt;
        *qv_tendency += -qs_dep/dt;

        // *diag_1 = g_therm_ice;
        // *diag_2 = F_v_snow;
    }
    else{
        *qs_tendency = 0.0;
        *ns_tendency = 0.0;
        *dep_tend = 0.0;
        *sub_tend = 0.0;
        *qv_tendency += 0.0;
    }
    return;
}

void sb_snow_melting(
        // thermodynamic settings
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        // INPUT VARIABLES
        double p0,         // air pressure
        double T,
        double qt,         // total water specific humidity
        double qv,         // vapor specific humidity
        double qs,         // snow specific humidity
        double ns,         // snow number density
        double mass_s,     // average mass of snow
        double Dm_s,       // mass-weighted diameter of snow
        double velocity_s, // falling velocity of snow
        double dt,
        // OUTPUT VARIABLES INDEX
        double* ns_tendency, double* qs_tendency,
        double* nr_tendency, double* qr_tendency
    ){
    // define local varialbes
    double lam, L, pv_sat;
    double lam_3, L_3, pv_sat_3;
    double qs_melt_tend, ns_melt_tend;
    double qs_melt, ns_melt;

    if (qs > SB_EPS && ns > SB_EPS && T > T_3){

        // lam = lam_fp(T);
        // L = L_fp(T,lam);
        // pv_sat = lookup(LT, T); // saturated vapor pressure at air temperature;
        pv_sat = saturation_vapor_pressure_ice(T);
        pv_sat_3 = saturation_vapor_pressure_ice(T_3);

        // lam_3 = lam_fp(T_3);
        // L_3 = L_fp(T_3,lam);
        // pv_sat_3 = lookup(LT, T_3); // saturated vapor pressure at T_3;

        const double F_v_q = 0.78 + 0.308*NSC_3*sqrt(Dm_s*velocity_s/KIN_VISC_AIR);
        const double F_v_n = 0.78 + 0.308*NSC_3*sqrt(Dm_s*velocity_s/KIN_VISC_AIR);
        // F_v_0 = sb_ventilation_coefficient(0.0, Dm_s, velocity_snow, snow_mass, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);
        // F_v_1 = sb_ventilation_coefficient(1.1, Dm_s, velocity_snow, snow_mass, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);

        const double thermo_melt = (KT/DVAPOR)*(T - T_3) + DVAPOR*L/Rv*(pv_sat/T - pv_sat_3/T_3);

        qs_melt_tend = 2.0*pi/L_MELTING * thermo_melt * ns * Dm_s * F_v_q;
        ns_melt_tend = 2.0*pi/L_MELTING * thermo_melt * ns * Dm_s * F_v_n / mass_s;
        qs_melt = fmin(qs, fmin(qs_melt_tend * dt, 0.0));
        ns_melt = fmin(ns, fmin(ns_melt_tend * dt, 0.0));
        
        *qr_tendency += qs_melt/dt;
        *nr_tendency += ns_melt/dt;
        *qs_tendency = -qs_melt/dt;
        *ns_tendency = -ns_melt/dt;
    }
    else{
        *qs_tendency = 0.0;
        *ns_tendency = 0.0;
        *qr_tendency += 0.0;
        *nr_tendency += 0.0;
    }
}

// void sb_nuc(const struct DimStruct *dims, 
//         // thermodynamic settings
//         struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
//         double* restrict density, // reference air density
//         double* restrict p0, // reference air pressure
//         double dt, // timestep
//         double IN, // given ice nuclei
//         double* restrict w,
//         double* restrict s, // specific entropy
//         double* restrict temperature,  // temperature of air parcel
//         double* restrict S,  // super saturation ratio
//         double* restrict qt, // total water specific humidity
//         double* restrict qv, // vapor
//         double* restrict nl, // cloud liquid number density
//         double* restrict ql, // cloud liquid water specific humidity
//         double* restrict ni, // cloud ice number density
//         double* restrict qi, // cloud ice water specific humidity
//         double* restrict diag_1, double* restrict diag_2, double* restrict diag_3,
//         //OUTPUT ARRAYS: q and n tendency
//         double* restrict nl_tendency, double* restrict ql_tendency,
//         double* restrict ni_tendency, double* restrict qi_tendency
//     ){
//     const ssize_t istride = dims->nlg[1] * dims->nlg[2];
//     const ssize_t jstride = dims->nlg[2];
//     const ssize_t imin = dims->gw;
//     const ssize_t jmin = dims->gw;
//     const ssize_t kmin = dims->gw;
//     const ssize_t imax = dims->nlg[0]-dims->gw;
//     const ssize_t jmax = dims->nlg[1]-dims->gw;
//     const ssize_t kmax = dims->nlg[2]-dims->gw;
//     for(ssize_t i=imin; i<imax; i++){
//         const ssize_t ishift = i * istride;
//         for(ssize_t j=jmin; j<jmax; j++){
//             const ssize_t jshift = j * jstride;
//             for(ssize_t k=kmin; k<kmax; k++){
//                 const ssize_t ijk = ishift + jshift + k;
//                 const double dzi = 1.0/dims->dx[2];
//                 const double C_ccn = 1.0e8;
//                 const double dS = S[ijk + 1] - S[ijk];
//
//                 sb_cloud_activation_hdcp(p0[k], qv[ijk], ql[ijk], 
//                         nl[ijk], w[ijk], dt, S[ijk], 
//                         // &diag_1[ijk],
//                         &ql_tendency[ijk], &nl_tendency[ijk]);
//                 
//                 // ============= Ice Phase Problem ================
//                 sb_ice_nucleation_mayer(LT, IN,
//                         temperature[ijk], qt[ijk], p0[k], 
//                         qv[ijk], ni[ijk], dt,
//                         // &diag_1[ijk], &diag_2[ijk], &diag_3[ijk],
//                         &qi_tendency[ijk], &ni_tendency[ijk]);
//
//                 // sb_ice_nucleation_hdcp(LT, lam_fp, L_fp,
//                 //         temperature[ijk], qt[ijk], p0[k], 
//                 //         qv[ijk], ql[ijk], ni[ijk], dt,
//                 //         // &diag_1[ijk], &diag_2[ijk], &diag_3[ijk],
//                 //         &qi_tendency[ijk], &ni_tendency[ijk]);
//                 
//                 // diag_2[ijk] = nl_tendency[ijk];
//                 // diag_3[ijk] = ni_tendency[ijk];
//             }
//         }
//     }
//     return;
// }


void sb_ice_microphysics_sources(const struct DimStruct *dims, 
        // thermodynamic settings
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        // two-moment specific settings based on SB08
        double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
        // INPUT VARIABLES ARRAY
        double* restrict density, // reference air density
        double* restrict p0, // reference air pressure
        double dt, // timestep
        double ccn, // given cloud condensation nuclei
        double IN, // given ice nuclei
        double* restrict temperature,  // temperature of air parcel
        double* restrict w, // vertical velocity
        // double* restrict S, // satratio of ice
        double* restrict qt, // total water specific humidity
        double* restrict nl, // cloud liquid number density
        double* restrict ql, // cloud liquid water specific humidity
        double* restrict ni, // cloud ice number density
        double* restrict qi, // cloud ice water specific humidity
        double* restrict nr, // rain droplet number density
        double* restrict qr, // rain droplet specific humidity
        double* restrict ns, // snow number density
        double* restrict qs, // snow specific humidity
        //OUTPUT ARRAYS: diagnose variables
        double* restrict Dm, double* restrict mass,
        double* restrict ice_self_col, double* restrict snow_ice_col,
        double* restrict snow_riming, double* restrict snow_dep, double* restrict snow_sub,
        //OUTPUT ARRAYS: q and n tendency
        double* restrict nl_tendency, double* restrict ql_tendency,
        double* restrict ni_tendency, double* restrict qi_tendency,
        double* restrict nr_tendency_micro, double* restrict qr_tendency_micro,
        double* restrict nr_tendency, double* restrict qr_tendency, 
        double* restrict ns_tendency_micro, double* restrict qs_tendency_micro, 
        double* restrict ns_tendency, double* restrict qs_tendency,
        double* restrict precip_rate, double* restrict evap_rate, double* restrict melt_rate
    ){

    //Here we compute the source terms for nr, qr and ns, qs (number and mass of rain and snow)
    //Temporal substepping is used to help ensure boundedness of moments

    // vapor tendency in loop
    double qv_tendency_tmp;
    
    // ---------Warm Process Tendency------------------
    double nr_tendency_tmp, qr_tendency_tmp, ql_tendency_tmp, nl_tendency_tmp;
    // ccn activation
    double nl_tendency_act, ql_tendency_act;

    // cloud droplet condensation (saturation adjustment)
    // double nl_tendency_cnd, ql_tendency_cnd;

    // autoconversion and accretion tendency of rain and cloud droplet
    double nr_tendency_au, qr_tendency_au, qr_tendency_ac;
    double nl_tendency_au, ql_tendency_au, ql_tendency_ac, nl_tendency_ac;
    // selfcollection and breakup tendency
    double nr_tendency_scbk;
    // rain evaporation tendency of rain and vapor 
    double nr_tendency_evp, qr_tendency_evp;
    double qv_tendency_evp;
    // -------------------------------------------------

    // ---------Ice Phase Process Tendency--------------
    double ns_tendency_tmp, qs_tendency_tmp, ni_tendency_tmp, qi_tendency_tmp;
    // ice nucleation
    double ni_tendency_nuc, qi_tendency_nuc;
    // ice deposition and sublimation
    double ni_tendency_dep, qi_tendency_dep;
    // collection tendency 
    // - ice self collection: i+i -> s
    double ni_tendency_ice_selcol, qi_tendency_ice_selcol;
    double ns_tendency_ice_selcol, qs_tendency_ice_selcol;
    // - snow self collection: s+s -> s
    double ns_tendency_snow_selcol;
    // - snow ice collection: s+i -> s
    double qs_tendency_si_col, ni_tendency_si_col, qi_tendency_si_col; 
    
    // - cloud droplet homogeneous freezing
    double ql_tendency_frz, nl_tendency_frz;
    // - rain droplet heterogeneous freezing
    double qr_tendency_frz, nr_tendency_frz;

    // riming tendency 
    // - cloud droplet and rain rimied to snow tendency
    double nl_tendency_snow_rime, ql_tendency_snow_rime;
    double nr_tendency_snow_rime, qr_tendency_snow_rime;
    double ns_tendency_rime, qs_tendency_rime;
    // - ice multiplication tendency
    double ni_tendency_snow_mult, qi_tendency_snow_mult;

    // deposition tendency
    double ns_tendency_dep, qs_tendency_dep;
    double qv_tendency_dep;

    // melting tendency
    double ns_tendency_melt, qs_tendency_melt;
    double nr_tendency_melt, qr_tendency_melt;
    // -------------------------------------------------

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
                const double dzi = 1.0/dims->dx[2];

                ql[ijk] = fmax(ql[ijk],0.0);
                nl[ijk] = fmax(fmin(nl[ijk], ql[ijk]/LIQUID_MIN_MASS),ql[ijk]/LIQUID_MAX_MASS);
                qi[ijk] = fmax(qi[ijk],0.0);
                ni[ijk] = fmax(fmin(ni[ijk], qi[ijk]/ICE_MIN_MASS),qi[ijk]/ICE_MAX_MASS);
                qr[ijk] = fmax(qr[ijk],0.0);
                nr[ijk] = fmax(fmin(nr[ijk], qr[ijk]/RAIN_MIN_MASS),qr[ijk]/RAIN_MAX_MASS);
                qs[ijk] = fmax(qs[ijk],0.0);
                ns[ijk] = fmax(fmin(ns[ijk], qs[ijk]/SB_SNOW_MIN_MASS),qs[ijk]/SB_SNOW_MAX_MASS);

                double qt_tmp = qt[ijk];

                // double ql_tmp = fmax(ql[ijk],0.0);
                double nl_diag = ccn/density[k];
                
                double ql_tmp = fmax(ql[ijk],0.0);
                double nl_tmp = fmax(fmin(nl[ijk], ql_tmp/LIQUID_MIN_MASS),ql_tmp/LIQUID_MAX_MASS);
                double qi_tmp = fmax(qi[ijk],0.0);
                double ni_tmp = fmax(fmin(ni[ijk], qi_tmp/ICE_MIN_MASS),qi_tmp/ICE_MAX_MASS);

                double qv_tmp = qt_tmp - ql_tmp - qi_tmp;

                double qr_tmp = fmax(qr[ijk],0.0);
                double nr_tmp = fmax(fmin(nr[ijk], qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                double qs_tmp = fmax(qs[ijk],0.0);
                double ns_tmp = fmax(fmin(ns[ijk], qs_tmp/SB_SNOW_MIN_MASS),qs_tmp/SB_SNOW_MAX_MASS);

                // define rain sand snow variables
                // and thermodynamic_variables 
                double sat_ratio_liq, sat_ratio_ice, g_therm_rain;
                double liquid_mass, Dm_l, velocity_liquid;
                double ice_mass, Dm_i, velocity_ice;
                double rain_mass, Dm_r, mu, Dp, velocity_rain;
                double snow_mass, Dm_s, velocity_snow;

                // precipitation and evaporation rate tmp variable
                double precip_tmp = 0.0;
                double precip_tend = 0.0;
                double evap_tmp = 0.0;
                double evap_tend = 0.0;
                double melt_tmp= 0.0;

                double time_added = 0.0, dt_, rate;
                ssize_t iter_count = 0;

                double qi_dep_tend, qi_sub_tend, qs_dep_tend, qs_sub_tend;

                do{
                    iter_count       += 1;
                    sat_ratio_liq = microphysics_saturation_ratio_liq(temperature[ijk], p0[k], qt_tmp);
                    sat_ratio_ice = microphysics_saturation_ratio_ice(temperature[ijk], p0[k], qt_tmp);
                    
                    qv_tendency_tmp = 0.0;
                    nr_tendency_tmp = 0.0; 
                    qr_tendency_tmp = 0.0; 
                    ql_tendency_tmp = 0.0; 
                    nl_tendency_tmp = 0.0;
                    nr_tendency_au = 0.0; 
                    qr_tendency_au = 0.0; 
                    qr_tendency_ac = 0.0;
                    nl_tendency_act = 0.0;
                    ql_tendency_act = 0.0;
                    // nl_tendency_cnd = 0.0;
                    // ql_tendency_cnd = 0.0;
                    nl_tendency_au = 0.0; 
                    ql_tendency_au = 0.0; 
                    ql_tendency_ac = 0.0;
                    nl_tendency_ac = 0.0;
                    nr_tendency_scbk = 0.0;
                    nr_tendency_evp = 0.0; 
                    qr_tendency_evp = 0.0;
                    qv_tendency_evp = 0.0;
                    ns_tendency_tmp = 0.0; 
                    qs_tendency_tmp = 0.0; 
                    ni_tendency_tmp = 0.0; 
                    qi_tendency_tmp = 0.0;
                    ni_tendency_nuc = 0.0;
                    qi_tendency_nuc = 0.0;
                    ni_tendency_dep = 0.0;
                    qi_tendency_dep = 0.0;
                    ni_tendency_ice_selcol = 0.0; 
                    qi_tendency_ice_selcol = 0.0;
                    ns_tendency_ice_selcol = 0.0; 
                    qs_tendency_ice_selcol = 0.0;
                    ns_tendency_snow_selcol = 0.0;
                    qs_tendency_si_col = 0.0; 
                    ni_tendency_si_col = 0.0; 
                    qi_tendency_si_col = 0.0; 
                    nl_tendency_snow_rime = 0.0; 
                    ql_tendency_snow_rime = 0.0;
                    nr_tendency_snow_rime = 0.0; 
                    qr_tendency_snow_rime = 0.0;
                    ns_tendency_rime = 0.0; 
                    qs_tendency_rime = 0.0;
                    ni_tendency_snow_mult = 0.0; 
                    qi_tendency_snow_mult = 0.0;
                    ns_tendency_dep = 0.0; 
                    qs_tendency_dep = 0.0;
                    qv_tendency_dep = 0.0;
                    ns_tendency_melt = 0.0; 
                    qs_tendency_melt = 0.0;
                    nr_tendency_melt = 0.0; 
                    qr_tendency_melt = 0.0;
                    ql_tendency_frz = 0.0;
                    nl_tendency_frz = 0.0;
                    qr_tendency_frz = 0.0;
                    nr_tendency_frz = 0.0;

                    qi_dep_tend = 0.0;
                    qi_sub_tend = 0.0;
                    qs_dep_tend = 0.0;
                    qs_sub_tend = 0.0;

                    //obtain some parameters of cloud droplets
                    liquid_mass = microphysics_mean_mass(nl_tmp, ql_tmp, LIQUID_MIN_MASS, LIQUID_MAX_MASS);// average mass of cloud droplets
                    Dm_l =  cbrt(liquid_mass * 6.0/DENSITY_LIQUID/pi);
                    velocity_liquid = 3.75e5 * cbrt(liquid_mass)*cbrt(liquid_mass) *(DENSITY_SB/density[k]);

                    //obtain some parameters of cloud ice particles
                    ice_mass = microphysics_mean_mass(ni_tmp, qi_tmp, ICE_MIN_MASS, ICE_MAX_MASS);// average mass of cloud droplets
                    Dm_i = SB_ICE_A * pow(ice_mass, SB_ICE_B);
                    velocity_ice = SB_ICE_alpha * pow(ice_mass, SB_ICE_beta) * sqrt(DENSITY_SB/density[k]);

                    //obtain some parameters of rain droplets
                    rain_mass = microphysics_mean_mass(nr_tmp, qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS); //average mass of rain droplet
                    Dm_r = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi); // mass weighted diameter of rain droplets
                    mu = rain_mu(density[k], qr_tmp, Dm_r);
                    Dp = sb_Dp(Dm_r, mu);
                    // simplified rain velocity based on equation 28 in SB06
                    velocity_rain = 159.0 * pow(rain_mass, 0.266) * sqrt(DENSITY_SB/density[k]);
                    g_therm_rain = microphysics_g(LT, lam_fp, L_fp, temperature[ijk]);

                    //obtain some parameters of snow
                    snow_mass = microphysics_mean_mass(ns_tmp, qs_tmp, SB_SNOW_MIN_MASS, SB_SNOW_MAX_MASS);
                    Dm_s = SB_SNOW_A * pow(snow_mass, SB_SNOW_B);
                    velocity_snow = SB_SNOW_alpha * pow(snow_mass, SB_SNOW_beta) * sqrt(DENSITY_SB/density[k]);
                    
                    // ==================== Main Content of Calculation ======================================
                    //find the maximum substep time
                    dt_ = dt - time_added;
                    
                    // const double dS = S[ijk + 1] - S[ijk];
                    // sb_ccn(ccn, S[ijk], dS, dzi, w[ijk], 
                    //         &ql_tendency_act, &nl_tendency_act);
                    
                    //compute the source terms of warm phase process: rain
                    sb_autoconversion_rain_tmp(droplet_nu, density[k], nl_tmp, ql_tmp, qr_tmp, 
                            &nr_tendency_au, &qr_tendency_au, &nl_tendency_au, &ql_tendency_au);
                    sb_accretion_rain_tmp(density[k], ql_tmp, qr_tmp, liquid_mass,
                            &qr_tendency_ac, &ql_tendency_ac, &nl_tendency_ac);
                    sb_selfcollection_breakup_rain(density[k], nr_tmp, qr_tmp, mu, rain_mass, Dm_r, 
                            &nr_tendency_scbk);
                    sb_evaporation_rain_tmp(g_therm_rain, sat_ratio_liq, nr_tmp, qr_tmp, 
                            mu, rain_mass, Dp, Dm_r, 
                            &nr_tendency_evp, &qr_tendency_evp, &qv_tendency_evp);

                    // compute the source of ice deposition
                    sb_ice_deposition(LT, lam_fp, L_fp, 
                            temperature[ijk], qt_tmp, p0[k], qi_tmp, ni_tmp, 
                            Dm_i, ice_mass, velocity_ice, dt_, sat_ratio_ice,
                            &qi_tendency_dep, &ni_tendency_dep,
                            &qi_dep_tend, &qi_sub_tend,
                            &qv_tendency_dep);

                    sb_freezing(droplet_nu, density[k], temperature[ijk], liquid_mass, 
                            rain_mass, ql_tmp, nl_tmp, qr_tmp, nr_tmp,
                            &ql_tendency_frz, &nl_tendency_frz,
                            &qr_tendency_frz, &nr_tendency_frz);

                    // compute the source terms of ice phase process: snow 

                    sb_snow_deposition(LT, lam_fp, L_fp, 
                            temperature[ijk], qt_tmp, p0[k], qs_tmp, ns_tmp, 
                            Dm_s, snow_mass, velocity_snow, dt_, sat_ratio_ice,
                            &qs_tendency_dep, &ns_tendency_dep, 
                            &qs_dep_tend, &qs_sub_tend,
                            // &Dm[ijk], &mass[ijk],
                            &qv_tendency_dep);

                    // ice phase collection processes
                    sb_ice_self_collection(temperature[ijk], qi_tmp, ni_tmp, Dm_i, velocity_ice, dt_,
                            &qs_tendency_ice_selcol, &ns_tendency_ice_selcol,
                            &qi_tendency_ice_selcol, &ni_tendency_ice_selcol);
                    sb_snow_self_collection(temperature[ijk], qs_tmp, ns_tmp, Dm_s, velocity_snow, dt_,
                            &ns_tendency_snow_selcol);
                    sb_snow_ice_collection(temperature[ijk], qi_tmp, ni_tmp, Dm_i, velocity_ice, 
                            qs_tmp, ns_tmp, Dm_s, velocity_snow, dt_,
                            &qs_tendency_si_col, &qi_tendency_si_col, &ni_tendency_si_col);
                    // ice phase riming processes
                    sb_snow_riming(temperature[ijk], ql_tmp, nl_tmp, Dm_l, velocity_liquid, 
                            qr_tmp, nr_tmp, Dm_r, velocity_rain, rain_mass, 
                            qs_tmp, ns_tmp, Dm_s, velocity_snow, dt_, qs_tendency_dep,
                            &ql_tendency_snow_rime, &nl_tendency_snow_rime, 
                            &qi_tendency_snow_mult, &ni_tendency_snow_mult,
                            &qr_tendency_snow_rime, &nr_tendency_snow_rime,
                            &qs_tendency_rime, &ns_tendency_rime);
                    sb_snow_melting(LT, lam_fp, L_fp, p0[k], temperature[ijk], 
                            qt_tmp, qv_tmp, qs_tmp, ns_tmp, snow_mass, Dm_s, velocity_snow, dt_,
                            &ns_tendency_melt, &qs_tendency_melt,
                            &nr_tendency_melt, &qr_tendency_melt);

                    //check the source term magnitudes
                    // vapor tendency sum
                    qv_tendency_tmp = qv_tendency_evp + qv_tendency_dep;
                    // rain tendency sum
                    nr_tendency_tmp = nr_tendency_au + nr_tendency_scbk + nr_tendency_evp + 
                                      nr_tendency_snow_rime + nr_tendency_melt - nr_tendency_frz;
                    qr_tendency_tmp = qr_tendency_au + qr_tendency_ac + qr_tendency_evp + 
                                      qr_tendency_snow_rime + qr_tendency_melt - qr_tendency_frz;
                    
                    // snow tendency sum
                    ns_tendency_tmp = ns_tendency_ice_selcol + ns_tendency_snow_selcol + ns_tendency_rime + 
                                      ns_tendency_dep + ns_tendency_melt;
                    qs_tendency_tmp = qs_tendency_ice_selcol + qs_tendency_si_col + qs_tendency_rime + 
                                      qs_tendency_dep + qs_tendency_melt;
                    
                    // cloud droplet tendency sum
                    ql_tendency_tmp = ql_tendency_au + ql_tendency_ac + ql_tendency_snow_rime - ql_tendency_frz;
                    nl_tendency_tmp = nl_tendency_au + nl_tendency_ac + nl_tendency_snow_rime - nl_tendency_frz;

                    // ice particle tendency sum
                    qi_tendency_tmp = qi_tendency_dep + qi_tendency_ice_selcol + qi_tendency_si_col + 
                                      qi_tendency_snow_mult + ql_tendency_frz + qr_tendency_frz;
                    ni_tendency_tmp = ni_tendency_dep + ni_tendency_ice_selcol + ni_tendency_si_col + 
                                      ni_tendency_snow_mult + nl_tendency_frz + nr_tendency_frz;

                    //Factor of 1.05 is ad-hoc
                    rate = 1.05 * ql_tendency_tmp * dt_ /(- fmax(ql_tmp,SB_EPS));
                    rate = fmax(1.05 * qi_tendency_tmp * dt_ /(-fmax(qi_tmp,SB_EPS)), rate);
                    rate = fmax(1.05 * nr_tendency_tmp * dt_ /(-fmax(nr_tmp,SB_EPS)), rate);
                    rate = fmax(1.05 * qr_tendency_tmp * dt_ /(-fmax(qr_tmp,SB_EPS)), rate);
                    rate = fmax(1.05 * ns_tendency_tmp * dt_ /(-fmax(ns_tmp,SB_EPS)), rate);
                    rate = fmax(1.05 * qs_tendency_tmp * dt_ /(-fmax(qs_tmp,SB_EPS)), rate);
                    if(rate > 1.0 && iter_count < MAX_ITER){
                        //Limit the timestep, but don't allow it to become vanishingly small
                        //Don't adjust if we have reached the maximum iteration number
                        dt_ = fmax(dt_/rate, 1.0e-3);
                    }
                    
                    precip_tend = qr_tendency_au + qr_tendency_ac + qs_tendency_rime +
                                 qs_tendency_ice_selcol + qs_tendency_si_col + qi_dep_tend + qs_dep_tend; // keep POSITIVE when precipation formed
                    evap_tend = -(qi_sub_tend + qs_sub_tend + qv_tendency_evp); // keep POSITIVE when evap/sub formed
                    
                    precip_tmp += precip_tend * dt_;
                    evap_tmp   += evap_tend * dt_;
                    melt_tmp   += qr_tendency_melt * dt_;

                    //Integrate forward in time
                    ql_tmp += ql_tendency_tmp * dt_;
                    nl_tmp += nl_tendency_tmp * dt_;
                    
                    qi_tmp += qi_tendency_tmp * dt_;
                    ni_tmp += ni_tendency_tmp * dt_;

                    qr_tmp += qr_tendency_tmp * dt_;
                    nr_tmp += nr_tendency_tmp * dt_;

                    qs_tmp += qs_tendency_tmp * dt_;
                    ns_tmp += ns_tendency_tmp * dt_;

                    qv_tmp += qv_tendency_tmp * dt_;

                    qr_tmp = fmax(qr_tmp,0.0);
                    nr_tmp = fmax(fmin(nr_tmp, qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                    qs_tmp = fmax(qs_tmp,0.0);
                    ns_tmp = fmax(fmin(ns_tmp, qs_tmp/SB_SNOW_MIN_MASS),qs_tmp/SB_SNOW_MAX_MASS);
                    ql_tmp = fmax(ql_tmp,0.0);
                    nl_tmp = fmax(fmin(nl_tmp, ql_tmp/LIQUID_MIN_MASS),ql_tmp/LIQUID_MAX_MASS);
                    qi_tmp = fmax(qi_tmp,0.0);
                    ni_tmp = fmax(fmin(ni_tmp, qi_tmp/ICE_MIN_MASS),qi_tmp/ICE_MAX_MASS);
                    qt_tmp = ql_tmp + qv_tmp + qi_tmp;

                    time_added += dt_ ;

                }while(time_added < dt);

                nr_tendency_micro[ijk] = (nr_tmp - nr[ijk])/dt;
                qr_tendency_micro[ijk] = (qr_tmp - qr[ijk])/dt;
                ns_tendency_micro[ijk] = (ns_tmp - ns[ijk])/dt;
                qs_tendency_micro[ijk] = (qs_tmp - qs[ijk])/dt;

                nl_tendency[ijk] += (nl_tmp - nl[ijk])/dt;
                ql_tendency[ijk] += (ql_tmp - ql[ijk])/dt;
                ni_tendency[ijk] += (ni_tmp - ni[ijk])/dt;
                qi_tendency[ijk] += (qi_tmp - qi[ijk])/dt;
                nr_tendency[ijk] += nr_tendency_micro[ijk];
                qr_tendency[ijk] += qr_tendency_micro[ijk];
                ns_tendency[ijk] += ns_tendency_micro[ijk];
                qs_tendency[ijk] += qs_tendency_micro[ijk];

                // diagnose snow varialbes
                ice_self_col[ijk] = qi_tendency_dep;
                precip_rate[ijk] = precip_tmp/dt;
                evap_rate[ijk]   = evap_tmp/dt;
                melt_rate[ijk]   = melt_tmp/dt;
            }
        }
    }
    return;
}

void sb_sedimentation_velocity_snow(const struct DimStruct *dims, 
        // INPUT VARIABLES ARRAY
        double* restrict ns,      // snow number density
        double* restrict qs,      // snow specific humidity
        // OUTPUT VARIABLES 
        double* restrict ns_velocity, 
        double* restrict qs_velocity
    ){

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

                double qs_tmp = fmax(qs[ijk],0.0);
                double ns_tmp = fmax(fmin(ns[ijk], qs_tmp/SB_SNOW_MIN_MASS),qs_tmp/SB_SNOW_MAX_MASS);
                double snow_mass = microphysics_mean_mass(ns_tmp, qs_tmp, SB_SNOW_MIN_MASS, SB_SNOW_MAX_MASS);
                
                double ns_vel_tmp = SB_SNOW_alpha * tgamma(6.0 + 3.0*SB_SNOW_beta)/tgamma(6.0) * pow(tgamma(6.0)/tgamma(9.0), SB_SNOW_beta) * pow(snow_mass, SB_SNOW_beta);
                double qs_vel_tmp = SB_SNOW_alpha * tgamma(9.0 + 3.0*SB_SNOW_beta)/tgamma(9.0) * pow(tgamma(6.0)/tgamma(9.0), SB_SNOW_beta) * pow(snow_mass, SB_SNOW_beta);
                ns_velocity[ijk] = -fmin(fmax(ns_vel_tmp, 0.0),10.0);
                qs_velocity[ijk] = -fmin(fmax(qs_vel_tmp, 0.0),10.0);
            }
        }
    }
     for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax-1 ; k++){
                const ssize_t ijk = ishift + jshift + k;
                ns_velocity[ijk] = interp_2(ns_velocity[ijk], ns_velocity[ijk+1]) ;
                qs_velocity[ijk] = interp_2(qs_velocity[ijk], qs_velocity[ijk+1]) ;
            }
        }
    }
    return;
}

void sb_2m_qt_source_formation(const struct DimStruct *dims, 
        double* restrict qt_tendency,
        double* restrict precip_rate, 
        double* restrict evap_rate){

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
                qt_tendency[ijk] += evap_rate[ijk] - precip_rate[ijk];
            }
        }
    }
    return;
}

void sb_2m_qt_source_debug(const struct DimStruct *dims, 
        double* restrict qt_tendency,
        double* restrict qr_tend,
        double* restrict qs_tend){

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
                qt_tendency[ijk] -= qr_tend[ijk] + qs_tend[ijk];
            }
        }
    }
    return;
}

// ========== Entropy Source Of SB Microphysics Scheme =============

void sb_entropy_source_precip(
        double L,
        double p0, 
        double T,
        double qt, 
        double qv,
        double precip_rate, 
        double* entropy_tendency
    ){

    double pd = pd_c(p0, qt, qv); // dry air pressure
    double pv = pv_c(p0, qt, qv); // vapor pressure
    double sd = sd_c(pd, T); // s_d(T): specific entropy of dry air
    double sv = sv_c(pv, T); // s_v(T): specific entropy of vapor 
    double sc = sc_c(L, T); // sc_c = -L/T

    *entropy_tendency = (sd - sv - sc) * precip_rate;

    return;
};

void sb_entropy_source_evap(
        // INPUT
        double pv_star_T,
        double L,
        double p0, 
        double T, 
        double Twet, 
        double qt, 
        double qv,
        double evap_rate, 
        // OUTPUT
        double* sd_tend,
        double* se_tend
    ){

    const double pd = pd_c(p0, qt, qv); // dry air pressure
    const double pv = pv_c(p0, qt, qv); // vapor pressure
    const double sd = sd_c(pd, T); // s_d(T), specific entropy of dry air
    const double sv = sv_c(pv, Twet); // s_v^*(Twet): specific entropy of vapor under wetbuble temperature
    const double sc = sc_c(L, Twet); // sc_c = -L/T
    // equation 50 in Pressel15, evap/sub term
    const double s_E = (sv + sc - sd);
    // equation 51 in Pressel15, vapor diffusivity term
    const double s_D = -Rv*log(pv/pv_star_T) + cpv*log(T/Twet); 
    
    *se_tend = s_E*evap_rate;
    *sd_tend = s_D*evap_rate;

    return;
};

void sb_entropy_source_melt(
        double T,
        double melt_rate, 
        double *entropy_tendency
    ){
    if(abs(melt_rate) > SB_EPS){
        *entropy_tendency = melt_rate * lhf / T;
    }
    else{
        *entropy_tendency = 0.0;
    }
    return;
};

void sb_entropy_source_heating_func(
        double T, 
        double Twet, 
        double Twet_up, // Twet[ijk + 1], should be Twet in the upper level 
        double q,
        double w_q, 
        double w,
        double c_tpye, // specific heat of falling precipitation species
        double dzi,
        double* entropy_tendency
    ){
    
    if(q > SB_EPS){
        *entropy_tendency += q * (fabs(w_q)-w) * c_tpye * (Twet_up - Twet)*dzi / T;
    }
    else{
        *entropy_tendency += 0.0;
    }
    return;
};

void sb_entropy_source_drag_func(
        double T, 
        double q,
        double w_q, 
        double* entropy_tendency
    ){
    if(q > SB_EPS){
        *entropy_tendency += g*q*fabs(w_q) / T;
    }
    else{
        *entropy_tendency += 0.0;
    }
    return;
};

void sb_2m_entropy_source(
        const struct DimStruct *dims, 
        // Thermo Variables
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        double* restrict p0, 
        double* restrict temperature,
        double* restrict Twet, 
        double* restrict w, 
        double* restrict qt, 
        double* restrict qv,
        double* restrict qr,
        double* restrict w_qr, 
        double* restrict qs,
        double* restrict w_qs, 
        double* restrict precip_rate, 
        double* restrict evap_rate, 
        double* restrict melt_rate, 
        // DIAGNOSE
        double* restrict sp, // entropy source of precipitation
        double* restrict se, // entropy source of evaporation
        double* restrict sd, // entropy source of vapor diffusion
        double* restrict sm, // entropy source of melting
        double* restrict sq, // entropy source of heating
        double* restrict sw, // entropy source of draging
        // OUTPUT
        double* restrict entropy_tendency
    ){
    
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;
    const double dzi = 1.0/dims->dx[2];

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                
                double S_P, S_E, S_D, S_M, S_Q, S_W;
                const double lam = lam_fp(temperature[ijk]);
                const double L = L_fp(temperature[ijk],lam);
                const double pv_star_T = lookup(LT, temperature[ijk]); // saturation vapor pressure

                // precipitation entropy source
                if (qr[ijk] > SB_EPS && qs[ijk] > 1.0e-10){
                    sb_entropy_source_precip(L, p0[k], temperature[ijk], qt[ijk], qv[ijk], precip_rate[ijk], &S_P);
                    // evaporation entropy source
                    sb_entropy_source_evap(pv_star_T, L, p0[k], temperature[ijk], Twet[ijk], qt[ijk], qv[ijk], evap_rate[ijk], &S_D, &S_E);
                    // melting entropy source 
                    sb_entropy_source_melt(temperature[ijk], melt_rate[ijk], &S_M);
                }
                else{
                    S_P = 0.0;
                    S_E = 0.0;
                    S_D = 0.0;
                    S_M = 0.0;
                }
                // heating entropy source for each specie: rain/snow
                S_Q = 0.0;
                sb_entropy_source_heating_func(temperature[ijk], Twet[ijk], Twet[ijk+1], qr[ijk], w_qr[ijk], w[ijk], dzi, cl, &S_Q);
                sb_entropy_source_heating_func(temperature[ijk], Twet[ijk], Twet[ijk+1], qs[ijk], w_qs[ijk], w[ijk], dzi, ci, &S_Q);

                // draging entropy source
                S_W = 0.0;
                sb_entropy_source_drag_func(temperature[ijk], qr[ijk], w_qr[ijk], &S_W);
                sb_entropy_source_drag_func(temperature[ijk], qs[ijk], w_qs[ijk], &S_W);
                
                entropy_tendency[ijk] += S_P + S_E + S_D + S_M + S_Q + S_W;

                // diagnosed each entropy source
                sp[ijk] = S_P;
                se[ijk] = S_E;
                sd[ijk] = S_D;
                sm[ijk] = S_M;
                sq[ijk] = S_Q;
                sw[ijk] = S_W;
            }
        }
    }
    return;
}

void sb_entropy_source_precipitation(
        const struct DimStruct *dims, 
        // Thermo Variables
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        double* restrict p0, 
        double* restrict temperature,
        double* restrict qt, 
        double* restrict qv,
        double* restrict precip_rate, 
        double* restrict entropy_tendency
    ){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    // entropy tendencies from formation of rain/snow/graupel
    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                double lam = lam_fp(temperature[ijk]);
                double L = L_fp(temperature[ijk],lam);
                // define local variables 
                double pd = pd_c(p0[k], qt[ijk], qv[ijk]); // dry air pressure
                double pv = pv_c(p0[k], qt[ijk], qv[ijk]); // vapor pressure
                double sd = sd_c(pd, temperature[ijk]); // s_d(T): specific entropy of dry air
                double sv = sv_c(pv, temperature[ijk]); // s_v(T): specific entropy of vapor 
                double sc = sc_c(L, temperature[ijk]); // sc_c = -L/T

                entropy_tendency[ijk] += (sd - sv - sc) * precip_rate[ijk];

            }
        }
    }
    return;
};

void sb_entropy_source_evap_sub_diff(
        const struct DimStruct *dims, 
        // Thermo variables 
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        // INPUT
        double* restrict p0, 
        double* restrict temperature,
        double* restrict Twet, 
        double* restrict qt, 
        double* restrict qv,
        double* restrict evap_rate, 
        // OUTPUT
        double* restrict entropy_tendency
    ){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    // entropy tendencies from evaporation of rain, sublimation of snow/graupel
    // and vapor diffusivity 
    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                double lam = lam_fp(temperature[ijk]);
                double L = L_fp(temperature[ijk],lam);
                const double pv_star_T = lookup(LT, temperature[ijk]); 
                
                // local variables defination
                double pd = pd_c(p0[ijk], qt[ijk], qv[ijk]); // dry air pressure
                double pv = pv_c(p0[ijk], qt[ijk], qv[ijk]); // vapor pressure
                double sd = sd_c(pd, temperature[ijk]); // s_d(T), specific entropy of dry air
                double sv = sv_c(pv, Twet[ijk]); // s_v^*(Twet): specific entropy of vapor under wetbuble temperature
                double sc = sc_c(L, Twet[ijk]); // sc_c = -L/T
                double S_E = (sv + sc - sd);  // equation 50 in Pressel15
                double S_D = -Rv*log(pv/pv_star_T) + cpv*log(temperature[ijk]/Twet[ijk]); // equation 51 in Pressel15
                
                entropy_tendency[ijk] += (S_E + S_D)*evap_rate[ijk];
            }
        }
    }
    return;
};

// void sb_entropy_source_melt(
//         const struct DimStruct *dims, 
//         double* restrict temperature,
//         double* restrict melt_rate, 
//         double* restrict entropy_tendency
//     ){
//
//     const ssize_t istride = dims->nlg[1] * dims->nlg[2];
//     const ssize_t jstride = dims->nlg[2];
//     const ssize_t imin = dims->gw;
//     const ssize_t jmin = dims->gw;
//     const ssize_t kmin = dims->gw;
//     const ssize_t imax = dims->nlg[0]-dims->gw;
//     const ssize_t jmax = dims->nlg[1]-dims->gw;
//     const ssize_t kmax = dims->nlg[2]-dims->gw;
//
//     //entropy tendencies from snow melt
//     //we use fact that melt_rate is negative when snow becomes rain
//     for(ssize_t i=imin; i<imax; i++){
//         const ssize_t ishift = i * istride;
//         for(ssize_t j=jmin; j<jmax; j++){
//             const ssize_t jshift = j * jstride;
//             for(ssize_t k=kmin; k<kmax; k++){
//                 const ssize_t ijk = ishift + jshift + k;
//                 entropy_tendency[ijk] += melt_rate[ijk] * lhf / temperature[ijk];
//             }
//         }
//     }
//     return;
// };


void sb_entropy_source_heating_rain(const struct DimStruct *dims, 
        double* restrict temperature, 
        double* restrict Twet, 
        double* restrict qr,
        double* restrict w_qr, 
        double* restrict w, 
        double* restrict entropy_tendency
    ){

    //derivative of Twet is upwinded
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;
    const double dzi = 1.0/dims->dx[2];


    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                entropy_tendency[ijk]+= qr[ijk]*(fabs(w_qr[ijk]) - w[ijk]) * cl * 
                        (Twet[ijk+1] - Twet[ijk])*dzi / temperature[ijk];
            }
        }
    }
    return;
};

void sb_entropy_source_heating_snow(
        const struct DimStruct *dims, 
        double* restrict temperature, 
        double* restrict Twet, 
        double* restrict qs,
        double* restrict w_qs, 
        double* restrict w, 
        double* restrict entropy_tendency
    ){

    //derivative of Twet is upwinded

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;
    const double dzi = 1.0/dims->dx[2];


    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                entropy_tendency[ijk]+= qs[ijk]*(fabs(w_qs[ijk]) - w[ijk]) * ci * 
                        (Twet[ijk+1] - Twet[ijk])*dzi / temperature[ijk];
            }
        }
    }
    return;
};

void sb_entropy_source_heating_graupel(
        const struct DimStruct *dims, 
        double* restrict temperature, 
        double* restrict Twet, 
        double* restrict qg,
        double* restrict w_qg, 
        double* restrict w, 
        double* restrict entropy_tendency
    ){

    //derivative of Twet is upwinded

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;
    const double dzi = 1.0/dims->dx[2];


    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                entropy_tendency[ijk]+= qg[ijk]*(fabs(w_qg[ijk]) - w[ijk]) * ci * 
                                        (Twet[ijk+1] - Twet[ijk])*dzi / temperature[ijk];
            }
        }
    }
    return;
};

// TODO: there is another drag in sb_liquid section, which need to be rearranged
void sb_entropy_source_drag_tmp(
        const struct DimStruct *dims, 
        double* restrict temperature, 
        double* restrict q,
        double* restrict w_q, 
        double* restrict entropy_tendency
    ){

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
                entropy_tendency[ijk] += g*q[ijk]*fabs(w_q[ijk]) / temperature[ijk];
            }
        }
    }
    return;
};


/// ================ To Facilitate Output ================
void sb_2m_prameters_wrapper(
        const struct DimStruct *dims, 
        double* restrict ql,
        double* restrict nl,
        double* restrict qi,
        double* restrict ni,
        double* restrict qr,
        double* restrict nr,
        double* restrict qs,
        double* restrict ns,
        double* restrict density,
        double* restrict Dm_l,
        double* restrict Dm_i,
        double* restrict Dm_r,
        double* restrict Dm_s,
        double* restrict mass_l,
        double* restrict mass_i,
        double* restrict mass_r,
        double* restrict mass_s,
        double* restrict velocity_l,
        double* restrict velocity_i,
        double* restrict velocity_r,
        double* restrict velocity_s
    ){
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
            
                double ql_tmp = fmax(ql[ijk],0.0);
                double nl_tmp = fmax(fmin(nl[ijk], ql_tmp/LIQUID_MIN_MASS),ql_tmp/LIQUID_MAX_MASS);
                double qi_tmp = fmax(qi[ijk],0.0);
                double ni_tmp = fmax(fmin(ni[ijk], qi_tmp/ICE_MIN_MASS),qi_tmp/ICE_MAX_MASS);
                double qr_tmp = fmax(qr[ijk],0.0);
                double nr_tmp = fmax(fmin(nr[ijk], qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                double qs_tmp = fmax(qs[ijk],0.0);
                double ns_tmp = fmax(fmin(ns[ijk], qs_tmp/SB_SNOW_MIN_MASS),qs_tmp/SB_SNOW_MAX_MASS);
                
                double liquid_mass = microphysics_mean_mass(nl_tmp, ql_tmp, LIQUID_MIN_MASS, LIQUID_MAX_MASS);// average mass of cloud droplets
                Dm_l[ijk] =  cbrt(liquid_mass * 6.0/DENSITY_LIQUID/pi);
                velocity_l[ijk] = 3.75e5 * cbrt(liquid_mass)*cbrt(liquid_mass) *(DENSITY_SB/density[k]);
                mass_l[ijk] = liquid_mass;

                double ice_mass = microphysics_mean_mass(ni_tmp, qi_tmp, ICE_MIN_MASS, ICE_MAX_MASS);// average mass of cloud droplets
                Dm_i[ijk] = SB_ICE_A * pow(ice_mass, SB_ICE_B);
                velocity_i[ijk] = SB_ICE_alpha * pow(ice_mass, SB_ICE_beta) * sqrt(DENSITY_SB/density[k]);
                mass_i[ijk] = ice_mass;

                double rain_mass = microphysics_mean_mass(nr_tmp, qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS); //average mass of rain droplet
                Dm_r[ijk] = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi); // mass weighted diameter of rain droplets
                velocity_r[ijk] = 159.0 * pow(rain_mass, 0.266) * sqrt(DENSITY_SB/density[k]);
                mass_r[ijk] = rain_mass;

                //obtain some parameters of snow
                double snow_mass = microphysics_mean_mass(ns_tmp, qs_tmp, SB_SNOW_MIN_MASS, SB_SNOW_MAX_MASS);
                Dm_s[ijk] = SB_SNOW_A * pow(snow_mass, SB_SNOW_B);
                velocity_s[ijk] = SB_SNOW_alpha * pow(snow_mass, SB_SNOW_beta) * sqrt(DENSITY_SB/density[k]);
                mass_s[ijk] = snow_mass;
            }
        }
    }
    return;
}

void  sb_ice_deposition_wrapper(
        const struct DimStruct *dims, 
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
        // INPUT VARIABLES
        double* restrict temperature, 
        double* restrict qt,         // total water specific humidity
        double* restrict p0,         // air pressure
        double* restrict density,
        double* restrict qi,         // ice specific humidity
        double* restrict ni,         // ice number density
        double dt, 
        // OUTPUT VARIABLES INDEX
        double* restrict qi_tendency,
        double* restrict ni_tendency, 
        double* restrict ice_dep_tend,
        double* restrict ice_sub_tend,
        double* restrict qv_tendency
    ){

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
                
                double qi_tmp = fmax(qi[ijk],0.0);
                double ni_tmp = fmax(fmin(ni[ijk], qi_tmp/ICE_MIN_MASS),qi_tmp/ICE_MAX_MASS);
                double ice_mass = microphysics_mean_mass(ni_tmp, qi_tmp, ICE_MIN_MASS, ICE_MAX_MASS);
                double Dm_i = SB_ICE_A * pow(ice_mass, SB_ICE_B);
                double velocity_ice = SB_ICE_alpha * pow(ice_mass, SB_ICE_beta) * sqrt(DENSITY_SB/density[k]);
                double sat_ratio_ice = microphysics_saturation_ratio_ice(temperature[ijk], p0[k], qt[ijk]);

                sb_ice_deposition(LT, lam_fp, L_fp, 
                    temperature[ijk], qt[ijk], p0[k], qi_tmp, ni_tmp, 
                    Dm_i, ice_mass, velocity_ice, dt, sat_ratio_ice,
                    &qi_tendency[ijk], &ni_tendency[ijk],
                    &ice_dep_tend[ijk], &ice_sub_tend[ijk], &qv_tendency[ijk]);
            }
        }
    }
    return;
}

void sb_snow_deposition_wrapper(
        const struct DimStruct *dims, 
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
        // INPUT VARIABLES
        double* restrict temperature, 
        double* restrict qt,         // total water specific humidity
        double* restrict p0,         // air pressure
        double* restrict density,
        double* restrict qs,         // ice specific humidity
        double* restrict ns,         // ice number density
        double dt, 
        // OUTPUT VARIABLES INDEX
        double* restrict qs_tendency,
        double* restrict ns_tendency, 
        double* restrict snow_dep_tend,
        double* restrict snow_sub_tend,
        double* restrict qv_tendency
    ){
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

                double qs_tmp = fmax(qs[ijk],0.0);
                double ns_tmp = fmax(fmin(ns[ijk], qs_tmp/ICE_MIN_MASS),qs_tmp/ICE_MAX_MASS);
                double snow_mass = microphysics_mean_mass(ns_tmp, qs_tmp, ICE_MIN_MASS, ICE_MAX_MASS);
                double Dm_s = SB_ICE_A * pow(snow_mass, SB_ICE_B);
                double velocity_snow = SB_ICE_alpha * pow(snow_mass, SB_ICE_beta) * sqrt(DENSITY_SB/density[k]);
                double sat_ratio_ice = microphysics_saturation_ratio_ice(temperature[ijk], p0[k], qt[ijk]);

                sb_ice_deposition(LT, lam_fp, L_fp, 
                    temperature[ijk], qt[ijk], p0[k], qs_tmp, ns_tmp, 
                    Dm_s, snow_mass, velocity_snow, dt, sat_ratio_ice,
                    &qs_tendency[ijk], &ns_tendency[ijk],
                    &snow_dep_tend[ijk], &snow_sub_tend[ijk], &qv_tendency[ijk]);
            }
        }
    }
    return;
}

void sb_ice_self_collection_wrapper(
        const struct DimStruct *dims, 
        double* restrict temperature, 
        double* restrict qi,         // ice specific humidity
        double* restrict ni,         // ice number density
        double* restrict density,
        double dt,
        double* restrict qs_tendency,
        double* restrict ns_tendency,
        double* restrict qi_tendency,
        double* restrict ni_tendency
    ){
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
                
                double qi_tmp = fmax(qi[ijk],0.0);
                double ni_tmp = fmax(fmin(ni[ijk], qi_tmp/ICE_MIN_MASS),qi_tmp/ICE_MAX_MASS);

                double ice_mass = microphysics_mean_mass(ni_tmp, qi_tmp, ICE_MIN_MASS, ICE_MAX_MASS);
                double Dm_i = SB_ICE_A * pow(ice_mass, SB_ICE_B);
                double velocity_ice = SB_ICE_alpha * pow(ice_mass, SB_ICE_beta) * sqrt(DENSITY_SB/density[k]);

                sb_ice_self_collection(temperature[ijk], qi_tmp, ni_tmp, Dm_i, velocity_ice, dt,
                        &qs_tendency[ijk], &ns_tendency[ijk],
                        &qi_tendency[ijk], &ni_tendency[ijk]);
            }
        }
    }
    return;
}

void sb_snow_self_collection_wrapper(
        const struct DimStruct *dims, 
        double* restrict temperature, 
        double* restrict qs,         // ice specific humidity
        double* restrict ns,         // ice number density
        double* restrict density,
        double dt,
        double* restrict ns_tendency
    ){
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

                double qs_tmp = fmax(qs[ijk],0.0);
                double ns_tmp = fmax(fmin(ns[ijk], qs_tmp/ICE_MIN_MASS),qs_tmp/ICE_MAX_MASS);

                double snow_mass = microphysics_mean_mass(ns_tmp, qs_tmp, ICE_MIN_MASS, ICE_MAX_MASS);
                double Dm_s = SB_ICE_A * pow(snow_mass, SB_ICE_B);
                double velocity_snow = SB_ICE_alpha * pow(snow_mass, SB_ICE_beta) * sqrt(DENSITY_SB/density[k]);
                sb_snow_self_collection(temperature[ijk], qs_tmp, ns_tmp, Dm_s, velocity_snow, dt,
                        &ns_tendency[ijk]);
            }
        }
    }
    return;
}

void sb_snow_ice_collection_wrapper(
        const struct DimStruct *dims, 
        double* restrict temperature, 
        double* restrict qs,         // ice specific humidity
        double* restrict ns,         // ice number density
        double* restrict qi,
        double* restrict ni,
        double* restrict density,
        double dt,
        double* restrict qs_tendency,
        double* restrict qi_tendency,
        double* restrict ni_tendency
    ){
    
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

                double qi_tmp = fmax(qi[ijk],0.0);
                double ni_tmp = fmax(fmin(ni[ijk], qi_tmp/ICE_MIN_MASS),qi_tmp/ICE_MAX_MASS);
                double qs_tmp = fmax(qs[ijk],0.0);
                double ns_tmp = fmax(fmin(ns[ijk], qs_tmp/ICE_MIN_MASS),qs_tmp/ICE_MAX_MASS);

                double ice_mass = microphysics_mean_mass(ni_tmp, qi_tmp, ICE_MIN_MASS, ICE_MAX_MASS);
                double Dm_i = SB_ICE_A * pow(ice_mass, SB_ICE_B);
                double velocity_ice = SB_ICE_alpha * pow(ice_mass, SB_ICE_beta) * sqrt(DENSITY_SB/density[k]);
                double snow_mass = microphysics_mean_mass(ns_tmp, qs_tmp, ICE_MIN_MASS, ICE_MAX_MASS);
                double Dm_s = SB_ICE_A * pow(snow_mass, SB_ICE_B);
                double velocity_snow = SB_ICE_alpha * pow(snow_mass, SB_ICE_beta) * sqrt(DENSITY_SB/density[k]);

                sb_snow_ice_collection(temperature[ijk], qi_tmp, ni_tmp, Dm_i, velocity_ice, 
                        qs_tmp, ns_tmp, Dm_s, velocity_snow, dt,
                        &qs_tendency[ijk], &qi_tendency[ijk], &ni_tendency[ijk]);
            }
        }
    }
    return;
}

void sb_snow_riming_wrapper(
        const struct DimStruct *dims, 
        double* restrict temperature, 
        double* restrict ql,
        double* restrict nl,
        double* restrict qr,
        double* restrict nr,
        double* restrict qs,
        double* restrict ns,
        double* restrict density,
        double* restrict qs_tend_dep,
        double dt,
        double* restrict ql_tendency,
        double* restrict nl_tendency,
        double* restrict qi_tendency,
        double* restrict ni_tendency,
        double* restrict qr_tendency,
        double* restrict nr_tendency,
        double* restrict qs_tendency,
        double* restrict ns_tendency
    ){
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
                
                double ql_tmp = fmax(ql[ijk],0.0);
                double nl_tmp = fmax(fmin(nl[ijk], ql_tmp/LIQUID_MIN_MASS),ql_tmp/LIQUID_MAX_MASS);
                double qr_tmp = fmax(qr[ijk],0.0);
                double nr_tmp = fmax(fmin(nr[ijk], qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                double qs_tmp = fmax(qs[ijk],0.0);
                double ns_tmp = fmax(fmin(ns[ijk], qs_tmp/SB_SNOW_MIN_MASS),qs_tmp/SB_SNOW_MAX_MASS);

                //obtain some parameters of cloud droplets
                double liquid_mass = microphysics_mean_mass(nl_tmp, ql_tmp, LIQUID_MIN_MASS, LIQUID_MAX_MASS);// average mass of cloud droplets
                double Dm_l =  cbrt(liquid_mass * 6.0/DENSITY_LIQUID/pi);
                double velocity_liquid = 3.75e5 * cbrt(liquid_mass)*cbrt(liquid_mass) *(DENSITY_SB/density[k]);

                //obtain some parameters of rain droplets
                double rain_mass = microphysics_mean_mass(nr_tmp, qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS); //average mass of rain droplet
                double Dm_r = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi); // mass weighted diameter of rain droplets
                // simplified rain velocity based on equation 28 in SB06
                double velocity_rain = 159.0 * pow(rain_mass, 0.266) * sqrt(DENSITY_SB/density[k]);

                //obtain some parameters of snow
                double snow_mass = microphysics_mean_mass(ns_tmp, qs_tmp, SB_SNOW_MIN_MASS, SB_SNOW_MAX_MASS);
                double Dm_s = SB_SNOW_A * pow(snow_mass, SB_SNOW_B);
                double velocity_snow = SB_SNOW_alpha * pow(snow_mass, SB_SNOW_beta) * sqrt(DENSITY_SB/density[k]);
                    
                sb_snow_riming(temperature[ijk], ql_tmp, nl_tmp, Dm_l, velocity_liquid, 
                        qr_tmp, nr_tmp, Dm_r, velocity_rain, rain_mass, 
                        qs_tmp, ns_tmp, Dm_s, velocity_snow, dt, qs_tend_dep[ijk],
                        &ql_tendency[ijk], &nl_tendency[ijk],
                        &qi_tendency[ijk], &ni_tendency[ijk],
                        &qr_tendency[ijk], &nr_tendency[ijk],
                        &qs_tendency[ijk], &ns_tendency[ijk]);
            }
        }
    }
    return;
}

void sb_snow_melting_wrapper(
        const struct DimStruct *dims, 
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
        double* restrict p0,
        double* restrict temperature,
        double* restrict qt,
        double* restrict qv,
        double* restrict qs,
        double* restrict ns,
        double* restrict density,
        double dt,
        double* restrict qs_tendency,
        double* restrict ns_tendency,
        double* restrict qr_tendency,
        double* restrict nr_tendency
    ){
    
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
                
                double qs_tmp = fmax(qs[ijk],0.0);
                double ns_tmp = fmax(fmin(ns[ijk], qs_tmp/SB_SNOW_MIN_MASS),qs_tmp/SB_SNOW_MAX_MASS);
                double snow_mass = microphysics_mean_mass(ns_tmp, qs_tmp, SB_SNOW_MIN_MASS, SB_SNOW_MAX_MASS);
                double Dm_s = SB_SNOW_A * pow(snow_mass, SB_SNOW_B);
                double velocity_snow = SB_SNOW_alpha * pow(snow_mass, SB_SNOW_beta) * sqrt(DENSITY_SB/density[k]);
                    
                sb_snow_melting(LT, lam_fp, L_fp, p0[k], temperature[ijk], 
                        qt[ijk], qv[ijk], qs_tmp, ns_tmp, 
                        snow_mass, Dm_s, velocity_snow, dt,
                        &ns_tendency[ijk], &qs_tendency[ijk],
                        &nr_tendency[ijk], &qr_tendency[ijk]);
            }
        }
    }
    return;
}
