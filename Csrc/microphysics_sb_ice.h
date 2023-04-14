#pragma once
#include "parameters.h"
#include "parameters_micro_sb.h"
#include "microphysics.h"
#include "microphysics_sb.h"
#include "advection_interpolation.h"
#include "entropies.h"
#include "thermodynamic_functions.h"
#include <math.h>

// ===========<<< SB Ice phase parameterization >>> ============
// Basic physic parameters definition based on Table 1 in Seifert and Beheng 2006
#define SB_LIQUID_A 0.124 // m kg^-β
#define SB_LIQUID_B 0.333333
#define SB_LIQUID_alpha 3.75e5
#define SB_LIQUID_beta 0.666666
#define SB_LIQUID_lamuda 1.0
#define SB_LIQUID_nu 1.0
#define SB_LIQUID_mu 1.0

#define SB_ICE_A 0.217 // m kg^-β
#define SB_ICE_B 0.302
#define SB_ICE_alpha 317.0
#define SB_ICE_beta 0.363
#define SB_ICE_lamuda 0.5
#define SB_ICE_nu 1.0
#define SB_ICE_mu 0.333333 // 1/3

#define SB_SNOW_A 8.156 // m kg^-β
#define SB_SNOW_B 0.526
#define SB_SNOW_alpha 27.7
#define SB_SNOW_beta 0.216
#define SB_SNOW_lamuda 0.5
#define SB_SNOW_nu 1.0
#define SB_SNOW_mu 0.333333 // 1/3
#define SB_SNOW_MIN_MASS 1.73e-9
#define SB_SNOW_MAX_MASS 1.0e-7
#define SB_N_ICE_MIN 1.579437940972532e+17 //
#define SB_N_ICE_MAX 21601762742.634903 //

#define N_MAX_ICE 1.579437940972532e+17
#define N_MIN_ICE 21601762742.634903


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
    double pv_sat_i = lookup(LT, temperature);

    double g_therm_ice = 1.0/(Rv*temperature/dvapor/pv_sat_i + L_IV/kappa_t/temperature * (L_IV/Rv/temperature - 1.0));
    // double g_therm = 1.0/(Rv*temperature/DVAPOR/pv_sat + L/KT/temperature * (L/Rv/temperature - 1.0));
    return g_therm_ice;
}

// see equation 43 and Appendix B in SB06
double sb_ventilation_coefficient(
        double n, // n-th moment;
        double Dm, // mass weighted diameter
        double velocity, // falling velocity of particle
        double mass, // mass
        double sb_b, // mass-related parameter b of particle
        double sb_beta, // mass-related parameter beta of particle
        double nu, // ν of particle
        double mu // μ of particle
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
        double k, // k-th moment
        double sb_b, // mass-related parameter b of particle 
        double nu, // ν of particle
        double mu // μ of particle
        ){

    double var_1 = tgamma( (2.0*sb_b + nu + 1.0 + k) / mu );
    double var_2 = tgamma( (nu + 1.0) / mu );
    double var_3 = tgamma( (nu + 2.0) / mu );
    double var_exponent = 2.0*sb_b + k; 

    return (var_1/var_2) * pow( (var_2/var_3), var_exponent );
}

// following the equation 91 in SB06
double sb_collection_delta_ab(
        double k, // k-th moment
        double sb_b_a, // mass-related parameter b of particle a
        double nu_a, // ν of particle a
        double mu_a, // μ of particle a
        double sb_b_b, // mass-related parameter b of particle b
        double nu_b, // ν of particle b
        double mu_b // μ of particle b
        ){
    
    double var_a_1 = tgamma( (sb_b_a + nu_a + 1.0 + k) / mu_a );
    double var_a_2 = tgamma( (nu_a + 1.0) / mu_a );
    double var_a_3 = tgamma( (nu_a + 2.0) / mu_a );

    double var_b_1 = tgamma( (sb_b_b + nu_b + 1.0) / mu_b );
    double var_b_2 = tgamma( (nu_b + 1.0) / mu_b );
    double var_b_3 = tgamma( (nu_b + 2.0) / mu_b );

    return 2.0 * (var_a_1/var_a_2) * (var_b_1/var_b_2) * pow((var_a_2/var_a_3), (sb_b_a + k)) * pow((var_b_2/var_b_3), sb_b_b);
}

// following the equation 92 in SB06
double sb_collection_vartheta_b(
        double k, // k-th moment
        double sb_b, // mass-related parameter b of particle 
        double sb_beta, // mass-related parameter β of particle 
        double nu, // ν of particle
        double mu // μ of particle
        ){

    double var_1 = tgamma( (2.0*sb_beta + 2.0*sb_b + nu + 1.0 + k) / mu );
    double var_2 = tgamma( (2.0*sb_b + nu + 1.0 + k) / mu  );
    double var_3 = tgamma( (nu + 1.0) / mu );
    double var_4 = tgamma( (nu + 2.0) / mu );

    return var_1/var_2 * pow((var_3/var_4), 2*sb_beta);
}

// following the equation 93 in SB06
double sb_collection_vartheta_ab(
        double k, // k-th moment
        double sb_b_a, // mass-related parameter b of particle a
        double sb_beta_a, // mass-related parameter β of particle a
        double nu_a, // ν of particle a
        double mu_a, // μ of particle a
        double sb_b_b, // mass-related parameter b of particle b
        double sb_beta_b, // mass-related parameter β of particle b
        double nu_b, // ν of particle b
        double mu_b // μ of particle b
        ){
    
    double var_a_1 = tgamma( (sb_beta_a + sb_b_a + nu_a + 1.0 + k) / mu_a );
    double var_a_2 = tgamma( (sb_b_a + nu_a + 1.0 + k) / mu_a );
    double var_a_3 = tgamma( (nu_a + 1.0) / mu_a );
    double var_a_4 = tgamma( (nu_a + 2.0) / mu_a );

    double var_b_1 = tgamma( (sb_beta_b + sb_b_b + nu_b + 1.0) / mu_b );
    double var_b_2 = tgamma( (sb_b_b + nu_b + 1.0) / mu_b );
    double var_b_3 = tgamma( (nu_b + 1.0) / mu_b );
    double var_b_4 = tgamma( (nu_b + 2.0) / mu_b );

    return 2.0 * (var_a_1/var_a_2) * (var_b_1/var_b_2) * pow((var_a_3/var_a_4), sb_beta_a) * pow((var_b_3/var_b_4), sb_beta_b);
}

// see equation 67 in SB06
double sticking_efficiencies(double T){
    return exp(0.09 * (T - 273.15));
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

void sb_aggretion_snow(
        // INPUT VARIABLES
        double ql, // cloud liquid specific humidity
        double qi, // clodu ice specific humidity
        double qs, // snow specific humidity
        double nl, // cloud liquid number density
        double ni, // clodu ice number density
        double ns, // snow number density
        double temperature,
        double Dm_l, // mass-weighted diameter of cloud liquid droplet
        double Dm_i, // mass-weighted diameter of cloud ice particle
        double Dm_s, // mass-weighted diameter of snow
        double velocity_liquid, // falling velocity of cloud liquid droplet
        double velocity_ice, // falling velocity of cloud ice particle
        double velocity_snow, // falling velocity of snow
        // OUTPUT VARIABLES INDEX
        double* ql_tendency, double* qi_tendency, 
        double* nl_tendency, double* ni_tendency,
        double* ns_tendency, double* qs_tendency){

    // The aggretion component include:
    // - ice aggretion to snow (i+i=s);
    // - snow aggretion ice particles (s+i=s);
    // - riming of cloud droplet (s+l=s);
    // - snow  elf aggretion (s+s=s)
    
    // define collection parameter delta and varthera
    double delta_i_0, delta_s_0, delta_l_0, delta_ii_0, delta_si_0, delta_sl_0, delta_ss_0;
    double vartheta_i_0, vartheta_s_0, vartheta_l_0, vartheta_ii_0, vartheta_si_0, vartheta_sl_0, vartheta_ss_0;
    double delta_i_1, delta_s_1, delta_l_1, delta_ii_1, delta_si_1, delta_sl_1, delta_ss_1;
    double vartheta_i_1, vartheta_s_1, vartheta_l_1, vartheta_ii_1, vartheta_si_1, vartheta_sl_1, vartheta_ss_1;

    // collision efficiencies E_ab:
    double E_ii, E_si, E_sl, E_ss;
    double epsilon_a, epsilon_b;
    double q_ii_tend, n_ii_tend, q_si_tend, n_si_tend, q_sl_tend, n_sl_tend, q_ss_tend, n_ss_tend;
    
    // first is the aggretion of ice to snow (i+i=s)
    if(qi > SB_EPS){
        delta_i_0 = sb_collection_delta_b(0.0, SB_ICE_B, SB_ICE_nu, SB_ICE_mu);
        delta_ii_0 = sb_collection_delta_ab(0.0, SB_ICE_B, SB_ICE_nu, SB_ICE_mu, SB_ICE_B, SB_ICE_nu, SB_ICE_mu);
        vartheta_i_0 = sb_collection_vartheta_b(0.0, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu);
        vartheta_ii_0 = sb_collection_vartheta_ab(0.0, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu);
        delta_i_1 = sb_collection_delta_b(1.0, SB_ICE_B, SB_ICE_nu, SB_ICE_mu);
        delta_ii_1 = sb_collection_delta_ab(1.0, SB_ICE_B, SB_ICE_nu, SB_ICE_mu, SB_ICE_B, SB_ICE_nu, SB_ICE_mu);
        vartheta_i_1 = sb_collection_vartheta_b(1.0, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu);
        vartheta_ii_1 = sb_collection_vartheta_ab(1.0, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu);
        
        epsilon_a = 0.2; // m s^-1
        epsilon_b = 0.2; // m s^-1
        E_ii = sticking_efficiencies(temperature); 
        
        // see equation 61-63 in SB06
        // with ICE as species "a" and ICE as species "b"

        q_ii_tend = pi/4.0 * E_ii * ni * qi * (delta_i_0*Dm_i*Dm_i + delta_ii_1*Dm_i*Dm_i + delta_i_1*Dm_i*Dm_i) * 
            pow((vartheta_i_0*velocity_ice*velocity_ice - vartheta_ii_1*velocity_ice*velocity_ice + vartheta_i_1*velocity_ice*velocity_ice + epsilon_a + epsilon_b), 0.5);
        n_ii_tend = pi/4.0 * E_ii * ni * ni * (delta_i_0*Dm_i*Dm_i + delta_ii_0*Dm_i*Dm_i + delta_i_0*Dm_i*Dm_i) * 
            pow((vartheta_i_0*velocity_ice*velocity_ice - vartheta_ii_0*velocity_ice*velocity_ice + vartheta_i_0*velocity_ice*velocity_ice + epsilon_a + epsilon_b), 0.5);
    }
    else{
        q_ii_tend = 0.0;
        n_ii_tend = 0.0;
    }
    
    // second is the aggretion of snow from ice (s+i=s)
    if (qi > SB_EPS && qs > SB_EPS && ns > SB_EPS){
        delta_s_0 = sb_collection_delta_b(0.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu);
        delta_i_0 = sb_collection_delta_b(0.0, SB_ICE_B, SB_ICE_nu, SB_ICE_mu);
        delta_si_0 = sb_collection_delta_ab(0.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu, SB_ICE_B, SB_ICE_nu, SB_ICE_mu);

        vartheta_s_0 = sb_collection_vartheta_b(0.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);
        vartheta_i_0 = sb_collection_vartheta_b(0.0, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu);
        vartheta_si_0 = sb_collection_vartheta_ab(0.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu);

        delta_i_1 = sb_collection_delta_b(1.0, SB_ICE_B, SB_ICE_nu, SB_ICE_mu);
        delta_si_1 = sb_collection_delta_ab(1.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu, SB_ICE_B, SB_ICE_nu, SB_ICE_mu);

        vartheta_i_1 = sb_collection_vartheta_b(1.0, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu);
        vartheta_si_1 = sb_collection_vartheta_ab(1.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu, SB_ICE_B, SB_ICE_beta, SB_ICE_nu, SB_ICE_mu);
        
        epsilon_a = 0.2; // m s^-1
        epsilon_b = 0.2; // m s^-1

        E_si = 1.0;
    
        // see equation 61-63 in SB06
        // with SNOW as species "a" and ICE as species "b"
    
        q_si_tend = pi/4.0 * E_si * ns * qi * (delta_s_0*Dm_s*Dm_s + delta_si_1*Dm_s*Dm_i + delta_i_1*Dm_i*Dm_i) * 
            pow((vartheta_s_0*velocity_snow*velocity_snow - vartheta_si_1*velocity_snow*velocity_ice + vartheta_i_1*velocity_ice*velocity_ice + epsilon_a + epsilon_b), 0.5);
        n_si_tend = pi/4.0 * E_si * ns * ni * (delta_s_0*Dm_s*Dm_s + delta_si_0*Dm_s*Dm_i + delta_i_0*Dm_i*Dm_i) * 
            pow((vartheta_s_0*velocity_snow*velocity_snow - vartheta_si_0*velocity_snow*velocity_ice + vartheta_i_0*velocity_ice*velocity_ice + epsilon_a + epsilon_b), 0.5);
    }
    else{
        q_si_tend = 0.0;
        n_si_tend = 0.0;
    }

    // third is the riming of snow from cloud droplet (s+l=s)
    if (ql > SB_EPS && qs > SB_EPS && ns > SB_EPS){
        delta_s_0 = sb_collection_delta_b(0.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu);
        delta_l_0 = sb_collection_delta_b(0.0, SB_LIQUID_B, SB_LIQUID_nu, SB_LIQUID_mu);
        delta_sl_0 = sb_collection_delta_ab(0.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu, SB_LIQUID_B, SB_LIQUID_nu, SB_LIQUID_mu);
        
        vartheta_s_0 = sb_collection_vartheta_b(0.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);
        vartheta_l_0 = sb_collection_vartheta_b(0.0, SB_LIQUID_B, SB_LIQUID_beta, SB_LIQUID_nu, SB_LIQUID_mu);
        vartheta_sl_0 = sb_collection_vartheta_ab(0.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu, SB_LIQUID_B, SB_LIQUID_beta, SB_LIQUID_nu, SB_LIQUID_mu);

        delta_l_1 = sb_collection_delta_b(1.0, SB_LIQUID_B, SB_LIQUID_nu, SB_LIQUID_mu);
        delta_sl_1 = sb_collection_delta_ab(1.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu, SB_LIQUID_B, SB_LIQUID_nu, SB_LIQUID_mu);
        vartheta_l_1 = sb_collection_vartheta_b(1.0, SB_LIQUID_B, SB_LIQUID_beta, SB_LIQUID_nu, SB_LIQUID_mu);
        vartheta_sl_1 = sb_collection_vartheta_ab(1.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu, SB_LIQUID_B, SB_LIQUID_beta, SB_LIQUID_nu, SB_LIQUID_mu);

        epsilon_a = 0.2; // m s^-1 for snow 
        epsilon_b = 0.0; // m s^-1 for cloud droplet
        
        E_sl = collection_efficiencies_cloud(Dm_s, Dm_l);
        
        // see equation 61-63 in SB06
        // with SNOW as species "a" and CLOUD DROPLET as species "b"
        
        q_sl_tend = pi/4.0 * E_si * ns * ql * (delta_s_0*Dm_s*Dm_s + delta_sl_1*Dm_s*Dm_i + delta_l_1*Dm_i*Dm_i) * 
            pow((vartheta_s_0*velocity_snow*velocity_snow - vartheta_sl_1*velocity_snow*velocity_ice + vartheta_l_1*velocity_ice*velocity_ice + epsilon_a + epsilon_b), 0.5);
        n_sl_tend = pi/4.0 * E_si * ns * nl * (delta_s_0*Dm_s*Dm_s + delta_sl_0*Dm_s*Dm_i + delta_l_0*Dm_i*Dm_i) * 
            pow((vartheta_s_0*velocity_snow*velocity_snow - vartheta_sl_0*velocity_snow*velocity_ice + vartheta_l_0*velocity_ice*velocity_ice + epsilon_a + epsilon_b), 0.5);
    }
    else{
        q_sl_tend = 0.0;
        n_sl_tend = 0.0;
    }
    
    // last if the self aggretion of snow (s+s=s)
    if (qs > SB_EPS && ns > SB_EPS){

        delta_s_0 = sb_collection_delta_b(0.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu);
        delta_ss_0 = sb_collection_delta_ab(0.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu);
        vartheta_s_0 = sb_collection_vartheta_b(0.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);
        vartheta_ss_0 = sb_collection_vartheta_ab(0.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);
        
        epsilon_a = 0.2; // m s^-1 for snow 
        epsilon_b = 0.2; // m s^-1 for snow
        
        E_ss = sticking_efficiencies(temperature); 

        // see equation 61-63 in SB06
        // with SNOW as species "a" and SNOW as species "b"

        n_ss_tend = pi/4.0 * E_ss * ns * ns * (delta_s_0*Dm_s*Dm_s + delta_ss_0*Dm_s*Dm_s + delta_s_0*Dm_s*Dm_s) * 
            pow((vartheta_s_0*velocity_snow*velocity_snow - vartheta_ss_0*velocity_snow*velocity_snow + vartheta_s_0*velocity_snow*velocity_snow + epsilon_a + epsilon_b), 0.5);
    }
    else{
        q_ss_tend = 0.0;
        n_ss_tend = 0.0;
    }
    // ================================================
    // ToDo: find out the number density effect of aggretion
    // ================================================
    *qs_tendency = q_ii_tend + q_si_tend + q_sl_tend;
    *qi_tendency = -q_ii_tend -q_si_tend;
    *ql_tendency = -q_sl_tend;

    *ni_tendency = -n_ii_tend -n_si_tend;
    *nl_tendency = -n_sl_tend;
    *ns_tendency = n_ss_tend;

    return;
}

void sb_diffusion_snow(
        // thermodynamic settings
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
        // INPUT VARIABLES
        double temperature, 
        double qt, // total water specific humidity
        double p0, // air pressure
        double Dm_s, // mass-weighted diameter of snow
        double snow_mass, // average mass of snow
        double velocity_snow, // falling velocity of snow
        double qs, // snow specific humidity
        double ns, // snow number density
        // OUTPUT VARIABLES INDEX
        double* ns_tendency, double* qs_tendency
        ){

    // The diffusion section includes two main content:
    // - deposition if sat_ratio > 0.0 (over saturated);
    // - sublimation if sat_ratio < 0.0 (under saturated)

    double qs_tendency_dep = 0.0;
    double qs_tendency_sub = 0.0;
    double qs_tendency_diff =0.0;
    
    // calculate the sat_ratio 
    double pv_star = lookup(LT, temperature);
    double qv_star = qv_star_c(p0, qt, pv_star);
    double satratio = qt/qv_star - 1.0;
    
    // specific setting for snow
    double c_i = 2.0;

    if(qs > 1e-12 && ns > 1e-12){

        // calculate g_therm factor during vapor diffusion
        double g_therm_ice = microphysics_g_sb_ice(LT, lam_fp, L_fp, temperature, DVAPOR, KT);
        double F_v_snow = 0.78 + 0.308*NSC_3*sqrt(Dm_s*velocity_snow/KIN_VISC_AIR);

        if(satratio >= 0.0){
            double qs_tendency_dep = 4.0*pi/c_i * g_therm_ice * Dm_s * F_v_snow * satratio;
        }
        else{
            double qs_tendency_sub = 4.0*pi/c_i * g_therm_ice * Dm_s * F_v_snow * satratio;
        }
        qs_tendency_diff = qs_tendency_dep + qs_tendency_sub;
        *qs_tendency = qs_tendency_diff;
    }
    else{
        *qs_tendency = 0.0;
        *ns_tendency = 0.0;
    }
    return;
}

void sb_melting_snow(
        // thermodynamic settings
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        // INPUT VARIABLES
        double p0, // air pressure
        double qt, // total water specific humidity
        double qv, // vapor specific humidity
        double qs, // snow specific humidity
        double ns, // snow number density
        double temperature,
        double snow_mass, // average mass of snow
        double Dm_s, // mass-weighted diameter of snow
        double velocity_snow, // falling velocity of snow
        // OUTPUT VARIABLES INDEX
        double* ns_tendency, double* qs_tendency
        ){

    double t_3 = 275.15;
    double lam = lam_fp(t_3);
    double L = L_fp(t_3,lam);
    double pv_sat = lookup(LT, t_3); // saturated vapor pressure at t_3;

    double pv = pv_c(p0, qt, qv); // partial vapor pressure
    double DT; // diffusivity of heat

    if (qs > SB_EPS && ns > SB_EPS && temperature > 273.15){
        double F_v_0 = 0.78 + 0.308*NSC_3*sqrt(Dm_s*velocity_snow/KIN_VISC_AIR);
        // double F_v_0 = sb_ventilation_coefficient(0.0, Dm_s, velocity_snow, snow_mass, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);
        double F_v_1 = 0.78 + 0.308*NSC_3*sqrt(Dm_s*velocity_snow/KIN_VISC_AIR);
        // double F_v_1 = sb_ventilation_coefficient(1.1, Dm_s, velocity_snow, snow_mass, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);

        double thermo_melt = (KT*DT/DVAPOR)*(temperature - t_3) 
                             + DVAPOR*L/Rv * (pv/temperature - pv_sat/t_3);

        *qs_tendency = 2.0*pi/L_MELTING * thermo_melt * ns * Dm_s * snow_mass * F_v_1;
        *ns_tendency = 2.0*pi/L_MELTING * thermo_melt * ns * Dm_s * 1.0/snow_mass * F_v_0;
    }
    else{
        *qs_tendency = 0.0;
        *ns_tendency = 0.0;
    }
}


void sb_ice_microphysics_sources(const struct DimStruct *dims, 
        // thermodynamic settings
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        // two-moment specific settings based on SB08
        double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
        // INPUT VARIABLES ARRAY
        double* restrict density, // reference air density
        double* restrict p0, // reference air pressure
        double* restrict temperature,  // temperature of air parcel
        double* restrict qt, // total water specific humidity
        double ccn, // given cloud condensation nuclei
        double in, // given ice nuclei
        double* restrict ql, // cloud liquid water specific humidity
        double* restrict qi, // cloud ice water specific humidity
        double* restrict nr, // rain droplet number density
        double* restrict qr, // rain droplet specific humidity
        double* restrict qs, // snow specific humidity
        double* restrict ns, // snow number density
        double dt, // timestep
        //OUTPUT ARRAYS 
        double* restrict nr_tendency_micro, double* restrict qr_tendency_micro, 
        double* restrict nr_tendency, double* restrict qr_tendency, 
        double* restrict ns_tendency_micro, double* restrict qs_tendency_micro, 
        double* restrict ns_tendency, double* restrict qs_tendency
        ){

    //Here we compute the source terms for nr, qr and ns, qs (number and mass of rain and snow)
    //Temporal substepping is used to help ensure boundedness of moments
    
    double nr_tendency_tmp, qr_tendency_tmp,ql_tendency_tmp, ql_tendency_agg, nl_tendency_tmp;
    double nr_tendency_au, nr_tendency_scbk, nr_tendency_evp;
    double qr_tendency_au, qr_tendency_ac, qr_tendency_evp;
    
    double ns_tendency_tmp, qs_tendency_tmp, qi_tendency_tmp, qi_tendency_agg, ni_tendency_tmp;
    double ns_tendency_agg, ns_tendency_diff, ns_tendency_dep, ns_tendency_sub, ns_tendency_melt;
    double qs_tendency_agg, qs_tendency_diff, qs_tendency_dep, qs_tendency_sub, qs_tendency_melt;
    
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

                qr[ijk] = fmax(qr[ijk],0.0);
                nr[ijk] = fmax(fmin(nr[ijk], qr[ijk]/RAIN_MIN_MASS),qr[ijk]/RAIN_MAX_MASS);
                qs[ijk] = fmax(qs[ijk],0.0);
                ns[ijk] = fmax(fmin(ns[ijk], qs[ijk]/SB_SNOW_MIN_MASS),qs[ijk]/SB_SNOW_MAX_MASS);

                double qt_tmp = qt[ijk];
                double ql_tmp = fmax(ql[ijk],0.0);
                double qi_tmp = fmax(qi[ijk],0.0);
                double qv_tmp = qt_tmp - ql_tmp - qi_tmp;
                double nl = ccn/density[k];
                double iwc = fmax(qi_tmp * density[k], SB_EPS);
                double ni = fmax(fmin(in, iwc*N_MAX_ICE),iwc*N_MIN_ICE);

                double qr_tmp = fmax(qr[ijk],0.0);
                double nr_tmp = fmax(fmin(nr[ijk], qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                double qs_tmp = fmax(qs[ijk],0.0);
                double ns_tmp = fmax(fmin(ns[ijk], qs_tmp/SB_SNOW_MIN_MASS),qs_tmp/SB_SNOW_MAX_MASS);

                // define rain sand snow variables
                // and thermodynamic_variables 
                double sat_ratio, g_therm_rain;
                double liquid_mass, Dm_l, velocity_liquid;
                double ice_mass, Dm_i, velocity_ice;
                double rain_mass, Dm_r, mu, Dp, velocity_rain;
                double snow_mass, Dm_s, velocity_snow;

                double time_added = 0.0, dt_, rate;
                ssize_t iter_count = 0;

                do{
                    iter_count       += 1;
                    sat_ratio         = microphysics_saturation_ratio(LT, temperature[ijk], p0[k], qt_tmp);
                    nr_tendency_au    = 0.0;
                    nr_tendency_scbk  = 0.0;
                    nr_tendency_evp   = 0.0;
                    qr_tendency_au    = 0.0;
                    qr_tendency_ac    = 0.0;
                    qr_tendency_evp   = 0.0;
                    
                    ns_tendency_agg  = 0.0;
                    ns_tendency_diff = 0.0;
                    ns_tendency_dep  = 0.0;
                    ns_tendency_sub  = 0.0;
                    ns_tendency_melt = 0.0;
                    qs_tendency_agg  = 0.0;
                    qs_tendency_diff = 0.0;
                    qs_tendency_dep  = 0.0;
                    qs_tendency_sub  = 0.0;
                    qs_tendency_melt = 0.0;

                    //obtain some parameters of cloud droplets
                    liquid_mass = microphysics_mean_mass(nl, ql_tmp, LIQUID_MIN_MASS, LIQUID_MAX_MASS);// average mass of cloud droplets
                    Dm_l =  cbrt(liquid_mass * 6.0/DENSITY_LIQUID/pi);
                    velocity_liquid = 3.75e5 * cbrt(liquid_mass)*cbrt(liquid_mass) *(DENSITY_SB/density[k]);

                    //obtain some parameters of cloud ice particles
                    ice_mass = microphysics_mean_mass(ni, qi_tmp, ICE_MAX_MASS, ICE_MAX_MASS);// average mass of cloud droplets
                    Dm_i = SB_ICE_A * pow(ice_mass, SB_ICE_B);
                    velocity_snow = SB_ICE_alpha * pow(ice_mass, SB_ICE_beta) * sqrt(DENSITY_SB/density[k]);
                    velocity_ice = 

                    //obtain some parameters of rain droplets
                    rain_mass = microphysics_mean_mass(nr_tmp, qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS); //average mass of rain droplet
                    Dm_r = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi); // mass weighted diameter of rain droplets
                    mu = rain_mu(density[k], qr_tmp, Dm_r);
                    Dp = sb_Dp(Dm_r, mu);
                    // simplified rain velocity based on equation 28 in SB06
                    velocity_rain = 159.0 * pow(rain_mass, 0.266) * sqrt(DENSITY_SB/density[k]);

                    //obtain some parameters of snow
                    snow_mass = microphysics_mean_mass(ns_tmp, qs_tmp, SB_SNOW_MIN_MASS, SB_SNOW_MAX_MASS);
                    Dm_s = SB_SNOW_A * pow(ice_mass, SB_SNOW_B);
                    velocity_snow = SB_SNOW_alpha * pow(snow_mass, SB_SNOW_beta) * sqrt(DENSITY_SB/density[k]);

                    g_therm_rain = microphysics_g(LT, lam_fp, L_fp, temperature[ijk]);

                    //compute the source terms
                    sb_autoconversion_rain(droplet_nu, density[k], nl, ql_tmp, qr_tmp, 
                            &nr_tendency_au, &qr_tendency_au);
                    sb_accretion_rain(density[k], ql_tmp, qr_tmp, &qr_tendency_ac);
                    sb_selfcollection_breakup_rain(density[k], nr_tmp, qr_tmp, mu, rain_mass, Dm_r, 
                            &nr_tendency_scbk);
                    sb_evaporation_rain(g_therm_rain, sat_ratio, nr_tmp, qr_tmp, mu, rain_mass, Dp, Dm_r, 
                            &nr_tendency_evp, &qr_tendency_evp);

                    sb_aggretion_snow(ql_tmp, qi_tmp, qs_tmp, nl, ni, ns_tmp, temperature[ijk],
                            Dm_l, Dm_i, Dm_s, velocity_liquid, velocity_ice, velocity_snow,
                            &ql_tendency_agg, &qi_tendency_agg, 
                            &nl_tendency_tmp, &ni_tendency_tmp, 
                            &ns_tendency_agg, &qs_tendency_agg);
                    sb_diffusion_snow(LT, lam_fp, L_fp, temperature[ijk], qt_tmp, p0[k], 
                            Dm_s, snow_mass, velocity_snow, qs_tmp, ns_tmp, 
                            &ns_tendency_diff, &qs_tendency_diff);
                    sb_melting_snow(LT, lam_fp, L_fp, p0[k], qt_tmp, qv_tmp, temperature[ijk],
                            qs_tmp, ns_tmp, snow_mass, Dm_s, velocity_snow,
                            &ns_tendency_melt, &qs_tendency_melt);

                    //find the maximum substep time
                    dt_ = dt - time_added;
                    //check the source term magnitudes
                    nr_tendency_tmp = nr_tendency_au + nr_tendency_scbk + nr_tendency_evp;
                    qr_tendency_tmp = qr_tendency_au + qr_tendency_ac + qr_tendency_evp;

                    ns_tendency_tmp = ns_tendency_agg + ns_tendency_diff + ns_tendency_melt;
                    qs_tendency_tmp = qs_tendency_agg + qs_tendency_diff + qs_tendency_melt;
                    
                    // ql_tendency_agg is negative
                    ql_tendency_tmp = -qr_tendency_au - qr_tendency_ac + ql_tendency_agg;
                    // qi_tendency_agg is negative
                    qi_tendency_tmp = qi_tendency_agg;

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

                    //Integrate forward in time
                    ql_tmp += ql_tendency_tmp * dt_;
                    qi_tmp += qi_tendency_tmp * dt_;
                    
                    nl += ql_tendency_tmp * dt_;
                    ni += qi_tendency_tmp * dt_;

                    qr_tmp += qr_tendency_tmp * dt_;
                    nr_tmp += nr_tendency_tmp * dt_;

                    qs_tmp += qs_tendency_tmp * dt_;
                    ns_tmp += ns_tendency_tmp * dt_;

                    qv_tmp += -(qr_tendency_evp + qs_tendency_diff) * dt_;

                    qr_tmp = fmax(qr_tmp,0.0);
                    nr_tmp = fmax(fmin(nr_tmp, qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                    qs_tmp = fmax(qs_tmp,0.0);
                    ns_tmp = fmax(fmin(ns_tmp, qs_tmp/SB_SNOW_MIN_MASS),qs_tmp/SB_SNOW_MAX_MASS);
                    ql_tmp = fmax(ql_tmp,0.0);
                    nl = fmax(fmin(nl, ql_tmp/LIQUID_MIN_MASS),ql_tmp/LIQUID_MAX_MASS);
                    qi_tmp = fmax(qi_tmp,0.0);
                    ni = fmax(fmin(ni, qi_tmp/ICE_MIN_MASS),qi_tmp/ICE_MAX_MASS);

                    qi_tmp = fmax(qi_tmp,0.0);
                    qt_tmp = ql_tmp + qv_tmp + qi_tmp;
                    time_added += dt_ ;

                }while(time_added < dt);

                nr_tendency_micro[ijk] = (nr_tmp - nr[ijk] )/dt;
                qr_tendency_micro[ijk] = (qr_tmp - qr[ijk])/dt;
                ns_tendency_micro[ijk] = (ns_tmp - ns[ijk] )/dt;
                qs_tendency_micro[ijk] = (qs_tmp - qs[ijk])/dt;

                nr_tendency[ijk] += nr_tendency_micro[ijk];
                qr_tendency[ijk] += qr_tendency_micro[ijk];
                ns_tendency[ijk] += ns_tendency_micro[ijk];
                qs_tendency[ijk] += qs_tendency_micro[ijk];
            }
        }
    }
    return;
}

void sb_sedimentation_velocity_snow(const struct DimStruct *dims, 
        // INPUT VARIABLES ARRAY
        double* restrict ns, // snow number density 
        double* restrict qs, // snow specific humidity
        double* restrict density, // reference density of air
        // OUTPUT VARIABLES 
        double* restrict ns_velocity, 
        double* restrict qs_velocity){

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
                double ice_mass = microphysics_mean_mass(ns_tmp, qs_tmp, SB_SNOW_MIN_MASS, SB_SNOW_MAX_MASS);
                
                double ns_vel_tmp = SB_SNOW_alpha * tgamma(6.0 + 3.0*SB_SNOW_beta)/tgamma(6.0) * pow(tgamma(6.0)/tgamma(9.0), SB_SNOW_beta) * pow(ice_mass, SB_SNOW_beta);
                double qs_vel_tmp = SB_SNOW_alpha * tgamma(9.0 + 3.0*SB_SNOW_beta)/tgamma(9.0) * pow(tgamma(6.0)/tgamma(9.0), SB_SNOW_beta) * pow(ice_mass, SB_SNOW_beta);
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

void sb_ice_qt_source_formation(const struct DimStruct *dims, 
        double* restrict qr_tendency,
        double* restrict qs_tendency, 
        double* restrict qt_tendency){

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
                qt_tendency[ijk] += -qr_tendency[ijk] - qs_tendency[ijk];
            }
        }
    }
    return;
}
