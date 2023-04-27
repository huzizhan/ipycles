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

// ------Ice Multiplication Coefficients------------
#define T_MULT_MIN 265.0 // K
#define T_MULT_MAX 270.0 // K
#define T_MULT_OPT 268.0 //
#define C_MULT 3.5e8 // TODO find the defination of this variable

// ------Riming of snow to graupel------------------
#define T_MAX_GR_RIME 270.16 // K
// ------ Other parameters adopted from Arc1M scheme
#define SB_N_ICE_MIN 1.579437940972532e+17 //
#define SB_N_ICE_MAX 21601762742.634903 //

#define N_MAX_ICE 1.579437940972532e+17
#define N_MIN_ICE 21601762742.634903

#define T_3 273.15 


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
    return fmax(0.1, fmin(exp(0.09*(T-T_3)), 1.0));
}

double cotton_efficiency(double T){
    double base = pow(10.0, (0.035*(T-T_3)-0.7));
    return fmin(base, 0.2);
}
// this component follows the equation 64-66 in SB06
double collection_efficiencies_cloud(double Dm_e, double Dm_c){
    double E_l, E_e;
    // calculation of E_l
    if(Dm_c < 1.5e-5){
        E_l = 0.0;
    }
    else if(Dm_c <= 4.0e-5){
        E_l = (Dm_c - 1.5e-5)/2.5e-5;
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

void sb_ice_self_collection(
    // INPUT variables 
    double T, // ambient atmosphere temperature 
    double qi, // ice specific content
    double ni, // ice number density
    double Dm_i, // ice diameter
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
    double D_crit_ii = 5.0e-6; // Diameter threshold for ice_selfcollection
    double qi_crit_ii = 1.0e-6; // q threshold for ice_selfcollection
    // TODO change the unit of qi: kg/m3 and D

    if(ni > 0.0 && qi > qi_crit_ii && Dm_i > D_crit_ii){
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
        double x_conv_ii = pow((D_crit_ii/SB_SNOW_A),(1.0/SB_SNOW_B));

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
    double T, // ambient atmosphere temperature 
    double qs, // snow specific humidity
    double ns, // snow number density 
    double Dm_s, // snow diameter
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
    double qs_crit_ss = 1.0e-9; // q threshold for ice_selfcollection
    // TODO change the unit of snow specific humidity of qs_crit_ss
    

    if(qs > qs_crit_ss){
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
    double qi_crit = 1.0e-6; // q threshold of ice for snow ice collection
    double qs_crit = 1.0e-6; // q threshold of snow for snow ice collection

    if (qi > qi_crit && qs > qs_crit){
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
    double nc,
    double qc,
    double Dm_c,
    double velocity_c,
    double ns,
    double qs,
    double Dm_s,
    double velocity_s,
    double* q_sc_tend,
    double* n_sc_tend
){
    // snow riming cloud droplet: s+i -> s
    // define local variables
    double E_sc; // collection efficience
    double epsilon_s, epsilon_c;
    double qc_crit = 1.0e-6; // q threshold of cloud droplet for snow cloud collection
    double Dm_c_crit = 1.0e-6; // D threshold of ice for snow ice collection

    if (qc > qc_crit && qs > qc_crit && Dm_c > Dm_c_crit && Dm_s > Dm_c_crit){
        double delta_s_0 = sb_collection_delta_b(0.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu);
        double delta_c_0 = sb_collection_delta_b(0.0, SB_LIQUID_B, SB_LIQUID_nu, SB_LIQUID_mu);
        double delta_sc_0 = sb_collection_delta_ab(0.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu, SB_LIQUID_B, SB_LIQUID_nu, SB_LIQUID_mu);
        double delta_c_1 = sb_collection_delta_b(1.0, SB_LIQUID_B, SB_LIQUID_nu, SB_LIQUID_mu);
        double delta_sc_1 = sb_collection_delta_ab(1.0, SB_SNOW_B, SB_SNOW_nu, SB_SNOW_mu, SB_LIQUID_B, SB_LIQUID_nu, SB_LIQUID_mu);

        double vartheta_s_0 = sb_collection_vartheta_b(0.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);
        double vartheta_c_0 = sb_collection_vartheta_b(0.0, SB_LIQUID_B, SB_LIQUID_beta, SB_LIQUID_nu, SB_LIQUID_mu);
        double vartheta_sc_0 = sb_collection_vartheta_ab(0.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu, SB_LIQUID_B, SB_LIQUID_beta, SB_LIQUID_nu, SB_LIQUID_mu);
        double vartheta_c_1 = sb_collection_vartheta_b(1.0, SB_LIQUID_B, SB_LIQUID_beta, SB_LIQUID_nu, SB_LIQUID_mu);
        double vartheta_sc_1 = sb_collection_vartheta_ab(1.0, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu, SB_LIQUID_B, SB_LIQUID_beta, SB_LIQUID_nu, SB_LIQUID_mu);

        E_sc = collection_efficiencies_cloud(Dm_s, Dm_c);
        epsilon_s = 0.2;
        epsilon_c = 0.0;

        *n_sc_tend = pi/4.0 * E_sc * ns * nc * (delta_s_0*Dm_s*Dm_s + delta_sc_0*Dm_s*Dm_c + delta_c_0*Dm_c*Dm_c) * 
            sqrt(vartheta_s_0*velocity_s*velocity_s - vartheta_sc_0*velocity_s*velocity_c + vartheta_c_0*velocity_c*velocity_c + epsilon_s + epsilon_c);
        *q_sc_tend = pi/4.0 * E_sc * ns * qc * (delta_s_0*Dm_s*Dm_s + delta_sc_1*Dm_s*Dm_c + delta_c_1*Dm_c*Dm_c) * 
            sqrt(vartheta_s_0*velocity_s*velocity_s - vartheta_sc_1*velocity_s*velocity_c + vartheta_c_1*velocity_c*velocity_c + epsilon_s + epsilon_c);
    }
    else{
        *n_sc_tend = 0.0;
        *q_sc_tend = 0.0;
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
    double* q_sr_tend,
    double* q_rs_tend,
    double* n_sr_tend
){
    // snow riming raindrops: s+r -> s (and r+s -> g)
    // define local variables
    double E_sr; // collection efficience
    double epsilon_s, epsilon_r;
    double qr_crit = 1.0e-6; // q threshold of cloud droplet for snow cloud collection
    double Dm_r_crit = 1.0e-6; // D threshold of ice for snow ice collection

    if (qr > qr_crit && qs > qr_crit && Dm_r > Dm_r_crit){
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
        *q_sr_tend = pi/4.0 * E_sr * nr * qs * (delta_r_0*Dm_r*Dm_r + delta_rs_1*Dm_r*Dm_s + delta_s_1*Dm_s*Dm_s) * 
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
    double T,
    double qc,
    double nc,
    double Dm_c,
    double velocity_c,
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
    double* qc_tendency,
    double* nc_tendency,
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
    double q_sc_tend, n_sc_tend;
    double q_sr_tend, n_sr_tend;
    double q_rs_tend;

    // first do the riming core calculation for snow of cloud droplet and snow
    // same treatment like collection
    sb_snow_cloud_riming(nc, qc, Dm_c, velocity_c, ns, qs, Dm_s, velocity_s, &q_sc_tend, &n_sc_tend);
    sb_snow_rain_riming(nr, qr, Dm_r, velocity_r, ns, qs, Dm_s, velocity_s, &q_sr_tend, &n_sr_tend, &q_rs_tend);

    double q_rime_all = q_sc_tend + q_sr_tend;

    // Depositional growth is stronger than riming growth, therefore snow stays snow:
    if (qs_tend_dep > 0.0 && qs_tend_dep > q_rime_all){
        // snow cloud riming
        if (q_sc_tend > 0.0){ 
            double q_sc = fmin(qc, q_sc_tend*dt);
            double n_sc = fmin(nc, n_sc_tend*dt);

            *qs_tendency += q_sc/dt;
            *qc_tendency += -q_sc/dt;
            *nc_tendency += -n_sc/dt;

            // ice multiplication
            if(T < T_3){
                double q_mult, n_mult;
                ice_multiplication(T, q_sc, &q_mult, &n_mult);

                *ni_tendency += n_mult/dt;
                *qi_tendency += q_mult/dt;
                *qs_tendency += -q_mult;
            }
        }
        // snow rain riming
        if (q_sr_tend > 0.0){
            double q_sr = fmin(qr, q_sr_tend*dt);
            double n_sr = fmin(nr, n_sr_tend*dt);
            *qs_tendency += q_sr/dt;
            *qr_tendency += q_sr/dt;
            *nr_tendency += n_sr/dt;
            
            // ice multiplication
            if(T < T_3){
                double q_mult, n_mult;
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
        if (q_sc_tend > 0.0){ 
            double q_sc = fmin(qc, q_sc_tend*dt);
            double n_sc = fmin(nc, n_sc_tend*dt);

            *qs_tendency += q_sc/dt;
            *qc_tendency += -q_sc/dt;
            *nc_tendency += -n_sc/dt;

            // ice multiplication
            if(T < T_3){
                double q_mult, n_mult;
                ice_multiplication(T, q_sc, &q_mult, &n_mult);

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
            double q_rs = fmin(qs, q_rs_tend*dt);
            double q_sr = fmin(qr, q_sr_tend*dt);
            double n = fmin(fmin(nr, n_sr_tend*dt), ns);

            *ns_tendency += -n/dt;
            *nr_tendency += -n/dt;
            *qs_tendency += -q_rs/dt;
            *qr_tendency += -q_sr/dt;
            
            // ice multiplication
            double q_mult, n_mult;
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
                    // Here remain a space for riming to graupel
                }
                // snow + frozen liquid stays snow
                else{
                    *ns_tendency += n/dt;
                    *qs_tendency += q_sr/dt + q_rs/dt - q_mult/dt;
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
        double qt, // total water specific humidity
        double p0, // air pressure
        double qs, // snow specific humidity
        double ns, // snow number density
        double Dm_s, // mass-weighted diameter of snow
        double mass_s, // average mass of snow
        double velocity_s, // falling velocity of snow
        double dt, 
        // OUTPUT VARIABLES INDEX
        double* ns_tendency, 
        double* qs_tendency,
        double* qv_tendency
    ){

    // The diffusion section includes two main content:
    // - deposition if sat_ratio > 0.0 (over saturated);
    // - sublimation if sat_ratio < 0.0 (under saturated)

    double qs_tendency_dep = 0.0;
    double qs_tendency_sub = 0.0;
    double qs_tendency_diff =0.0;
    double qs_dep = 0.0;
    double ns_dep = 0.0;
    
    // calculate the sat_ratio 
    double pv_star = lookup(LT, T);
    double qv_star = qv_star_c(p0, qt, pv_star);
    double satratio = qt/qv_star - 1.0;
    
    // specific setting for snow
    double c_i = 2.0;

    if(qs > SB_EPS && ns > SB_EPS){

        // calculate g_therm factor during vapor diffusion
        double g_therm_ice = microphysics_g_sb_ice(LT, lam_fp, L_fp, T, DVAPOR, KT);
        double F_v_snow = 0.78 + 0.308*NSC_3*sqrt(Dm_s*velocity_s/KIN_VISC_AIR);

        if(satratio >= 0.0){
            double qs_tendency_dep = 4.0*pi/c_i * g_therm_ice * Dm_s * F_v_snow * satratio;
        }
        else{
            double qs_tendency_sub = 4.0*pi/c_i * g_therm_ice * Dm_s * F_v_snow * satratio;
        }
        qs_tendency_diff = qs_tendency_dep + qs_tendency_sub;

        qs_dep = fmax(qs_tendency_diff*dt, -qs);
        ns_dep = fmax(ns + fmin(qs_tendency_diff, 0.0)/mass_s/2.0, 0.0);

        *qs_tendency = qs_dep/dt;
        *ns_tendency = ns_dep/dt;
        *qv_tendency = -qs_dep/dt;
    }
    else{
        *qs_tendency = 0.0;
        *ns_tendency = 0.0;
        *qv_tendency = 0.0;
    }
    return;
}

void sb_melting_snow(
        // thermodynamic settings
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), 
        // INPUT VARIABLES
        double p0, // air pressure
        double T,
        double qt, // total water specific humidity
        double qv, // vapor specific humidity
        double qs, // snow specific humidity
        double ns, // snow number density
        double mass_s, // average mass of snow
        double Dm_s, // mass-weighted diameter of snow
        double velocity_s, // falling velocity of snow
        double dt,
        // OUTPUT VARIABLES INDEX
        double* ns_tendency, double* qs_tendency,
        double* nr_tendency, double* qr_tendency
    ){
    // define local varialbes
    double lam, L, pv_sat;
    double lam_3, L_3, pv_sat_3;
    double F_v_q, F_v_n, thermo_melt;
    double qs_melt_tend, ns_melt_tend;
    double qs_melt, ns_melt;

    if (qs > SB_EPS && ns > SB_EPS && T > T_3){

        lam = lam_fp(T);
        L = L_fp(T,lam);
        pv_sat = lookup(LT, T); // saturated vapor pressure at air temperature;

        lam_3 = lam_fp(T_3);
        L_3 = L_fp(T_3,lam);
        pv_sat_3 = lookup(LT, T_3); // saturated vapor pressure at T_3;

        F_v_q = 0.78 + 0.308*NSC_3*sqrt(Dm_s*velocity_s/KIN_VISC_AIR);
        F_v_n = 0.78 + 0.308*NSC_3*sqrt(Dm_s*velocity_s/KIN_VISC_AIR);
        // F_v_0 = sb_ventilation_coefficient(0.0, Dm_s, velocity_snow, snow_mass, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);
        // F_v_1 = sb_ventilation_coefficient(1.1, Dm_s, velocity_snow, snow_mass, SB_SNOW_B, SB_SNOW_beta, SB_SNOW_nu, SB_SNOW_mu);

        thermo_melt = (KT/DVAPOR)*(T - T_3) + DVAPOR*L/Rv*(pv_sat/T - pv_sat_3/T_3);

        qs_melt_tend = 2.0*pi/L_MELTING * thermo_melt * ns * Dm_s * F_v_q;
        ns_melt_tend = 2.0*pi/L_MELTING * thermo_melt * ns * Dm_s * F_v_n / mass_s;
        qs_melt = fmin(qs, fmin(qs_melt_tend * dt, 0.0));
        ns_melt = fmin(ns, fmin(ns_melt_tend * dt, 0.0));
        
        *qr_tendency = qs_melt/dt;
        *nr_tendency = ns_melt/dt;
        *qs_tendency = -qs_melt/dt;
        *ns_tendency = -ns_melt/dt;
    }
    else{
        *qs_tendency = 0.0;
        *ns_tendency = 0.0;
        *qr_tendency = 0.0;
        *nr_tendency = 0.0;
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
    
    // ---------Warm Process Tendency------------------
    double nr_tendency_tmp, qr_tendency_tmp, ql_tendency_tmp, nl_tendency_tmp;
    // autoconversion and accretion tendency
    double nr_tendency_au, qr_tendency_au, qr_tendency_ac;
    // selfcollection and breakup tendency
    double nr_tendency_scbk;
    // rain evaporation tendency
    double nr_tendency_evp, qr_tendency_evp;
    // -------------------------------------------------

    // ---------Ice Phase Process Tendency--------------
    double ns_tendency_tmp, qs_tendency_tmp, ni_tendency_tmp, qi_tendency_tmp;
    // collection tendency 
    // - ice self collection: i+i -> s
    double ni_tendency_ice_selcol, qi_tendency_ice_selcol;
    double ns_tendency_ice_selcol, qs_tendency_ice_selcol;
    // - snow self collection: s+s -> s
    double ns_tendency_snow_selcol;
    // - snow ice collection: s+i -> s
    double qs_tendency_si_col, ni_tendency_si_col, qi_tendency_si_col; 

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
                    
                    nr_tendency_tmp = 0.0; 
                    qr_tendency_tmp = 0.0; 
                    ql_tendency_tmp = 0.0; 
                    nl_tendency_tmp = 0.0;
                    nr_tendency_au = 0.0; 
                    qr_tendency_au = 0.0; 
                    qr_tendency_ac = 0.0;
                    nr_tendency_scbk = 0.0;
                    nr_tendency_evp = 0.0; 
                    qr_tendency_evp = 0.0;
                    ns_tendency_tmp = 0.0; 
                    qs_tendency_tmp = 0.0; 
                    ni_tendency_tmp = 0.0; 
                    qi_tendency_tmp = 0.0;
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
                    g_therm_rain = microphysics_g(LT, lam_fp, L_fp, temperature[ijk]);

                    //obtain some parameters of snow
                    snow_mass = microphysics_mean_mass(ns_tmp, qs_tmp, SB_SNOW_MIN_MASS, SB_SNOW_MAX_MASS);
                    Dm_s = SB_SNOW_A * pow(ice_mass, SB_SNOW_B);
                    velocity_snow = SB_SNOW_alpha * pow(snow_mass, SB_SNOW_beta) * sqrt(DENSITY_SB/density[k]);
                    
                    // -------------------- Main Content of Calculation --------------------------------------
                    //find the maximum substep time
                    dt_ = dt - time_added;

                    // compute the source terms of ice phase process: snow 
                    sb_diffusion_snow(LT, lam_fp, L_fp, temperature[ijk], qt_tmp, p0[k], 
                            Dm_s, snow_mass, velocity_snow, qs_tmp, ns_tmp, 
                            &ns_tendency_dep, &qs_tendency_dep, &qv_tendency_dep);

                    // ice phase collection processes
                    sb_ice_self_collection(temperature[ijk], qi_tmp, ni, Dm_i, velocity_ice, dt,
                            &qs_tendency_ice_selcol, &ns_tendency_ice_selcol,
                            &qi_tendency_ice_selcol, &ni_tendency_ice_selcol);
                    sb_snow_self_collection(temperature[ijk], qs_tmp, ns_tmp, Dm_s, velocity_snow, dt,
                            &ns_tendency_snow_selcol);
                    sb_snow_ice_collection(temperature[ijk], qi_tmp, ni, Dm_i, velocity_ice, 
                            qs_tmp, ns_tmp, Dm_s, velocity_snow, dt,
                            &qs_tendency_si_col, &qi_tendency_si_col, &ni_tendency_si_col);

                    // ice phase riming processes
                    sb_snow_riming(temperature[ijk], ql_tmp, nl, Dm_l, velocity_liquid, 
                            qr_tmp, nr_tmp, Dm_r, velocity_rain, rain_mass, 
                            qs_tmp, ns_tmp, Dm_s, velocity_snow, snow_mass, qs_tendency_dep,
                            &ql_tendency_snow_rime, &nl_tendency_snow_rime, 
                            &qi_tendency_snow_mult, &ni_tendency_snow_mult,
                            &qr_tendency_snow_rime, &nr_tendency_snow_rime,
                            &qs_tendency_rime, &ns_tendency_rime);

                    sb_melting_snow(LT, lam_fp, L_fp, p0[k], qt_tmp, qv_tmp, temperature[ijk],
                            qs_tmp, ns_tmp, snow_mass, Dm_s, velocity_snow,
                            &ns_tendency_melt, &qs_tendency_melt,
                            &nr_tendency_melt, &qr_tendency_melt);
                    
                    //compute the source terms of warm phase process: rain
                    sb_autoconversion_rain(droplet_nu, density[k], nl, ql_tmp, qr_tmp, 
                            &nr_tendency_au, &qr_tendency_au);
                    sb_accretion_rain(density[k], ql_tmp, qr_tmp, &qr_tendency_ac);
                    sb_selfcollection_breakup_rain(density[k], nr_tmp, qr_tmp, 
                            mu, rain_mass, Dm_r, &nr_tendency_scbk);
                    sb_evaporation_rain(g_therm_rain, sat_ratio, nr_tmp, qr_tmp, mu, 
                            rain_mass, Dp, Dm_r, &nr_tendency_evp, &qr_tendency_evp);

                    //check the source term magnitudes
                    // rain tendency sum
                    nr_tendency_tmp = nr_tendency_au + nr_tendency_scbk + nr_tendency_evp + 
                                      nr_tendency_snow_rime + nr_tendency_melt;
                    qr_tendency_tmp = qr_tendency_au + qr_tendency_ac + qr_tendency_evp + 
                                      qr_tendency_snow_rime + qr_tendency_melt;
                    
                    // snow tendency sum
                    ns_tendency_tmp = ns_tendency_ice_selcol + ns_tendency_snow_selcol + ns_tendency_rime + 
                                      ns_tendency_dep + ns_tendency_melt;
                    qs_tendency_tmp = qs_tendency_ice_selcol + qs_tendency_si_col + qs_tendency_rime + 
                                      qs_tendency_dep + qs_tendency_melt;
                    
                    // cloud droplet tendency sum
                    ql_tendency_tmp = -qr_tendency_au - qr_tendency_ac + ql_tendency_snow_rime;
                    nl_tendency_tmp = nl_tendency_snow_rime;
                    // ice particle tendency sum
                    qi_tendency_tmp = qi_tendency_ice_selcol + qi_tendency_si_col + qi_tendency_snow_mult;
                    ni_tendency_tmp = ni_tendency_ice_selcol + ni_tendency_si_col + ni_tendency_snow_mult;

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

                    qv_tmp += (qv_tendency_dep - qr_tendency_evp) * dt_;

                    qr_tmp = fmax(qr_tmp,0.0);
                    nr_tmp = fmax(fmin(nr_tmp, qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                    qs_tmp = fmax(qs_tmp,0.0);
                    ns_tmp = fmax(fmin(ns_tmp, qs_tmp/SB_SNOW_MIN_MASS),qs_tmp/SB_SNOW_MAX_MASS);
                    ql_tmp = fmax(ql_tmp,0.0);
                    nl = fmax(fmin(nl, ql_tmp/LIQUID_MIN_MASS),ql_tmp/LIQUID_MAX_MASS);
                    qi_tmp = fmax(qi_tmp,0.0);
                    ni = fmax(fmin(ni, qi_tmp/ICE_MIN_MASS),qi_tmp/ICE_MAX_MASS);
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
