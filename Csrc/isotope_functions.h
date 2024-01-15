#pragma once
#include "parameters.h"
#include "parameters_micro_sb.h"
#include "thermodynamic_functions.h"
#include "microphysics.h"
#include "microphysics_sb.h"
#include "microphysics_arctic_1m.h"
#include "lookup.h"
#include <math.h>
#define SB_EPS_iso 1.02e-13
// #define KT  2.5e-2 // J/m/1s/K
// #define DVAPOR 3.0e-5 // m^2/s

static inline double equilibrium_fractionation_factor_O18_liquid(double t){
// fractionation factor α_eq for 018 is based equations from Majoube 1971
// α_eq specificly is α_l/v 
	double alpha_lv = exp(1137.0/(t*t) - 0.4156/t - 2.0667e-3);  
    return alpha_lv;
}

static inline double equilibrium_fractionation_factor_HDO_liquid(double t){
// fractionation factor α_eq for HDO is based equations from Majoube 1971
// α_eq specificly is α_l/v 
	double alpha_lv = exp(24844.0/(t*t) - 76.248/t + 52.612e-3);
    return alpha_lv;
}

// ================ alpha_k in surface evaporation =================

static inline double kinetic_fractionation_factor_C_G_model(
        double eD, // diffusivity ratio of vapor D/D_i
        double ustar, // fraction velocity
        double zlevel // z height
    ){
    // kinetic fractionation during evaporation from ocean based on Merlivat and Jouzel 1979(JGR, v84, 5029-5033)
    double chi = 0.4;
    // calculate roughness height, air viscosity, and Reynolds number
    double z0 = (ustar*ustar)/(81.1*g);
    double Re = (ustar*z0)/KIN_VISC_AIR;
    double dvap = DVAPOR;
    
    // small Reynolds number, assume smooth ocean surface
    // double n = 2.0/3.0;
    // double rho_t_rho_M = ((1/chi)*log((ustar*zlevel)/(30*KIN_VISC_AIR)))/pow(13.6*(KIN_VISC_AIR/dvap), n);
    
    // large Reynolds number, assume rough ocean surface
    double n = 1.0/2.0;
    double rho_t_rho_M = (1/chi)*log(10.0/z0) - 5.0/(7.3*pow(Re, 0.25)*pow(KIN_VISC_AIR/dvap, n));
    // double rho_t_rho_M = 0.0;
    
    double alpha_k = (pow(eD,n) - 1.0)/(pow(eD,n) + rho_t_rho_M);
    return alpha_k;
}

// Rayleigh distillation is adopted from Wei's paper in 2018 for qt_iso initialization
static inline void Rayleigh_distillation(double qt, double* qt_O18, double* qt_HDO){
    double delta_O18 = 8.99 * log((qt*1000)/0.622) - 42.9;
    double delta_HDO = 8.0 * delta_O18 + 10.0;
    double R_O18 = (delta_O18/1000 + 1) * R_std_O18;
    double R_HDO = (delta_HDO/1000 + 1) * R_std_HDO;
    *qt_O18 = R_O18*qt;
    *qt_HDO = R_HDO*qt;
    return;
}

// Rayleigh distillation is adopted from Wei's paper in 2018 for qt_iso initialization
// only for   
static inline double Rayleigh_distillation_O18(double qt){
    double delta;
    double R;
    delta = 8.99 * log((qt*1000)/0.622) - 42.9;
    R = (delta/1000 + 1) * R_std_O18;
    return R*qt;
}

static inline double Rayleigh_distillation_HDO(double qt){
    double delta_O18 = 8.99 * log((qt*1000)/0.622) - 42.9;
    double delta_HDO = 8.0 * delta_O18 + 10.0;
    double R_HDO = (delta_HDO/1000 + 1) * R_std_HDO;
    return R_HDO*qt;
}

// calculate delta of specific water phase variable, values of isotopeic varialbe is after scaled.
static inline double q_2_delta(double const q_iso, double const q){
    return ((q_iso/q) - 1) * 1000;
}

static inline double q_2_R(double const q_iso, double const q){
    return q_iso/q;
}

// return the qv_tacer values
static inline double eq_frac_function(double const qt_tracer, double const qv_, double const ql_, double const alpha){
    return qt_tracer / (1.0+(ql_/qv_)*alpha);
}

static inline double global_meteoric_water(double delta_O18){
    return  8.0 * delta_O18 + 10.0;
}

static inline double C_G_model(
        double RH, 
        double alpha_eq, 
        double alpha_k,
        double R_Liquid,
        double R_air
    ){
    double R_sur_evap;
    double relative_humidity = fmin(RH, 1.0);
    R_sur_evap = alpha_k*(R_Liquid*alpha_eq - relative_humidity*R_air)/(1.0 - relative_humidity);
    return R_sur_evap;
}

static inline double epsilon_k(
        double theta,
        double RH,
        double n,
        double D_ratio // heavier over lighter
    ){
    double relative_humidity = fmin(RH, 1.0);
    double c_k = n*(1.0 - D_ratio)*1e3;
    double epsilon_k = theta*(1-relative_humidity)*c_k;
    return epsilon_k;
}

static inline double C_G_model_delta(
        double RH, 
        double alpha_eq, 
        double epsilon_k,
        double delta_Liquid,
        double delta_air
    ){
    double relative_humidity = fmin(RH, 1.0);
    double epsilon_star = (1-alpha_eq)*1e3;
    double delta_evap = (alpha_eq*delta_Liquid - relative_humidity*delta_air - (epsilon_star + epsilon_k)) / 
            ((1.0 - relative_humidity) + 1.0e-3*epsilon_k);
    return delta_evap;
}

// This section is adopted from Dar, 2020, equation 7
// Which is adopted from the method in Merlivat 1978
static inline double C_G_model_Merlivat(
        double RH, 
        double alpha_eq, 
        double alpha_k,
        double R_Liquid 
    ){
    double R_sur_evap;
    double relative_humidity = fmin(RH, 1.0);
    R_sur_evap = (alpha_eq*alpha_k*R_Liquid)/((1-relative_humidity)+alpha_k*relative_humidity);
    return R_sur_evap;
}

static inline double C_G_model_O18(double RH,  double temperature, double alpha_k){
    double alpha_eq;
    double R_sur_evap;
    // in case the relative humidity at surface is large than 1.0
    double relative_humidity = fmin(RH, 1.0);
    double R_O18_liquid = R_std_O18;
    alpha_eq = 1.0 / equilibrium_fractionation_factor_O18_liquid(temperature);
    R_sur_evap = (alpha_eq*alpha_k*R_O18_liquid)/((1-relative_humidity)+alpha_k*relative_humidity);
    return R_sur_evap;
    // return 2.0052e-3;
}

static inline double C_G_model_HDO_test(double R_O18){
    double delta_O18 = (R_O18/R_std_O18 - 1)*1000.0;
    double delta_HDO = global_meteoric_water(delta_O18);
    return (delta_HDO/1000.0 + 1) * R_std_HDO;
}

static inline double C_G_model_HDO(double RH,  double temperature, double alpha_k){
    double alpha_eq;
    double R_sur_evap;
    // in case the relative humidity at surface is large than 1.0
    double relative_humidity = fmin(RH, 1.0);
    double delta_HDO_liquid = global_meteoric_water(0.0);
    double R_HDO_liquid = (delta_HDO_liquid/1000 + 1) * R_std_HDO;
    alpha_eq = 1.0 / equilibrium_fractionation_factor_HDO_liquid(temperature);

    R_sur_evap = (alpha_eq*alpha_k*R_HDO_liquid)/((1-relative_humidity)+alpha_k*relative_humidity);
    return R_sur_evap;
}

// ======== C_G_model_Dar_2020 ======= 
// This section is adopted from Dar, 2020, equation 18
static inline double C_G_model_Dar(
        double RH, 
        double alpha_eq,
        double Diffusivity_Iso,
        double R_Liquid,
        double x,
        double gamma
        ){
    // TODO: need to double check the Diffusivity_Iso
    double R_sur_evap;
    // the turbulence index of atmosphere, 1.0 means no turbulence, 0.0 means no fractionation
    // double x = 0.6; 
    double relative_humidity = fmin(RH, 1.0);
    R_sur_evap = (R_Liquid*gamma) / (alpha_eq*(Diffusivity_Iso*(gamma - relative_humidity) + relative_humidity));
    return R_sur_evap;
}

// This section is adopted from Dar, 2020, equation 18
static inline double C_G_model_Dar_2020_O18(
        double RH, 
        double temperature, 
        double Diffusivity_Iso
        ){
    // TODO: need to double check the Diffusivity_Iso
    double R_sur_evap;
    double x = 0.6; // the turbulence index of atmosphere, 1.0 means no turbulence, 0.0 means no fractionation
    double relative_humidity = fmin(RH, 1.0);
    double alpha_eq = equilibrium_fractionation_factor_O18_liquid(temperature);
    R_sur_evap = (R_std_O18*1.0)/(alpha_eq*(Diffusivity_Iso*(1.0 - RH) + RH));
    return R_sur_evap;
}

static inline double C_G_model_Dar_2020_HDO(
        double RH, 
        double temperature, 
        double Diffusivity_Iso
        ){
    // TODO: need to double check the Diffusivity_Iso
    double R_sur_evap;
    double x = 0.6; // the turbulence index of atmosphere, 1.0 means no turbulence, 0.0 means no fractionation
    double relative_humidity = fmin(RH, 1.0);
    double delta_HDO_liquid = global_meteoric_water(0.0);
    double R_HDO_liquid = (delta_HDO_liquid/1000 + 1) * R_std_HDO;
    double alpha_eq = equilibrium_fractionation_factor_HDO_liquid(temperature);
    R_sur_evap = (R_HDO_liquid*1.0)/(alpha_eq*(Diffusivity_Iso*(1.0 - RH) + RH));
    return R_sur_evap;
}
// reference of fresh water fresh water isotope ratio in Lake Tai Hu, see
// Estimating evaporation over a large and shallow lake using stable isotopic method: 
// A case study of Lake Taihu, Xiao's 2017
// reference values are in Fig.5 
static inline double C_G_model_O18_Mpace_ST(double RH,  double temperature, double alpha_k){
    double alpha_eq;
    double R_sur_evap;
    // in case the relative humidity at surface is large than 1.0
    double relative_humidity = fmin(RH, 1.0);
    double R_surfacer_water = 1.9951e-3; // delta_o18 = -5‰
    alpha_eq = 1.0 / equilibrium_fractionation_factor_O18_liquid(temperature);
    R_sur_evap = alpha_eq*alpha_k*R_surfacer_water/((1-relative_humidity)+alpha_k*relative_humidity);
    return R_sur_evap;
}

static inline double C_G_model_HDO_Mpace_ST(double RH,  double temperature, double alpha_k){
    double alpha_eq;
    double R_sur_evap;
    // in case the relative humidity at surface is large than 1.0
    double relative_humidity = fmin(RH, 1.0);
    double R_surfacer_water = 1.52831e-3; // delta_hdo = -20‰
    alpha_eq = 1.0 / equilibrium_fractionation_factor_HDO_liquid(temperature);
    R_sur_evap = alpha_eq*alpha_k*R_surfacer_water/((1-relative_humidity)+alpha_k*relative_humidity);
    return R_sur_evap;
}

static inline double iso_vapor_diffusivity(const double temperature, const double p0, const int index){
    /*Straka 2009 (5.2)*/
    double vapor_diff; 
    double vapor_diff_tmp = 2.11e-5 * pow((temperature / 273.15), 1.94) * (p0 / 101325.0);
    switch(index){
        case 2:
            // Merlivat (1970)
            vapor_diff = vapor_diff_tmp*0.9723;
        case 3:
            // Merlivat (1970)
            vapor_diff = vapor_diff_tmp*0.9755;
    }
    return vapor_diff;
};

// ===========<<<  SB_Warm Scheme  >>> ============

double microphysics_g_iso_SB_Liquid(struct LookupStruct *LT, 
        double (*lam_fp)(double), 
        double (*L_fp)(double, double),
        double iso_type,
        double temperature, 
        double p0, 
        double qr, 
        double qr_iso, 
        double qv, 
        double qv_iso, 
        double sat_ratio,
        double dvap, 
        double kt){

    double lam          = lam_fp(temperature);
    double L            = L_fp(temperature,lam);
    double pv_sat       = lookup(LT, temperature);
    double rho_sat      = pv_sat/Rv/temperature;

    // blossey's scheme for isotopic fractionation
    // double b_l       = (DVAPOR*L*L*rho_sat)/KT/Rv/(temperature*temperature); 

    // own scheme for isotopic fractionation, based on SB_Liquid evaporation scheme
    double b_l          = (dvap*rho_sat)*(L/kt/temperature)*(L/Rv/temperature - 1.0); 

    double S_l          = sat_ratio + 1.0;
    double R_qv_ambient = qv_iso / qv;
   
    double R_qr = qr_iso/qr;
    double alpha_eq;
    if (iso_type == 1.0){
        alpha_eq = equilibrium_fractionation_factor_O18_liquid(temperature);
    }
    else if(iso_type == 2.0){
        alpha_eq = equilibrium_fractionation_factor_HDO_liquid(temperature);
    }
    double R_qr_surface = R_qr / alpha_eq;
    
    // double g_therm_iso = dvap*rho_sat*R_qv_ambient*((rat)*(1.0 + b_l*S_l)/(1+b_l) - S_l);
    double g_therm_iso = dvap*rho_sat*R_qv_ambient*((R_qr_surface/R_qv_ambient)*(1.0 + b_l*S_l)/(1+b_l) - S_l);
    return g_therm_iso;
};

double microphysics_g_iso_Arc1M(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
         double temperature, double p0, double qr, double qr_iso, double qv, double qv_iso, double sat_ratio, double alpha_eq,
         double dvap, double kt){

    double lam      = lam_fp(temperature);
    double L        = L_fp(temperature,lam);
    double pv_sat   = lookup(LT, temperature);
    double rho_sat  = pv_sat/Rv/temperature;

    // double b_l       = (DVAPOR*L*L*rho_sat)/KT/Rv/(temperature*temperature); // blossey's scheme for isotopic fractionation
    double b_l          = (dvap*rho_sat)*(L/kt/temperature)*(L/Rv/temperature - 1.0); // own scheme for isotopic fractionation, based on SB_Liquid evaporation scheme
    
    double R_qr         = qr_iso / qr;
    double R_qv_ambient = qv_iso / qv;
    double R_qr_surface = R_qr / alpha_eq;
    // double rat = R_qr_surface/R_qv_ambient;
    
    double S_ratio = sat_ratio + 1.0; // INPUT sat_ratio is POSITIVE or NEGETIVE
    
    // double g_therm_iso = D_O18*rho_sat*R_qv_ambient*((rat)*(1.0 + b_l*S_l)/(1+b_l) - S_l);
    double g_therm_iso = dvap*rho_sat*R_qv_ambient*((R_qr_surface/R_qv_ambient)*(1.0 + b_l*S_ratio)/(1+b_l) - S_ratio);
    return g_therm_iso;
};

// ================================================
// ToDo: microphysics isotope thermodynamic variable should 
// been applied without LT or lam_fp settings in next step
// ================================================
double microphysics_g_iso_rain_SBSI(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
         double temperature, double p0, double qr, double qr_iso, double qv, double qv_iso, double sat_ratio,
         double dvap, double kt){
    double lam          = lam_fp(temperature);
    double L            = L_fp(temperature,lam);
    double pv_sat       = saturation_vapor_pressure_water(temperature);
    double rho_sat      = pv_sat/Rv/temperature;
    // double b_l       = (DVAPOR*L*L*rho_sat)/KT/Rv/(temperature*temperature); // blossey's scheme for isotopic fractionation
    double b_l          = (dvap*rho_sat)*(L/kt/temperature)*(L/Rv/temperature - 1.0); // own scheme for isotopic fractionation, based on SB_Liquid evaporation scheme

    double S_l          = sat_ratio + 1.0;
    
    double R_qr         = qr_iso / qr;
    double R_qv_ambient = qv_iso / qv;
    double alpha_eq     = equilibrium_fractionation_factor_O18_liquid(temperature);
    double R_qr_surface = R_qr / alpha_eq;
    // double rat = R_qr_surface/R_qv_ambient;
    
    // double g_therm_iso = D_O18*rho_sat*R_qv_ambient*((rat)*(1.0 + b_l*S_l)/(1+b_l) - S_l);
    double g_therm_iso = dvap*rho_sat*R_qv_ambient*((R_qr_surface/R_qv_ambient)*(1.0 + b_l*S_l)/(1+b_l) - S_l);
    return g_therm_iso;
};

double microphysics_g_iso_ice_SBSI(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
         double temperature, double p0, double qr, double qr_iso, double qv, double qv_iso, double sat_ratio,
         double dvap, double kt){

    double lam          = lam_fp(temperature);
    double L            = L_fp(temperature,lam);
    double pv_sat       = saturation_vapor_pressure_ice(temperature);
    double rho_sat      = pv_sat/Rv/temperature;

    // double b_l       = (DVAPOR*L*L*rho_sat)/KT/Rv/(temperature*temperature); // blossey's scheme for isotopic fractionation
    double b_l          = (dvap*rho_sat)*(L/kt/temperature)*(L/Rv/temperature - 1.0); // own scheme for isotopic fractionation, based on SB_Liquid evaporation scheme

    double S_l          = sat_ratio + 1.0;
    
    double R_qr         = qr_iso / qr;
    double R_qv_ambient = qv_iso / qv;
    double alpha_eq     = equilibrium_fractionation_factor_O18_liquid(temperature);
    double R_qr_surface = R_qr / alpha_eq;
    // double rat = R_qr_surface/R_qv_ambient;
    
    // double g_therm_iso = D_O18*rho_sat*R_qv_ambient*((rat)*(1.0 + b_l*S_l)/(1+b_l) - S_l);
    double g_therm_iso = dvap*rho_sat*R_qv_ambient*((R_qr_surface/R_qv_ambient)*(1.0 + b_l*S_l)/(1+b_l) - S_l);
    return g_therm_iso;
};

// double microphysics_g_iso(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
//                              double temperature, double p0, double qr, double qr_iso, double qv, double qv_iso, double sat_ratio,
//                              double dvap, double kt){
//     double lam          = lam_fp(temperature);
//     double L            = L_fp(temperature,lam);
//     // double pv_sat       = lookup(LT, temperature);
//     double pv_sat       = saturation_vapor_pressure_water(temperature);
//     double rho_sat      = pv_sat/Rv/temperature;
//     // double b_l       = (DVAPOR*L*L*rho_sat)/KT/Rv/(temperature*temperature); // blossey's scheme for isotopic fractionation
//     double b_l          = (dvap*rho_sat)*(L/kt/temperature)*(L/Rv/temperature - 1.0); // own scheme for isotopic fractionation, based on SB_Liquid evaporation scheme
//
//     double S_l          = sat_ratio + 1.0;
//     double D_O18        = dvap*0.9723;
//     
//     double R_qr         = qr_iso / qr;
//     double R_qv_ambient = qv_iso / qv;
//     double alpha_eq     = equilibrium_fractionation_factor_O18_liquid(temperature);
//     double R_qr_surface = R_qr / alpha_eq;
//     // double rat = R_qr_surface/R_qv_ambient;
//     
//     // double g_therm_iso = D_O18*rho_sat*R_qv_ambient*((rat)*(1.0 + b_l*S_l)/(1+b_l) - S_l);
//     double g_therm_iso = D_O18*rho_sat*R_qv_ambient*((R_qr_surface/R_qv_ambient)*(1.0 + b_l*S_l)/(1+b_l) - S_l);
//     return g_therm_iso;
// };

static inline void sb_iso_rain_autoconversion(
        double ql, 
        double ql_iso, 
        double qr_auto_tendency, 
        double* qr_iso_auto_tendency
    ){
    if (ql > 0.0 && ql_iso > SB_EPS){
        *qr_iso_auto_tendency = qr_auto_tendency * (ql_iso/ql);
    }
    else{
        *qr_iso_auto_tendency = 0.0;
    }
    return;
}

static inline void sb_iso_rain_accretion(
        double ql, 
        double ql_iso, 
        double qr_accre_tendency, 
        double* qr_iso_accre_tendency
    ){
    if (ql > 0.0 && ql_iso > SB_EPS){
        *qr_iso_accre_tendency = qr_accre_tendency * (ql_iso/ql);
    }
    else{
        *qr_iso_accre_tendency = 0.0;
    }
    return;
}

static inline void sb_iso_rain_evap_nofrac(
        double qr, 
        double qr_iso, 
        double qr_evap_tendency, 
        double* qr_iso_evap_tendency
    ){
    if (qr > SB_EPS && qr_iso > SB_EPS){
        *qr_iso_evap_tendency = qr_evap_tendency * (qr_iso/qr);
    }
    else{
        *qr_iso_evap_tendency = 0.0;
    }
    return;
}

void sb_iso_evaporation_rain(double g_therm_iso, 
        double sat_ratio, 
        double nr, 
        double qr, 
        double mu, 
        double qr_iso, 
        double rain_mass, 
        double Dp,
        double Dm, 
        double* qr_iso_tendency
    ){
    double gamma, dpfv, phi_v;
    const double bova      = B_RAIN_SED/A_RAIN_SED;
    const double cdp       = C_RAIN_SED * Dp;
    const double mupow     = mu + 2.5;
    double qr_tendency_tmp = 0.0;
    if(qr < SB_EPS || nr < SB_EPS || qr_iso < SB_EPS){
        *qr_iso_tendency = 0.0;
    }
    else if(sat_ratio >= 0.0){
        *qr_iso_tendency = 0.0;
    }
    else{
        gamma = 0.7;
        // gamma = 0.7 is used by DALES ; alternative expression gamma= d_eq/Dm
        // * exp(-0.2*mudouble vapor__diff, vapor_HDO16_diff; 2double
        // vapor__diff, vapor_HDO16_diff;) is used by S08;
        phi_v = 1.0 - (0.5  * bova * pow(1.0 +  cdp, -mupow) + 0.125 * bova * bova * pow(1.0 + 2.0*cdp, -mupow)
                      + 0.0625 * bova * bova * bova * pow(1.0 +3.0*cdp, -mupow) + 0.0390625 * bova * bova * bova * bova * pow(1.0 + 4.0*cdp, -mupow));
        dpfv = (A_VENT_RAIN * tgamma(mu + 2.0) * Dp + B_VENT_RAIN * NSC_3 * A_NU_SQ * tgamma(mupow) * pow(Dp, 1.5) * phi_v)/tgamma(mu + 1.0);

        qr_tendency_tmp  = 2.0 * pi * g_therm_iso * nr * dpfv;
        *qr_iso_tendency = -qr_tendency_tmp;
    }
    return;
}

// ===========<<< iso 1-m ice scheme >>> ============

// ============= Default equilibrium_fractionation_factor of O18 and HDO =============
static inline double equilibrium_fractionation_factor_O18_ice(double t){
// fractionation factor α_eq for 018 for vapor between ice, based equations from Majoube 1970
	double alpha_ice = exp(11.839/t - 2.8224e-2);  
    return alpha_ice; // alpha_ice > 1.0
}

static inline double equilibrium_fractionation_factor_HDO_ice(double t){
// fractionation factor α_eq for HDO for vapor between ice, based equations from Merlivat 1967
	double alpha_ice = exp(16289/(t*t) - 9.45e-2);  
    return alpha_ice;
}

// ====== Default equilibrium_fractionation_factor of O18 and HDO =======
static inline double equilibrium_fractionation_factor_O18_ice_Ellehoj(double t){
// fractionation factor α_eq for 018 for vapor between ice, based equations from Elleboj 2013
	double alpha_ice_O18 = exp(0.0831 - 49.192/t + 8312.5/(t*t));  
    return alpha_ice_O18;
}
static inline double equilibrium_fractionation_factor_HDO_ice_Ellehoj(double t){
// fractionation factor α_eq for HDO for vapor between ice, based equations from Elleboj 2013
	double alpha_ice_O18 = exp(0.2133 - 203.10/t + 48888.0/(t*t));  
    return alpha_ice_O18;
}


static inline double equilibrium_fractionation_factor_HDO_ice_Lamb(double t){
// fractionation factor α_eq for HDO for vapor between ice, based equations from Lamb 2017
	double alpha_ice = exp(13525/(t*t) - 5.59e-2);  
    return alpha_ice;
}

// ===========<<< alpha_k of ice fractionation based on Blossey'10 scheme >>> ============
    
    //-------------------------------------------------------------
    // INPUT VARIABLES
    //-------------------------------------------------------------
    // alpha_s: equilibrium fractionation factor for ice
    // diff_vapor: diffusivity of common water vapor
    // diff_iso: diffusivity of isotope water vapor (O18 or HDO)
    //-------------------------------------------------------------
    // OUTPUT VARIABLES
    //-------------------------------------------------------------
    // alpha_k_ice: kinetic fractionation factor for ice
    //------------------------------------------------------------- 
    // this function is adopted from Blossey's 2015, for calculate of alpha_k

double alpha_k_ice_equation_Blossey_Arc1M(
        struct LookupStruct *LT, 
        double (*lam_fp)(double), 
        double (*L_fp)(double, double),
        double temperature, 
        double p0, 
        double qt, 
        double alpha_s, 
        double diff_vapor, 
        double diff_iso){
    // this only works in Arc1M scheme as pv_sat is calculated based on PyCLES default lookup table

    double lam            = lam_fp(temperature);
    double L              = L_fp(temperature,lam);
    double pv_sat_ice     = lookup(LT, temperature);
    double rho_sat_ice    = pv_sat_ice/Rv/temperature;
    // calculate sat_ratio of vapor respect to ice, S_s is the same simple in Blossey's 2015
    double qv_sat_ice     = qv_star_c(p0,qt,pv_sat_ice);
    double S_s            = qt/qv_sat_ice;
    double diff_ratio     = diff_vapor / diff_iso;
    double b_s            = (diff_vapor*rho_sat_ice)*(L/KT/temperature)*(L/Rv/temperature - 1.0);
    double alpha_k_ice    = (1 + b_s) * S_s * (1/(alpha_s*diff_ratio*(S_s - 1) + 1 + b_s*S_s));
    return alpha_k_ice;
}

double alpha_k_ice_equation_Blossey(
        struct LookupStruct *LT, 
        double (*lam_fp)(double), 
        double (*L_fp)(double, double),
        double temperature, 
        double p0, 
        double qt, 
        double alpha_s, 
        double diff_vapor, 
        double diff_iso){
    // this only works in Arc1M scheme as pv_sat is calculated based on PyCLES default lookup table

    double lam            = lam_fp(temperature);
    double L              = L_fp(temperature,lam);
    double pv_sat_ice     = saturation_vapor_pressure_ice(temperature);
    double rho_sat_ice    = pv_sat_ice/Rv/temperature;
    // calculate sat_ratio of vapor respect to ice, S_s is the same simple in Blossey's 2015
    double qv_sat_ice     = qv_star_c(p0,qt,pv_sat_ice);
    double S_s            = qt/qv_sat_ice;
    double diff_ratio     = diff_vapor / diff_iso;
    double b_s            = (diff_vapor*rho_sat_ice)*(L/KT/temperature)*(L/Rv/temperature - 1.0);
    double alpha_k_ice    = (1 + b_s) * S_s * (1/(alpha_s*diff_ratio*(S_s - 1) + 1 + b_s*S_s));
    return alpha_k_ice;
}

double alpha_k_ice_equation_Jouzel(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
        double temperature, double p0, double qt, double alpha_s, double diff_vapor, double diff_iso){
    //-------------------------------------------------------------
    // INPUT VARIABLES
    //-------------------------------------------------------------
    // alpha_s: equilibrium fractionation factor for ice
    // diff_vapor: diffusivity of common water vapor
    // diff_iso: diffusivity of isotope water vapor (O18 or HDO)
    //-------------------------------------------------------------
    // OUTPUT VARIABLES
    //-------------------------------------------------------------
    // alpha_k_ice: kinetic fractionation factor for ice
    //------------------------------------------------------------- 
    // this function is adopted from Jouzel's 1984, for calculate of alpha_k
    double lam            = lam_fp(temperature);
    double L              = L_fp(temperature,lam);
    double pv_sat_ice     = saturation_vapor_pressure_water(temperature);
    double rho_sat_ice    = pv_sat_ice/Rv/temperature;
    // calculate sat_ratio of vapor respect to ice, S_s means the sat_ratio
    double qv_sat_ice     = qv_star_c(p0,qt,pv_sat_ice);
    double S_s            = qt/qv_sat_ice;
    double diff_ratio     = diff_vapor / diff_iso;
    double alpha_k_ice    = S_s/(alpha_s*diff_ratio*(S_s-1.0)+1.0);
    return alpha_k_ice;
}

// ===========<<< Arc1M scheme isotpoe tracer processes >>> ============

void arc1m_iso_autoconversion_rain(double qrain_tendency_aut, double ql, double ql_iso, double* qrain_iso_tendency_auto){
    double R_ql;
    if(ql > SMALL && ql_iso > SMALL){
        R_ql = ql_iso/ql;
        *qrain_iso_tendency_auto = qrain_tendency_aut * R_ql;
    }
    else{
        *qrain_iso_tendency_auto = 0.0;
    }
    return;
}

void arc1m_iso_evap_rain_nofrac(double qrain_tendency_evap, double R_qrain, double* qrain_iso_tendency_evap){
    *qrain_iso_tendency_evap = qrain_tendency_evap * R_qrain;
    return;
}

void arc1m_iso_autoconversion_snow(double qsnow_tendency_aut, double qi, double qi_iso, double* qsnow_iso_tendency_auto){
    double R_qi;
    if(qi > SMALL && qi_iso > 1.0e-15){
        R_qi = qi_iso/qi;
        *qsnow_iso_tendency_auto = qsnow_tendency_aut * R_qi;
    }
    else{
        *qsnow_iso_tendency_auto = 0.0;
    }
    return;
}

void arc1m_iso_evap_snow_nofrac(double qsnow_tendency_evap, double R_qsnow, double* qsnow_iso_tendency_evap){
    *qsnow_iso_tendency_evap = qsnow_tendency_evap * R_qsnow;
    return;
}

void arc1m_iso_melt_snow(double qsnow_tendency_melt, double qsnow, double qsnow_iso, double* qsnow_iso_tendency_melt){
    double R_qsnow;
    if(qsnow > SMALL && qsnow_iso > 1.0e-15){
        R_qsnow = qsnow_iso/qsnow;
        *qsnow_iso_tendency_melt = qsnow_tendency_melt * R_qsnow;
    }
    else{
        *qsnow_iso_tendency_melt = 0.0;
    }
    return;
}

void arc1m_iso_accretion_all(double density, double p0, double temperature, double ccn, double ql, double qi, double ni,
    double qrain, double nrain, double qsnow, double nsnow,
    double ql_iso, double qi_iso, double qrain_iso, double qsnow_iso,
    double* ql_iso_tendency, double* qi_iso_tendency, double* qrain_iso_tendency, double* qsnow_iso_tendency){
    // ===========<<< micro-source calculation during accretion>>> ============
    double factor_r  = 0.0;
    double factor_s  = 0.0;
    double piacr     = 0.0;

    double e_ri      = 1.0;
    double e_si      = exp(9.0e-2*(temperature - 273.15));
    double e_rl      = 0.85;
    double e_sl      = 0.8;

    double liq_diam  = liquid_dmean(density, ql, ccn);
    double snow_diam = snow_dmean(density, qsnow, nsnow);
    double rain_diam = rain_dmean(density, qrain, nrain);
    double rain_lam  = rain_lambda(density, qrain, nrain);
    double snow_lam  = snow_lambda(density, qsnow, nsnow);
    double ice_lam   = ice_lambda(density, qi, ni);
    double rain_vel  = C_RAIN*pow(rain_diam, D_RAIN);
    double snow_vel  = C_SNOW*pow(snow_diam, D_SNOW);

    if( snow_diam < 150.0e-6 ){
        e_sl = 0.0;
    }
    else{
        if( liq_diam < 15.0e-6 ){
            e_sl = 0.0;
        }
        else if( liq_diam < 40.0e-6 ){
            e_sl = (liq_diam - 15.0e-6) * e_sl / 25.0e-6;
        }
    }

    if( qrain > SMALL ){
        factor_r = density*GD3_RAIN*nrain*pi*ALPHA_ACC_RAIN*C_RAIN*0.25*pow(rain_lam, (-D_RAIN - 3.0));
        if( qi > SMALL ){
            piacr = nrain*ni/ice_lam*e_ri*pi*0.25*A_RAIN*C_RAIN*GD6_RAIN*pow(rain_lam, (-D_RAIN - 6.0));
        }
    }

    if( qsnow > SMALL ){
        factor_s = density*GD3_SNOW*nsnow*pi*ALPHA_ACC_SNOW*C_SNOW*0.25*pow(snow_lam, (-D_SNOW-3.0));
    }

    double src_ri  = -piacr/density;
    double src_rl  = factor_r*e_rl*ql/density;
    double src_si  = factor_s*e_si*qi/density;
    double rime_sl = factor_s*e_sl*ql/density;
    double src_sl  = 0.0;

    if( temperature > 273.16 ){
        src_sl = -cvl/lf0*(temperature-273.16)*rime_sl;
        src_rl = src_rl + rime_sl - src_sl;
    }
    else{
        src_sl = rime_sl + (factor_r*e_ri*qi + piacr)/density;
    }

    /* Now precip-precip interactions */
    double src_r = 0.0;
    double src_s = 0.0;
    double dv, k_2s, k_2r;

    if( qrain > small && qsnow > small ){
        dv   = fabs(rain_vel - snow_vel);
        k_2s = (30.0/(pow(rain_lam, 6.0)*snow_lam) + 12.0/(pow(rain_lam, 5.0)*pow(snow_lam, 2.0))
                + 3.0/(pow(rain_lam, 4.0)*pow(snow_lam, 3.0)));
        k_2r = (1.0/(pow(rain_lam, 3.0)*pow(snow_lam, 3.0)) + 3.0/(pow(rain_lam, 2.0)*pow(snow_lam, 4.0))
                + 6.0/(rain_lam*pow(snow_lam, 5.0)));
        if( temperature < 273.16 ){
            src_s = pi*dv*nsnow*nrain*A_RAIN*k_2s/density;
            src_r = -src_s;
        }
        else{
            src_r = pi*dv*nsnow*nrain*A_SNOW*k_2r/density;
            src_s = -src_r;
        }
    }

    // *qrain_tendency = src_r + src_rl + src_ri;
    // *qsnow_tendency = src_s + src_sl + src_si;
    // *ql_tendency    = -(src_rl + rime_sl);
    // *qi_tendency    = -(src_ri + src_si);
    
    // ===========<<< Iso source calculations >>> ============
    double R_ql, R_qi, R_qrain, R_qsnow;
    if(ql > SMALL && ql_iso > SMALL){
        R_ql = ql_iso/ql;
    }
    else{
        R_ql = 0.0;
    }
    if(qi > SMALL && qi_iso > SMALL){
        R_qi = qi_iso/qi;
    }
    else{
        R_qi = 0.0;
    }

    if(qrain > 1.0e-15 && qrain_iso > 1.0e-15){
        R_qrain = qrain_iso/qrain;
    }
    else{
        R_qrain = 0.0;
    }
    
    if(qsnow > 1.0e-15 && qsnow_iso > 1.0e-15){
        R_qsnow = qsnow_iso/qsnow;
    }
    else{
        R_qsnow = 0.0;
    }

    double src_rl_iso  = src_rl * R_ql;
    double src_ri_iso  = src_ri * R_qi;
    double src_sl_iso  = src_sl * R_ql;
    double src_si_iso  = src_si * R_qi;
    double rime_sl_iso = rime_sl *R_ql;
    double src_s_iso, src_r_iso;
    if (temperature < 273.16){
        src_s_iso = src_s * R_qrain;
        src_r_iso = -src_s_iso;
    }
    else{
        src_r_iso = src_r * R_qsnow;
        src_s_iso = -src_s_iso;
    }

    *qrain_iso_tendency = src_r_iso + src_rl_iso + src_ri_iso;
    *qsnow_iso_tendency = src_s_iso + src_sl_iso + src_si_iso;
    *ql_iso_tendency    = -(src_rl_iso + rime_sl_iso);
    *qi_iso_tendency    = -(src_ri_iso + src_si_iso);

    return;
}

void arc1m_iso_evap_rain(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                   double density, const double p0, double temperature, double sat_ratio,
                   double qt, double qv, double qrain, double nrain, double gtherm_iso,
                   double qv_iso, double qrain_iso, double *qrain_iso_tendency){
    double beta       = 2.0;
    double vapor_diff = vapor_diffusivity(temperature, p0);
    double therm_cond = thermal_conductivity(temperature);
    double rain_diam  = rain_dmean(density, qrain, nrain);
    double rain_vel   = C_RAIN*pow(rain_diam, D_RAIN);
    double rain_lam   = rain_lambda(density, qrain, nrain);

    double re, vent, gther_iso; 
    if( sat_ratio < 0.0 && qrain > 1.0e-15 && qrain_iso > 1.0e-15){
        re = rain_diam*rain_vel/VISC_AIR;
        vent = 0.78 + 0.27*sqrt(re);
        double qrain_evap_tend_tmp = 4.0*pi/beta*vent*gther_iso*nrain/(rain_lam*rain_lam)/density;
        *qrain_iso_tendency = -qrain_evap_tend_tmp;
    }
    return;
}

void arc1m_iso_evap_snow_kinetic(double qsnow_tendency_evap, 
        double qsnow, 
        double qsnow_iso,
        double qv,
        double qv_iso,
        double alpha_s,
        double alpha_k,
        double* qsnow_iso_tendency_evap
        ){
    // depostion happened with kinetic fractionation
    if (qsnow_tendency_evap >= 0.0){
        *qsnow_iso_tendency_evap = qsnow_tendency_evap * alpha_s * alpha_k * (qv_iso/qv);
    }
    // sublimation happened without kinetic fractionation
    else{
        *qsnow_iso_tendency_evap = qsnow_tendency_evap * (qsnow_iso/qsnow);
    }
    return;
}
void arc1m_iso_evap_snow(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
        double density, const double p0, double temperature,
        double qt, double qv, double qsnow, double nsnow, double gtherm_iso,
        double qv_iso, double qsnow_iso, double* qsnow_iso_tendency){

    double beta = 3.0;

    double vapor_diff = vapor_diffusivity(temperature, p0);
    double therm_cond = thermal_conductivity(temperature);
    double snow_diam = snow_dmean(density, qsnow, nsnow);
    double snow_vel = C_SNOW*pow(snow_diam, D_SNOW);
    double snow_lam = snow_lambda(density, qsnow, nsnow);

    if( qsnow > 1.0e-15 ){
        double re = snow_diam*snow_vel/VISC_AIR;
        double vent = 0.65 + 0.39*sqrt(re);
        double qsnow_iso_tend_tmp = 4.0*pi/beta*vent*gtherm_iso*nsnow/(snow_lam*snow_lam)/density;
        *qsnow_iso_tendency = -qsnow_iso_tend_tmp;
    }
    return;
}

// ============= SB 2M coupled isotope scheme ================
void iso_sb_2m_cloud_liquid_fraction(
        const double type,
        const double T,
        const double qv,
        const double ql,
        const double qvl_iso,
        const double qv_iso,
        const double ql_iso,
        double* qv_iso_eq,
        double* ql_iso_eq
    ){

    double alpha_eq_lv, qv_iso_tmp, ql_iso_tmp;

    if(type == 1.0){
        alpha_eq_lv = equilibrium_fractionation_factor_O18_liquid(T);
    }
    else if(type == 2.0){
        alpha_eq_lv = equilibrium_fractionation_factor_HDO_liquid(T);
    }
    
    if(ql > SB_EPS){
        *qv_iso_eq = eq_frac_function(qvl_iso, qv, ql, alpha_eq_lv);
        *ql_iso_eq = qvl_iso - *qv_iso_eq;
    }
    else{
        *qv_iso_eq = qvl_iso;
        *ql_iso_eq = 0.0;
    }
    return;
}

void iso_sb_2m_cloud_ice_fraction(
        const double type,
        const double T,
        const double qi_tend_nuc,
        const double qv,
        const double qv_iso,
        double* qi_iso_tend
    ){
    // TODO: consider the αₖ
    double alpha_s, qi_iso_tmp;

    if(type == 1.0){
        alpha_s = equilibrium_fractionation_factor_O18_ice(T);
    }
    else if(type == 2.0){
        alpha_s = equilibrium_fractionation_factor_HDO_ice(T);
    }
    
    if(qi_tend_nuc > 0.0){
        *qi_iso_tend = qi_tend_nuc * (qv_iso/qv) * alpha_s;
    }
    else{
        *qi_iso_tend = 0.0;
    }
    return;
}

void iso_sb_2m_depostion(
        struct LookupStruct *LT, 
        double (*lam_fp)(double), 
        double (*L_fp)(double, double),
        const double type,
        const double T, 
        const double p0, 
        const double qt,
        const double qv,
        const double qv_iso,
        const double diff_vapor, 
        const double diff_iso,
        const double q_tend_dep,
        double* qv_iso_tend,
        double* q_iso_tend
    ){

    // if(q_var > 1e-10 && q_var_iso > 1e-10 && q_tend_dep > 0.0){
    if(q_tend_dep > 0.0){

        double alpha_s, alpha_k;
        if(type == 1.0){
            alpha_s = equilibrium_fractionation_factor_O18_ice(T);
        }
        else if(type == 2.0){
            alpha_s = equilibrium_fractionation_factor_HDO_ice(T);
        }

        alpha_k = alpha_k_ice_equation_Blossey(LT, lam_fp, L_fp, 
                T, p0, qt, alpha_s, DVAPOR, diff_iso);
        *q_iso_tend = alpha_s * alpha_k * q_tend_dep * (qv_iso/qv);
        // *q_iso_tend = q_tend_dep * (qv_iso/qv);
    }
    else{
        *q_iso_tend = 0.0;
    }

    *qv_iso_tend -= *q_iso_tend;

    return;
}

void iso_sb_2m_sublimation(const double q_var,
        const double q_var_iso,
        const double q_tend_sub,
        double* qv_iso_tend,
        double* q_iso_tend
    ){

    if(q_var > SB_EPS && q_var_iso > SB_EPS && q_tend_sub < 0.0){
        *q_iso_tend = q_tend_sub * (q_var_iso/q_var);
    }
    else{
        *q_iso_tend = 0.0;
    }

    *qv_iso_tend -= *q_iso_tend;
    return;
}

void sb_iso_ice_collection_snow(
        const double qi,
        const double qi_iso,
        const double qs_tend_ice_selcol,
        const double qs_tend_si_col,
        double* qs_iso_tendency,
        double* qi_iso_tendency
    ){
    if(qi > SB_EPS && qi_iso > SB_EPS){
        *qs_iso_tendency = (qs_tend_ice_selcol + qs_tend_si_col) * (qi_iso/qi);
        *qi_iso_tendency = -(qs_tend_ice_selcol + qs_tend_si_col) * (qi_iso/qi);
    }
    else{
        *qs_iso_tendency = 0.0;
        *qi_iso_tendency = 0.0;
    }
    return;
}

void sb_iso_riming_snow(
    const double ql,
    const double qr,
    const double ql_iso,
    const double qr_iso,
    const double ql_tend_snow_rime, // negative
    const double qr_tend_snow_rime, // negative
    double* ql_iso_tendency,
    double* qr_iso_tendency,
    double* qs_iso_tendency
    ){

    if(ql > SB_EPS && ql_iso > SB_EPS){
        *ql_iso_tendency += ql_tend_snow_rime * (ql_iso/ql);
        *qs_iso_tendency += -ql_tend_snow_rime * (ql_iso/ql);
    }
    else{
        *ql_iso_tendency += 0.0;
        *qs_iso_tendency += 0.0;
    }
    
    if(qr > SB_EPS && qr_iso > SB_EPS){
        *qr_iso_tendency += qr_tend_snow_rime * (qr_iso/qr);
        *qs_iso_tendency += -qr_tend_snow_rime * (qr_iso/qr);
    }
    else{
        *qr_iso_tendency += 0.0;
        *qs_iso_tendency += 0.0;
    }

    return;
}

void sb_iso_melt_snow(
    const double qs,
    const double qs_iso,
    const double qs_tend_melt, // negative
    double* qr_iso_tendency,
    double* qs_iso_tendency
    ){

    if(qs > SB_EPS && qs_iso > SB_EPS){
        *qs_iso_tendency = qs_tend_melt * (qs_iso/qs);
        *qr_iso_tendency = -qs_tend_melt * (qs_iso/qs);
    }
    else{
        *qs_iso_tendency = 0.0;
        *qr_iso_tendency = 0.0;
    }
    return;
}

void sb_iso_frz_ice(
    const double ql,
    const double ql_iso,
    const double qr,
    const double qr_iso,
    const double ql_tend_frz, // negative
    const double qr_tend_frz, // negative
    double* qi_iso_tendency
    ){
    
    double ql_iso_tend, qr_iso_tend;

    if(ql > SB_EPS && ql_iso > SB_EPS){
        ql_iso_tend = ql_tend_frz * (ql_iso/ql);
    }
    else{
        ql_iso_tend = 0.0;
    }
    
    if(qr > SB_EPS && qr_iso > SB_EPS){
        qr_iso_tend = qr_tend_frz * (qr_iso/qr);
    }
    else{
        qr_iso_tend = 0.0;
    }
    
    *qi_iso_tendency = qr_iso_tend + ql_iso_tend;

    return;
}

// void iso_sb_2m_snow_depostion(){
//     return;
// };
