#pragma once
#include "parameters.h"
#include <math.h>

static inline double exner_c(const double p0){
    return pow((p0/p_tilde),kappa);
}

static inline double theta_c(const double p0, const double T){
    // Dry potential temperature
    return T / exner_c(p0);
}

static inline double thetali_c(const double p0, const double T, const double qt, const double ql, const double qi, const double L){
    // Liquid ice potential temperature consistent with Tripoli and Cotton (1981)
    return theta_c(p0, T) * exp(-L*(ql/(1.0 - qt) + qi/(1.0 - qt))/(T*cpd));
}

static inline double pd_c(const double p0,const double qt, const double qv){
    return p0*(1.0-qt)/(1.0 - qt + eps_vi * qv);
}

static inline double pv_c(const double p0, const double qt, const double qv){
    return p0 * eps_vi * qv /(1.0 - qt + eps_vi * qv);
}

static inline double density_temperature_c(const double T, const double qt, const double qv){
    return T * (1.0 - qt + eps_vi * qv);
}

static inline double theta_rho_c(const double p0, const double T, const double qt, const double qv){
    return density_temperature_c(T,qt,qv)/exner_c(p0);
}

static inline double cpm_c(const double qt){
    return (1.0-qt) * cpd + qt * cpv;
}

static inline double thetas_c(const double s, const double qt){
    // expression of entropy temperature based on Equ 47, in Pressel15
    return T_tilde*exp((s-(1.0-qt)*sd_tilde - qt*sv_tilde)/cpm_c(qt));
}

static inline double thetas_t_c(const double p0, const double T, const double qt, const double qv, const double qc, const double L){
    // expression of entropy temperature based on temperature and other related thermodynamic variables, based on Equ 48, in Pressel15
    const double qd = 1.0 - qt;
    const double pd = pd_c(p0,qt,qt-qc);
    const double pv = pv_c(p0,qt,qt-qc);
    const double cpm = cpm_c(qt);
    return T * pow(p_tilde/pd,qd * Rd/cpm)*pow(p_tilde/pv,qt*Rv/cpm)*exp(-L * qc/(cpm*T));
}

static inline double entropy_from_thetas_c(const double thetas, const double qt){
    // expression of specific entropy
    return cpm_c(qt) * log(thetas/T_tilde) + (1.0 - qt)*sd_tilde + qt * sv_tilde;
}

static inline double buoyancy_c(const double alpha0, const double alpha){
    return g * (alpha - alpha0)/alpha0;
}

static inline double qv_star_c(const double p0, const double qt, const double pv){
    return eps_v * (1.0 - qt) * pv / (p0 - pv) ;
}

static inline double alpha_c(double p0, double T, double  qt, double qv){
    //specific volume, equation 44 in Pressel, 2015
    return (Rd * T)/p0 * (1.0 - qt + eps_vi * qv);
}

// =============== New method to calculate saturation vapor over liquid and ice ==============
// This method is adopted from Maarten H. P. Ambaum's 2020
// which provide a new solution for simple and accurate 
// calculation of saturation vapor pressure over ice and liquid
// and this is adopted to replace the look-up table method 
// in pycles before, aming to simulate the wbf process better (S_l != S_i)
// ===========================================================================================
static inline double saturation_vapor_pressure_water(double temperature){
    // define variables for vapor-liquid saturation vapor
    const double e_s0 = 611.655; // Pa
    const double T_0 = 273.16; // K, temperature at triple-point
    const double L_0 = 2.501e6; // J kg^-1 
    const double cpl_cpv = 2180; // J kg^-1 K^-1

    double L = L_0 - cpl_cpv*(temperature - T_0);
    return e_s0*pow(T_0/temperature, cpl_cpv/Rv) * exp(L_0/Rv/T_0 - L/Rv/temperature);
}
static inline double saturation_vapor_pressure_water_simple(double temperature){
    const double e_s0 = 611.655; // Pa
    const double T_0 = 273.16; // K, temperature at triple-point
    const double L_0 = 2.501e6; // J kg^-1 
    return e_s0 * exp(L_0/Rv/T_0 - L_0/Rv/temperature);
}
static inline double saturation_vapor_pressure_ice(double temperature){
    // define variables for vapor-ice saturation vapor;
    const double e_i0 = 611.655; // Pa
    const double T_0 = 273.16; // K, temperature at triple-point
    const double cpi_cpv = 212; // J kg^-1 K^-1
    const double L_s0 = 2.834e6; // J kg^-1
    double L_s = L_s0 - cpi_cpv*(temperature - T_0);
    return e_i0 * pow(T_0/temperature, cpi_cpv/Rv) * exp(L_s0/Rv/T_0 - L_s/Rv/temperature);
}

