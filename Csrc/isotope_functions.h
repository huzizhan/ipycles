#pragma once
#include "parameters.h"
#include "thermodynamic_functions.h"
#include "microphysics.h"
#include "microphysics_sb.h"
#include "lookup.h"
#include <math.h>
#define SB_EPS  1.0e-13 //small value
#define SB_EPS_iso 1.02e-13
#define KT  2.5e-2 // J/m/1s/K
#define DVAPOR 3.0e-5 // m^2/s

static inline double equilibrium_fractionation_factor_H2O18_liquid(double t){
// fractionation factor α_eq for 018 is based equations from Majoube 1971
	double alpha_tmp = exp(1137/(t*t) - 0.4156/t -2.0667e-3);  
    return alpha_tmp;
}

// Rayleigh distillation is adopted from Wei's paper in 2018 for qt_iso initialization
static inline double Rayleigh_distillation(double qt){
    double delta;
    double R;
    delta = 8.99 * log((qt*1000)/0.622) - 42.9;
    R = (delta/1000 + 1) * R_std_O18;
    return R*qt;
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

static inline double C_G_model(double RH,  double temperature, double alpha_k){
    double alpha_eq = 1.0 / equilibrium_fractionation_factor_H2O18_liquid(temperature);
    double R_sur_evap = alpha_eq*alpha_k*R_std_O18/((1-RH)+alpha_k*RH);
    return R_sur_evap;
}

static inline double vapor_diffusivity(const double temperature, const double p0, const int index){
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

// <<<<< SB_Warm Scheme >>>>>
double thermal_conductivity(const double temperature){
    /*Straka 2009 (5.33)*/
    double kappa_T = 2.591e-2 * pow((temperature / 296.0), 1.5) * (416.0 / (temperature - 120.0));
    return kappa_T;
};

double microphysics_g_std(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), double temperature){
    double lam = lam_fp(temperature);
    double L = L_fp(temperature,lam);
    double pv_sat = lookup(LT, temperature);
    double rho_sat = pv_sat/Rv/temperature;
    // double b_l = (DVAPOR*L*L*rho_sat)/KT/Rv/(temperature*temperature); // blossey's scheme for evaporation
    double b_l = (DVAPOR*rho_sat)*(L/KT/temperature)*(L/Rv/temperature - 1.0); //SB_Liquid evaporation scheme

    double g_therm = DVAPOR*rho_sat/(1.0+b_l);
    return g_therm;
}

double microphysics_g_iso(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                             double temperature, double p0, double qr, double qr_iso, double qv, double qv_iso, double sat_ratio){
    double lam = lam_fp(temperature);
    double L = L_fp(temperature,lam);
    double pv_sat = lookup(LT, temperature);
    double rho_sat = pv_sat/Rv/temperature;
    // double b_l = (DVAPOR*L*L*rho_sat)/KT/Rv/(temperature*temperature); // blossey's scheme for isotopic fractionation
    double b_l = (DVAPOR*rho_sat)*(L/KT/temperature)*(L/Rv/temperature - 1.0); // own scheme for isotopic fractionation, based on SB_Liquid evaporation scheme

    // isotopic components needed for calculation
    double R_qr         = qr_iso / qr;
    double R_qv_ambient = qv_iso / qv;
    double alpha_eq     = equilibrium_fractionation_factor_H2O18_liquid(temperature);
    double R_qr_surface = R_qr / alpha_eq;

    // blossey' scheme for isotopic fractionation calculation including b_l and gterm competent
    // double b_l          = (DVAPOR*L*L*rho_sat)/KT/Rv/(temperature*temperature); 
    // double g_therm_iso  = D_O18*rho_sat*R_qv_ambient*((rat)*(1.0 + b_l*S_l)/(1+b_l) - S_l); 

    double b_l          = (D_vapor*rho_sat)*(L/kt/temperature)*(L/Rv/temperature - 1.0); // own scheme for isotopic fractionation, based on SB_Liquid evaporation scheme
    double g_therm_iso  = D_O18*rho_sat*R_qv_ambient*((R_qr_surface/R_qv_ambient)*(1.0 + b_l*sat_ratio)/(1+b_l) - sat_ratio);
    
    double R_qr = qr_iso / qr;
    double R_qv_ambient = qv_iso / qv;
    double alpha_eq = equilibrium_fractionation_factor_H2O18(temperature);
    double R_qr_surface = R_qr / alpha_eq;
    // double rat = R_qr_surface/R_qv_ambient;
    
    // double g_therm_iso = D_O18*rho_sat*R_qv_ambient*((rat)*(1.0 + b_l*S_l)/(1+b_l) - S_l);
    double g_therm_iso = D_O18*rho_sat*R_qv_ambient*((R_qr_surface/R_qv_ambient)*(1.0 + b_l*S_l)/(1+b_l) - S_l);
    return g_therm_iso;
};

static inline void sb_iso_rain_autoconversion(double ql, double ql_iso, double qr_auto_tendency, double* qr_iso_auto_tendency){
    double R_iso;
    if(ql < SB_EPS || ql_iso < SB_EPS_iso){
    // if liquid specific humidity is negligibly small, set source terms to zero
        *qr_iso_auto_tendency = 0.0;
    }
    else{
        R_iso = ql_iso / ql;
        *qr_iso_auto_tendency =  R_iso * qr_auto_tendency;
    }
    return;
}
static inline void sb_iso_rain_accretion(double ql, double qr, double ql_iso, double qr_iso, double qr_accre_tendency, double* qr_iso_accre_tendency){
    double R_iso;

    if(ql < SB_EPS || qr < SB_EPS || ql_iso < SB_EPS_iso || qr_iso < SB_EPS_iso){ 
        *qr_iso_accre_tendency = 0.0;
    }
    else{
        R_iso = ql_iso / ql;
        *qr_iso_accre_tendency = R_iso * qr_accre_tendency;
    }
    return;
}

void sb_iso_evaporation_rain(double g_therm_iso, double sat_ratio, double nr, double qr, double mu, double qr_iso, double rain_mass, double Dp,
double Dm, double* qr_tendency){
    double gamma, dpfv, phi_v;
    const double bova = B_RAIN_SED/A_RAIN_SED;
    const double cdp  = C_RAIN_SED * Dp;
    const double mupow = mu + 2.5;
    double qr_tendency_tmp = 0.0;
    if(qr < SB_EPS || nr < SB_EPS || qr_iso < SB_EPS){
        *qr_tendency = 0.0;
    }
    else if(sat_ratio >= 0.0){
        *qr_tendency = 0.0;
    }
    else{
        gamma = 0.7; // gamma = 0.7 is used by DALES ; alternative expression gamma= d_eq/Dm * exp(-0.2*mudouble vapor_H2O18_diff, vapor_HDO16_diff; 2double vapor_H2O18_diff, vapor_HDO16_diff;) is used by S08;
        phi_v = 1.0 - (0.5  * bova * pow(1.0 +  cdp, -mupow) + 0.125 * bova * bova * pow(1.0 + 2.0*cdp, -mupow)
                      + 0.0625 * bova * bova * bova * pow(1.0 +3.0*cdp, -mupow) + 0.0390625 * bova * bova * bova * bova * pow(1.0 + 4.0*cdp, -mupow));


        dpfv  = (A_VENT_RAIN * tgamma(mu + 2.0) * Dp + B_VENT_RAIN * NSC_3 * A_NU_SQ * tgamma(mupow) * pow(Dp, 1.5) * phi_v)/tgamma(mu + 1.0);

        qr_tendency_tmp = 2.0 * pi * g_therm_iso * nr * dpfv;
        *qr_tendency = -qr_tendency_tmp;
    }
    return;
}

// ===========<<< iso 1-m ice scheme >>> ============

static inline double equilibrium_fractionation_factor_H2O18_ice(double t){
// fractionation factor α_eq for 018 for vapor between ice, based equations from Majoube 1971
	double alpha_ice = exp(11.839/t - 2.8224e-2);  
    return alpha_ice;
}

double alpha_k_ice_equation(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                             double temperature, double p0, double qt, double alpha_s){
    double lam         = lam_fp(temperature);
    double L           = L_fp(temperature,lam);
    double pv_sat_ice  = lookup(LT, temperature);
    double rho_sat_ice = pv_sat_ice/Rv/temperature;
    // calculate sat_ratio of vapor respect to ice, S_s is the same simple in Blossey's 2015
    double qv_sat_ice  = qv_star_c(p0,qt,pv_sat_ice);
    double S_s         = qt/qv_sat_ice;
    double D_O18       = DVAPOR*0.9723;
    double diff_ratio  = DVAPOR / D_O18;
    double b_s         = (DVAPOR*rho_sat_ice)*(L/KT/temperature)*(L/Rv/temperature - 1.0);
    double alpha_k_ice = (1 + b_s) * S_s * (1/(alpha_s*diff_ratio*(S_s - 1) + 1 + b_s*S_s));
    return alpha_k_ice;
}

