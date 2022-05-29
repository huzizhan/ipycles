#pragma once
#include "parameters.h"
#include "thermodynamic_functions.h"
#include "microphysics.h"
#include "microphysics_sb.h"
#include "microphysics_arctic_1m.h"
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

    double S_l = sat_ratio + 1.0;
    double D_O18 = DVAPOR*0.9723;
    
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
    if (ql > SB_EPS && ql_iso > SB_EPS){
        *qr_iso_auto_tendency = qr_auto_tendency * (ql_iso/ql);
    }
    else{
        *qr_iso_auto_tendency = 0.0;
    }
}
static inline void sb_iso_rain_accretion(double ql, double ql_iso, double qr_accre_tendency, double* qr_iso_accre_tendency){
    if (ql > SB_EPS && ql_iso > SB_EPS){
        *qr_iso_accre_tendency = qr_accre_tendency * (ql_iso/ql);
    }
    else{
        *qr_iso_accre_tendency = 0.0;
    }
}
static inline void sb_iso_rain_evap_nofrac(double qr, double qr_iso, double qr_evap_tendency, double* qr_iso_evap_tendency){
    if (qr > SB_EPS && qr_iso > SB_EPS){
        *qr_iso_evap_tendency = qr_evap_tendency * (qr_iso/qr);
    }
    else{
        *qr_iso_evap_tendency = 0.0;
    }
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
        *qr_iso_tendency = -qr_tendency_tmp;
    }
    return;
}

// ===========<<< iso 1-m ice scheme >>> ============

static inline double equilibrium_fractionation_factor_H2O18_ice(double t){
// fractionation factor α_eq for 018 for vapor between ice, based equations from Majoube 1971
	double alpha_ice = exp(11.839/t - 2.8224e-2);  
    return alpha_ice;
}

double alpha_k_ice_equation_Blossey(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                             double temperature, double p0, double qt, double alpha_s){
    // this function is adopted from Blossey's 2015, for calculate of alpha_k
    double lam            = lam_fp(temperature);
    double L              = L_fp(temperature,lam);
    double pv_sat_ice     = lookup(LT, temperature);
    double rho_sat_ice    = pv_sat_ice/Rv/temperature;
    // calculate sat_ratio of vapor respect to ice, S_s is the same simple in Blossey's 2015
    double qv_sat_ice     = qv_star_c(p0,qt,pv_sat_ice);
    double S_s            = qt/qv_sat_ice;
    double vapor_O18_diff = DVAPOR*0.9723;
    double diff_ratio     = DVAPOR / vapor_O18_diff;
    double b_s            = (DVAPOR*rho_sat_ice)*(L/KT/temperature)*(L/Rv/temperature - 1.0);
    double alpha_k_ice    = (1 + b_s) * S_s * (1/(alpha_s*diff_ratio*(S_s - 1) + 1 + b_s*S_s));
    return alpha_k_ice;
}

double alpha_k_ice_equation_Jouzel(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                             double temperature, double p0, double qt, double alpha_s){
    // this function is adopted from Jouzel's 1984, for calculate of alpha_k
    double lam            = lam_fp(temperature);
    double L              = L_fp(temperature,lam);
    double pv_sat_ice     = lookup(LT, temperature);
    double rho_sat_ice    = pv_sat_ice/Rv/temperature;
    // calculate sat_ratio of vapor respect to ice, S_s means the sat_ratio
    double qv_sat_ice     = qv_star_c(p0,qt,pv_sat_ice);
    double S_s            = qt/qv_sat_ice;
    double vapor_O18_diff = DVAPOR*0.9723;
    double diff_ratio     = DVAPOR / vapor_O18_diff;
    double alpha_k_ice    = S_s/(alpha_s*diff_ratio*(S_s-1.0)+1.0);
    return alpha_k_ice;
}

void arc1m_iso_auto_acc_rain(double qrain_tendency_aut, double qrain_tendency_acc, double qrain, double qrain_iso, 
                       double* qrain_iso_tendency_auto, double *qrain_iso_tendency_acc){
    double iso_ratio = qrain_iso / qrain;
    if(qrain_tendency_aut == 0.0){
        *qrain_iso_tendency_auto = 0.0;
    }
    else{
        *qrain_iso_tendency_auto = qrain_tendency_aut * iso_ratio;
    }
    if(qrain_tendency_acc == 0.0){
        *qrain_iso_tendency_acc = 0.0;
    }
    else{
        *qrain_iso_tendency_acc = qrain_tendency_acc * iso_ratio;
    }
    return;
}

void arc1m_iso_auto_acc_snow(double qsnow_tendency_aut, double qsnow_tendency_acc, double qsnow, double qsnow_iso, 
                       double* qsnow_iso_tendency_auto, double *qsnow_iso_tendency_acc){
    double iso_ratio = qsnow_iso / qsnow;
    if(qsnow_tendency_aut == 0.0){
        *qsnow_iso_tendency_auto = 0.0;
    }
    else{
        *qsnow_iso_tendency_auto = qsnow_tendency_aut * iso_ratio;
    }
    if(qsnow_tendency_acc == 0.0){
        *qsnow_iso_tendency_acc = 0.0;
    }
    else{
        *qsnow_iso_tendency_acc = qsnow_tendency_acc * iso_ratio;
    }
    return;
}

void arc1m_iso_evap_rain(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                   double density, const double p0, double temperature,
                   double qt, double qv, double qrain, double nrain, 
                   double qv_iso, double qrain_iso, double *qrain_iso_tendency){
    double beta           = 2.0;
    double pv_star        = lookup(LT, temperature);
    double qv_star        = qv_star_c(p0, qt, pv_star);
    double satratio       = qt/qv_star;
    double therm_cond     = thermal_conductivity(temperature);
    double rain_diam      = rain_dmean(density, qrain, nrain);
    double rain_vel       = C_RAIN*pow(rain_diam, D_RAIN);
    double rain_lam       = rain_lambda(density, qrain, nrain);

    double vapor_diff     = vapor_diffusivity(temperature, p0);
    double vapor_O18_diff = vapor_diff * 0.9723;

    double re, vent, gtherm;

    if( satratio < 1.0 && qrain > 1.0e-15){
        re                  = rain_diam*rain_vel/VISC_AIR;
        vent                = 0.78 + 0.27*sqrt(re);
        // gtherm              = 1.0e-7/(2.2*temperature/pv_star + 220.0/temperature);
        // gtherm              = 1.0 / ( (Rv*temperature/vapor_diff/pv_star) + (8.028e12/therm_cond/Rv/(temperature*temperature)) );
        double gtherm_iso   = microphysics_g_iso(LT, lam_fp, L_fp, temperature, 
                                                 vapor_diff, vapor_O18_diff, therm_cond, p0, satratio,
                                                 qrain, qrain_iso, qv, qv_iso);
        *qrain_iso_tendency = 4.0*pi/beta*(satratio - 1.0)*vent*gtherm_iso*nrain/(rain_lam*rain_lam)/density;
    }
    return;
}

void arc1m_iso_evap_snow_nofrac(double qsnow_tendency_evap, double qsnow, double qsnow_iso, double* qsnow_iso_tendency_evap){
    double iso_ratio = qsnow_iso / qsnow;
    if(qsnow_tendency_evap == 0.0){
        *qsnow_iso_tendency_evap = 0.0;
    }
    else{
        *qsnow_iso_tendency_evap = qsnow_tendency_evap * iso_ratio;
    }
    return;
}

void arc1m_iso_evap_snow_withfrac(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                   double density, const double p0, double temperature,
                   double qt, double qv, double qsnow, double nsnow, 
                   double qv_iso, double qsnow_iso, double* qsnow_iso_tendency){
    double beta           = 3.0;
    double pv_star        = lookup(LT, temperature);
    double qv_star        = qv_star_c(p0, qt, pv_star);
    double satratio       = qt/qv_star;
    double therm_cond     = thermal_conductivity(temperature);
    double snow_diam      = snow_dmean(density, qsnow, nsnow);
    double snow_vel       = C_SNOW*pow(snow_diam, D_SNOW);
    double snow_lam       = snow_lambda(density, qsnow, nsnow);


    double vapor_diff     = vapor_diffusivity(temperature, p0);
    double vapor_O18_diff = vapor_diff * 0.9723;

    double re, vent, gtherm;

    if( satratio < 1.0 && qsnow > 1.0e-15){
        double re           = snow_diam*snow_vel/VISC_AIR;
        double vent         = 0.65 + 0.39*sqrt(re);
        // gtherm           = 1.0e-7/(2.2*temperature/pv_star + 220.0/temperature);
        // gtherm           = 1.0 / ( (Rv*temperature/vapor_diff/pv_star) + (8.028e12/therm_cond/Rv/(temperature*temperature)) );
        double gtherm_iso   = microphysics_g_iso(LT, lam_fp, L_fp, temperature,
                                                 vapor_diff, vapor_O18_diff, therm_cond, p0, satratio,
                                                 qsnow, qsnow_iso, qv, qv_iso);
        *qsnow_iso_tendency = 4.0*pi/beta*(satratio - 1.0)*vent*gtherm_iso*nsnow/(snow_lam*snow_lam)/density;
    }
    return;
}

void arc1m_iso_melt_snow(double qsnow_tendency_melt, double qsnow, double qsnow_iso, double* qsnow_iso_tendency_melt){
    double iso_ratio = qsnow_iso / qsnow;
    if(qsnow_tendency_melt == 0.0){
        *qsnow_iso_tendency_melt = 0.0;
    }
    else{
        *qsnow_iso_tendency_melt = qsnow_tendency_melt * iso_ratio;
    }
    return;
}

