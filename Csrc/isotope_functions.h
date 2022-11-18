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

// ===========<<<  SB_Warm Scheme  >>> ============

double microphysics_g_std(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), double temperature, double dvap, double kt){
    double lam = lam_fp(temperature);
    double L = L_fp(temperature,lam);
    double pv_sat = lookup(LT, temperature);
    double rho_sat = pv_sat/Rv/temperature;

    /*blossey's scheme for evaporation*/
    // double b_l = (DVAPOR*L*L*rho_sat)/KT/Rv/(temperature*temperature); 
    
    /*Straka 2009 (6.13)*/
    double b_l = (dvap*rho_sat)*(L/kt/temperature)*(L/Rv/temperature - 1.0);

    double g_therm = dvap*rho_sat/(1.0+b_l);
    return g_therm;
}

double microphysics_g_iso(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                             double temperature, double p0, double qr, double qr_iso, double qv, double qv_iso, double sat_ratio,
                             double dvap, double kt){
    double lam          = lam_fp(temperature);
    double L            = L_fp(temperature,lam);
    double pv_sat       = lookup(LT, temperature);
    double rho_sat      = pv_sat/Rv/temperature;
    // double b_l       = (DVAPOR*L*L*rho_sat)/KT/Rv/(temperature*temperature); // blossey's scheme for isotopic fractionation
    double b_l          = (dvap*rho_sat)*(L/kt/temperature)*(L/Rv/temperature - 1.0); // own scheme for isotopic fractionation, based on SB_Liquid evaporation scheme

    double S_l          = sat_ratio + 1.0;
    double D_O18        = dvap*0.9723;
    
    double R_qr         = qr_iso / qr;
    double R_qv_ambient = qv_iso / qv;
    double alpha_eq     = equilibrium_fractionation_factor_H2O18_liquid(temperature);
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
    return;
}

static inline void sb_iso_rain_accretion(double ql, double ql_iso, double qr_accre_tendency, double* qr_iso_accre_tendency){
    if (ql > SB_EPS && ql_iso > SB_EPS){
        *qr_iso_accre_tendency = qr_accre_tendency * (ql_iso/ql);
    }
    else{
        *qr_iso_accre_tendency = 0.0;
    }
    return;
}

static inline void sb_iso_rain_evap_nofrac(double qr, double qr_iso, double qr_evap_tendency, double* qr_iso_evap_tendency){
    if (qr > SB_EPS && qr_iso > SB_EPS){
        *qr_iso_evap_tendency = qr_evap_tendency * (qr_iso/qr);
    }
    else{
        *qr_iso_evap_tendency = 0.0;
    }
    return;
}

void sb_iso_evaporation_rain(double g_therm_iso, double sat_ratio, double nr, double qr, double mu, double qr_iso, double rain_mass, double Dp,
double Dm, double* qr_iso_tendency){
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
        // * exp(-0.2*mudouble vapor_H2O18_diff, vapor_HDO16_diff; 2double
        // vapor_H2O18_diff, vapor_HDO16_diff;) is used by S08;
        phi_v = 1.0 - (0.5  * bova * pow(1.0 +  cdp, -mupow) + 0.125 * bova * bova * pow(1.0 + 2.0*cdp, -mupow)
                      + 0.0625 * bova * bova * bova * pow(1.0 +3.0*cdp, -mupow) + 0.0390625 * bova * bova * bova * bova * pow(1.0 + 4.0*cdp, -mupow));
        dpfv  = (A_VENT_RAIN * tgamma(mu + 2.0) * Dp + B_VENT_RAIN * NSC_3 * A_NU_SQ * tgamma(mupow) * pow(Dp, 1.5) * phi_v)/tgamma(mu + 1.0);

        qr_tendency_tmp  = 2.0 * pi * g_therm_iso * nr * dpfv;
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

void arc1m_iso_autoconversion_rain(double qrain_tendency_aut, double R_ql, double* qrain_iso_tendency_auto){
    *qrain_iso_tendency_auto = qrain_tendency_aut * R_ql;
    return;
}

void arc1m_iso_evap_rain_nofrac(double qrain_tendency_evap, double R_qrain, double* qrain_iso_tendency_evap){
    *qrain_iso_tendency_evap = qrain_tendency_evap * R_qrain;
    return;
}

void arc1m_iso_autoconversion_snow(double qsnow_tendency_aut, double R_qi, double* qsnow_iso_tendency_auto){
    *qsnow_iso_tendency_auto = qsnow_tendency_aut * R_qi;
    return;
}

void arc1m_iso_evap_snow_nofrac(double qsnow_tendency_evap, double R_qsnow, double* qsnow_iso_tendency_evap){
    *qsnow_iso_tendency_evap = qsnow_tendency_evap * R_qsnow;
    return;
}

void arc1m_iso_melt_snow(double qsnow_tendency_melt, double R_qsnow, double* qsnow_iso_tendency_melt){
    *qsnow_iso_tendency_melt = qsnow_tendency_melt * R_qsnow;
    return;
}

void arc1m_iso_accretion_all(double density, double p0, double temperature, double ccn, double ql, double qi, double ni,
                   double qrain, double nrain, double qsnow, double nsnow,
                   double R_ql, double R_qi, double R_qrain, double R_qsnow,
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

void arc1m_std_evap_rain(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                      double density, const double p0, double temperature,
                      double qt, double qrain, double nrain, double* qrain_tendency){
    double beta = 2.0;
    double pv_star = lookup(LT, temperature);
    double qv_star = qv_star_c(p0, qt, pv_star);
    double satratio = qt/qv_star;
    double vapor_diff = vapor_diffusivity(temperature, p0);
    double therm_cond = thermal_conductivity(temperature);
    double rain_diam = rain_dmean(density, qrain, nrain);
    double rain_vel = C_RAIN*pow(rain_diam, D_RAIN);
    double rain_lam = rain_lambda(density, qrain, nrain);

    double re, vent, gtherm;

    if( satratio < 1.0 && qrain > 1.0e-15){
        re = rain_diam*rain_vel/VISC_AIR;
        vent = 0.78 + 0.27*sqrt(re);
        gtherm = microphysics_g_std(LT, lam_fp, L_fp, temperature, vapor_diff, therm_cond);
        *qrain_tendency = 4.0*pi/beta*(satratio - 1.0)*vent*gtherm*nrain/(rain_lam*rain_lam)/density;
    }

    return;
};

void arc1m_iso_evap_rain(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                   double density, const double p0, double temperature,
                   double qt, double qv, double qrain, double nrain, 
                   double qv_iso, double qrain_iso, double *qrain_iso_tendency){
    double beta       = 2.0;
    double pv_star    = lookup(LT, temperature);
    double qv_star    = qv_star_c(p0, qt, pv_star);
    double satratio   = qt/qv_star;
    double vapor_diff = vapor_diffusivity(temperature, p0);
    double therm_cond = thermal_conductivity(temperature);
    double rain_diam  = rain_dmean(density, qrain, nrain);
    double rain_vel   = C_RAIN*pow(rain_diam, D_RAIN);
    double rain_lam   = rain_lambda(density, qrain, nrain);

    double re, vent, gther_iso; 
    if( satratio < 1.0 && qrain > 1.0e-15 && qrain_iso > 1.0e-15){
        re = rain_diam*rain_vel/VISC_AIR;
        vent = 0.78 + 0.27*sqrt(re);
        double sat_ratio = satratio - 1.0;
        gther_iso = microphysics_g_iso(LT, lam_fp, L_fp, temperature, p0, qrain, qrain_iso, qv, qv_iso, sat_ratio, vapor_diff, therm_cond);
        double qrain_evap_tend_tmp = 4.0*pi/beta*vent*gther_iso*nrain/(rain_lam*rain_lam)/density;
        *qrain_iso_tendency = -qrain_evap_tend_tmp;
    }
    return;
}

void arc1m_iso_evap_snow(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                   double density, const double p0, double temperature,
                   double qt, double qv, double qsnow, double nsnow, 
                   double qv_iso, double qsnow_iso, double* qsnow_iso_tendency){
    double beta = 3.0;
    double pv_star = lookup(LT, temperature);
    double qv_star = qv_star_c(p0, qt, pv_star);
    double satratio = qt/qv_star;

    double vapor_diff = vapor_diffusivity(temperature, p0);
    double therm_cond = thermal_conductivity(temperature);
    double snow_diam = snow_dmean(density, qsnow, nsnow);
    double snow_vel = C_SNOW*pow(snow_diam, D_SNOW);
    double snow_lam = snow_lambda(density, qsnow, nsnow);

    if( qsnow > 1.0e-15 ){
        double re = snow_diam*snow_vel/VISC_AIR;
        double vent = 0.65 + 0.39*sqrt(re);
        double sat_ratio = satratio - 1.0;
        double gtherm_iso = microphysics_g_iso(LT, lam_fp, L_fp, temperature, p0, qsnow, qsnow_iso, qv, qv_iso, sat_ratio, vapor_diff, therm_cond);
        double qsnow_iso_tend_tmp = 4.0*pi/beta*vent*gtherm_iso*nsnow/(snow_lam*snow_lam)/density;
        *qsnow_iso_tendency = -qsnow_iso_tend_tmp;
    }
    return;
}

// ===========<<< Single Ice microphysics scheme coupling with Isotope processes >>> ============
void sb_iso_ice_nucleation(const double qi_tendency_nuc, const double alpha_s_ice, double* qi_iso_tendency_nuc){
    //-------------------------------------------------------------
    // INPUT VARIABLES
    //-------------------------------------------------------------
    // qi_tendency_nuc: single ice tendency during nucleation
    // alpha_s_ice: equilibrium fractionation factor between vapor and ice
    //-------------------------------------------------------------
    // OUTPUT VARIABLES
    //-------------------------------------------------------------
    // qi_iso_tendency_nuc: single ice isotope content tendency due to nucleation;
    //-------------------------------------------------------------
    *qi_iso_tendency_nuc = alpha_s_ice*qi_tendency_nuc;
    return;
};

void sb_iso_ice_freezing(const double ql_tendency_frz, const double qr_tendency_frz, const double R_ql, const double R_qr,
        double* qr_iso_tendency_frz, double* ql_iso_tendency_frz, double* qi_iso_tendency_frz){
    //-------------------------------------------------------------
    // INPUT VARIABLES
    //-------------------------------------------------------------
    // ql_tendency_frz: ql tendency during freezing, calculated from sb_freezing_ice section, POSITIVE
    // qr_tendency_frz: qr tendency during freezing, calculated from sb_freezing_ice section, POSITIVE
    // R_ql: isotope ratio of ql;
    // R_qr: isotope ratio of qr;
    //-------------------------------------------------------------
    // OUTPUT VARIABLES
    //-------------------------------------------------------------
    // qi_iso_tendency_frz: single ice isotope content tendency due to freezing
    //-------------------------------------------------------------
    *qr_iso_tendency_frz = qr_tendency_frz * R_qr;
    *ql_iso_tendency_frz = ql_tendency_frz * R_ql;
    *qi_iso_tendency_frz = ql_tendency_frz*R_ql + qr_tendency_frz*R_qr;
    return;
};
void sb_iso_ice_accretion_cloud(const double qi_tendency_acc, const double R_qi, double* qi_iso_tendency_acc){
    //-------------------------------------------------------------
    // INPUT VARIABLES
    //-------------------------------------------------------------
    // qi_tendency_acc: single ice tendency during cloud liquid accretion
    // R_ql: isotope ratio of single ice;
    //-------------------------------------------------------------
    // OUTPUT VARIABLES
    //-------------------------------------------------------------
    // qi_iso_tendency_acc: single ice isotope content tendency due to cloud liquid accretion
    //-------------------------------------------------------------
    *qi_iso_tendency_acc = qi_tendency_acc*R_qi;
    return;
};

void sb_iso_ice_melting(const double qi_tendency_mlt, const double R_qi, double* qi_iso_tendency_mlt){
    //-------------------------------------------------------------
    // INPUT VARIABLES
    //-------------------------------------------------------------
    // qi_tendency_mlt: single ice tendency during melting
    // R_ql: isotope ratio of single ice;
    //-------------------------------------------------------------
    // OUTPUT VARIABLES
    //-------------------------------------------------------------
    // qi_iso_tendency_mlt: single ice isotope content tendency due to melting
    //-------------------------------------------------------------
    *qi_iso_tendency_mlt = qi_tendency_mlt*R_qi;
    return;
};

void sb_iso_ice_sublimation(const double qi_tendency_sub, const double R_qi, double* qi_iso_tendency_sub){
    //-------------------------------------------------------------
    // INPUT VARIABLES
    //-------------------------------------------------------------
    // qi_tendency_sub: single ice tendency during melting
    // R_ql: isotope ratio of single ice;
    //-------------------------------------------------------------
    // OUTPUT VARIABLES
    //-------------------------------------------------------------
    // qi_iso_tendency_sub: single ice isotope content tendency due to melting
    //-------------------------------------------------------------
    *qi_iso_tendency_sub = qi_tendency_sub*R_qi;
    return;
};
void sb_iso_ice_deposition(const double alpha_k_ice, const double alpha_s_ice, 
        const double qi_tendency_dep, const double F_ratio, double* qi_iso_tendency_dep){
    //-------------------------------------------------------------
    // INPUT VARIABLES
    //-------------------------------------------------------------
    // alpha_k_ice: kinetic fractionation factor between vapor and ice in supper-saturation 
    // alpha_s_ice: equilibrium fractionation factor between vapor and ice
    // qi_tendency_dep: single ice tendency during deposition
    // F_ratio: ventilation factor ratio, between light and heavy water isotopes
    //-------------------------------------------------------------
    // OUTPUT VARIABLES
    //-------------------------------------------------------------
    // qi_iso_tendency_dep: single ice isotope content tendency due to deposition
    //-------------------------------------------------------------

    *qi_iso_tendency_dep = qi_tendency_dep*alpha_s_ice*alpha_k_ice * F_ratio;
};
