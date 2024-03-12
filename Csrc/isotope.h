#pragma once
#include "parameters.h"
#include "isotope_functions.h"
#include "thermodynamics_sa.h"
#include "thermodynamics_sb.h"
#include "entropies.h"
#include "thermodynamic_functions.h"
#include "microphysics.h"
#include "microphysics_sb.h"
#include "microphysics_sb_si.h"
#include "microphysics_sb_liquid.h"
#include "microphysics_sb_ice.h"
#include "microphysics_arctic_1m.h"
#include "advection_interpolation.h"
#include <math.h>
// #define SB_EPS 1.0e-13

// the following are the ratio of diffusivity of HDO and 
// H2O18, and are adopted from Cappa's 2003
#define DIFF_HDO_RATIO 0.9691
#define DIFF_O18_RATIO 0.9839

void iso_equilibrium_fractionation_No_Microphysics(struct DimStruct *dims, double* restrict temperature,
    double* restrict qt, double* restrict qv_DV, double* restrict ql_DV, double* restrict qv_std, double* restrict ql_std, 
    double* restrict qt_O18, double* restrict qv_O18, double* restrict ql_O18, 
    double* restrict qt_HDO, double* restrict qv_HDO, double* restrict ql_HDO){

    ssize_t i,j,k;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];

    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
                for (k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    double qv_std_tmp, ql_std_tmp, qv_O18_tmp, ql_O18_tmp, qv_HDO_tmp, ql_HDO_tmp;
                    double alpha_eq_lv_O18 = equilibrium_fractionation_factor_O18_liquid(temperature[ijk]);
                    double alpha_eq_lv_HDO = equilibrium_fractionation_factor_HDO_liquid(temperature[ijk]);

                    qv_std_tmp = qv_DV[ijk];
                    qv_O18_tmp = eq_frac_function(qt_O18[ijk], qv_DV[ijk], ql_DV[ijk], alpha_eq_lv_O18);
                    qv_HDO_tmp = eq_frac_function(qt_HDO[ijk], qv_DV[ijk], ql_DV[ijk], alpha_eq_lv_HDO);

                    ql_std_tmp = ql_DV[ijk];
                    ql_O18_tmp = qt_O18[ijk] - qv_O18_tmp;
                    ql_HDO_tmp = qt_HDO[ijk] - qv_HDO_tmp;
                    
                    qv_std[ijk] = qv_std_tmp;
                    ql_std[ijk] = ql_std_tmp;
                    qv_O18[ijk] = qv_O18_tmp;
                    ql_O18[ijk] = ql_O18_tmp;
                    qv_HDO[ijk] = qv_HDO_tmp;
                    ql_HDO[ijk] = ql_HDO_tmp;
                } // End k loop
            } // End j loop
        } // End i loop
    return;
}
// Scaling the isotope specific humidity values back to correct magnitude
void statsIO_isotope_scaling_magnitude(struct DimStruct *dims, double* restrict tmp_values){
    ssize_t i;
    const ssize_t imin = 0;
    const ssize_t imax = dims->nlg[2];
    for (i=imin; i<imax; i++){
        tmp_values[i] *= R_std_O18;
    } 
    return;
}

void tracer_sb_liquid_microphysics_sources(const struct DimStruct *dims, struct LookupStruct *LT, 
    double (*lam_fp)(double), double (*L_fp)(double, double),
    double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
    double* restrict density, double* restrict p0,  double* restrict temperature,  
    double* restrict qt, double ccn,
    double* restrict ql, double* restrict nr, double* restrict qr, double dt,
    double* restrict nr_tendency_micro, double* restrict qr_tendency_micro, 
    double* restrict nr_std_tendency, double* restrict qr_std_tendency,
    double* restrict qr_O18, double* restrict qt_O18, double* restrict qv_O18, double* restrict ql_O18,
    double* restrict qr_HDO, double* restrict qt_HDO, double* restrict qv_HDO, double* restrict ql_HDO,
    double* restrict qr_O18_tendency_micro, double* restrict qr_O18_tendency, 
    double* restrict qr_HDO_tendency_micro, double* restrict qr_HDO_tendency){

    //Here we compute the source terms for nr and qr (number and mass of rain)
    //Temporal substepping is used to help ensure boundedness of moments
    double rain_mass, Dm, mu, Dp, nr_tendency_tmp, qr_tendency_tmp, ql_tendency_tmp;
    double nr_tendency_au, nr_tendency_scbk, nr_tendency_evp;
    double qr_tendency_au, qr_tendency_ac,  qr_tendency_evp;
    double sat_ratio;
    double qr_O18_tmp, qr_O18_tend, qr_O18_tendency_tmp, qt_O18_tendency_tmp, qv_O18_tendency_tmp, ql_O18_tendency_tmp;
    double qr_HDO_tmp, qr_HDO_tend, qr_HDO_tendency_tmp, qt_HDO_tendency_tmp, qv_HDO_tendency_tmp, ql_HDO_tendency_tmp;
    double qr_O18_auto_tendency, qr_O18_accre_tendency, qr_O18_evap_tendency;
    double qr_HDO_auto_tendency, qr_HDO_accre_tendency, qr_HDO_evap_tendency;

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
                qr[ijk]           = fmax(qr[ijk],0.0);
                nr[ijk]           = fmax(fmin(nr[ijk], qr[ijk]/RAIN_MIN_MASS),qr[ijk]/RAIN_MAX_MASS);
                double qv_tmp     = qt[ijk] - fmax(ql[ijk],0.0);
                double qt_tmp     = qt[ijk];
                double nl         = ccn/density[k];
                double ql_tmp     = fmax(ql[ijk],0.0);
                double qr_tmp     = fmax(qr[ijk],0.0);
                double nr_tmp     = fmax(fmin(nr[ijk], qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                double g_therm    = microphysics_g(LT, lam_fp, L_fp, temperature[ijk]);

                double ql_O18_tmp = fmax(ql_O18[ijk], 0.0);
                double qr_O18_tmp = fmax(qr_O18[ijk], 0.0);
                double qv_O18_tmp = qv_O18[ijk];

                double ql_HDO_tmp = fmax(ql_HDO[ijk], 0.0);
                double qr_HDO_tmp = fmax(qr_HDO[ijk], 0.0);
                double qv_HDO_tmp = qv_HDO[ijk];

                //holding nl fixed since it doesn't change between timesteps
                double time_added  = 0.0, dt_, rate;
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

                    //obtain some parameters
                    rain_mass = microphysics_mean_mass(nr_tmp, qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS);
                    Dm        = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi);
                    mu        = rain_mu(density[k], qr_tmp, Dm);
                    // Dp     = Dm * cbrt(tgamma(mu + 1.0) / tgamma(mu + 4.0));
                    Dp        = sb_Dp(Dm, mu);

                    //compute the source terms
                    sb_autoconversion_rain(droplet_nu, density[k], nl, ql_tmp, qr_tmp, &nr_tendency_au, &qr_tendency_au);
                    sb_accretion_rain(density[k], ql_tmp, qr_tmp, &qr_tendency_ac);
                    sb_selfcollection_breakup_rain(density[k], nr_tmp, qr_tmp, mu, rain_mass, Dm, &nr_tendency_scbk);
                    double diagnose_tmp;
                    sb_evaporation_rain_debug(LT, lam_fp, L_fp, temperature[ijk], sat_ratio, nr_tmp, qr_tmp, mu, 
                            rain_mass, Dp, Dm, &diagnose_tmp, &nr_tendency_evp, &qr_tendency_evp);
                    //find the maximum substep time
                    dt_ = dt - time_added;
                    //check the source term magnitudes
                    nr_tendency_tmp = nr_tendency_au + nr_tendency_scbk + nr_tendency_evp;
                    qr_tendency_tmp = qr_tendency_au + qr_tendency_ac + qr_tendency_evp;
                    ql_tendency_tmp = -qr_tendency_au - qr_tendency_ac;

                    // //iso_tendencies initilize
                    qr_O18_auto_tendency  = 0.0;
                    qr_O18_accre_tendency = 0.0;
                    qr_O18_evap_tendency  = 0.0;

                    qr_HDO_auto_tendency  = 0.0;
                    qr_HDO_accre_tendency = 0.0;
                    qr_HDO_evap_tendency  = 0.0;

                    // iso_tendencies calculations
                    // the autoconversion and accretion processes are isotope non-fractionational
                    sb_iso_rain_autoconversion(ql_tmp, ql_O18_tmp, qr_tendency_au, &qr_O18_auto_tendency);
                    sb_iso_rain_autoconversion(ql_tmp, ql_HDO_tmp, qr_tendency_au, &qr_HDO_auto_tendency);
                    sb_iso_rain_accretion(ql_tmp, ql_O18_tmp, qr_tendency_ac, &qr_O18_accre_tendency);
                    sb_iso_rain_accretion(ql_tmp, ql_HDO_tmp, qr_tendency_ac, &qr_HDO_accre_tendency);

                    double diff_O18 = DVAPOR*DIFF_O18_RATIO;
                    double diff_HDO = DVAPOR*DIFF_HDO_RATIO;
                    double g_therm_O18 = microphysics_g_iso_SB_Liquid(LT, lam_fp, L_fp, 1.0, temperature[ijk], p0[k], 
                            qr_tmp, qr_O18_tmp, qv_tmp, qv_O18_tmp, sat_ratio, diff_O18, KT);
                    double g_therm_HDO = microphysics_g_iso_SB_Liquid(LT, lam_fp, L_fp, 2.0, temperature[ijk], p0[k], 
                            qr_tmp, qr_HDO_tmp, qv_tmp, qv_HDO_tmp, sat_ratio, diff_HDO, KT);

                    sb_iso_evaporation_rain(g_therm_O18, sat_ratio, nr_tmp, qr_tmp, mu, qr_O18_tmp, 
                            rain_mass, Dp, Dm, &qr_O18_evap_tendency);
                    sb_iso_evaporation_rain(g_therm_HDO, sat_ratio, nr_tmp, qr_tmp, mu, qr_HDO_tmp, 
                            rain_mass, Dp, Dm, &qr_HDO_evap_tendency);
                    
                    // iso_tendencies add
                    qr_O18_tendency_tmp = qr_O18_auto_tendency + qr_O18_accre_tendency + qr_O18_evap_tendency;
                    ql_O18_tendency_tmp = -qr_O18_auto_tendency - qr_O18_accre_tendency;

                    qr_HDO_tendency_tmp = qr_HDO_auto_tendency + qr_HDO_accre_tendency + qr_HDO_evap_tendency;
                    ql_HDO_tendency_tmp = -qr_HDO_auto_tendency - qr_HDO_accre_tendency;
                    
                    // Factor of 1.05 is ad-hoc
                    rate = 1.05 * ql_tendency_tmp * dt_ /(- fmax(ql_tmp,SB_EPS));
                    rate = fmax(1.05 * nr_tendency_tmp * dt_ /(-fmax(nr_tmp,SB_EPS)), rate);
                    rate = fmax(1.05 * qr_tendency_tmp * dt_ /(-fmax(qr_tmp,SB_EPS)), rate);
                    if(rate > 1.0 && iter_count < MAX_ITER){
                        //Limit the timestep, but don't allow it to become vanishingly small
                        //Don't adjust if we have reached the maximum iteration number
                        dt_ = fmax(dt_/rate, 1.0e-3);
                    }
                    //Integrate forward in time
                    ql_tmp += ql_tendency_tmp * dt_;
                    nr_tmp += nr_tendency_tmp * dt_;
                    qr_tmp += qr_tendency_tmp * dt_;
                    qv_tmp += -qr_tendency_evp * dt_;

                    qr_tmp  = fmax(qr_tmp,0.0);
                    nr_tmp  = fmax(fmin(nr_tmp, qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                    ql_tmp  = fmax(ql_tmp,0.0);
                    qt_tmp  = ql_tmp + qv_tmp;
                    time_added += dt_ ;
                    
                    // isotope tracers Intergrate forward in time
                    qr_O18_tmp += qr_O18_tendency_tmp * dt_;
                    ql_O18_tmp += ql_O18_tendency_tmp * dt_;
                    qv_O18_tmp += -qr_O18_evap_tendency * dt_;
                    qr_HDO_tmp += qr_HDO_tendency_tmp * dt_;
                    ql_HDO_tmp += ql_HDO_tendency_tmp * dt_;
                    qv_HDO_tmp += -qr_HDO_evap_tendency * dt_;

                    qr_O18_tmp  = fmax(qr_O18_tmp, 0.0);
                    ql_O18_tmp  = fmax(ql_O18_tmp, 0.0);
                    qr_HDO_tmp  = fmax(qr_HDO_tmp, 0.0);
                    ql_HDO_tmp  = fmax(ql_HDO_tmp, 0.0);

                    time_added += dt_ ;
                }while(time_added < dt);
                nr_tendency_micro[ijk]  = (nr_tmp - nr[ijk] )/dt;
                qr_tendency_micro[ijk]  = (qr_tmp - qr[ijk])/dt;
                nr_std_tendency[ijk]   += nr_tendency_micro[ijk];
                qr_std_tendency[ijk]   += qr_tendency_micro[ijk];

                qr_O18_tendency_micro[ijk]  = (qr_O18_tmp - qr_O18[ijk])/dt;
                qr_O18_tendency[ijk]       += qr_O18_tendency_micro[ijk];
                qr_HDO_tendency_micro[ijk]  = (qr_HDO_tmp - qr_HDO[ijk])/dt;
                qr_HDO_tendency[ijk]       += qr_HDO_tendency_micro[ijk];
            }
        }
    }
    return;
}

// ===========<<< iso 1_m ice scheme for wbf >>> ============

double ice_kinetic_frac_function(
        double qi_before, 
        double qi_iso, 
        double qi_after, 
        double R_v,
        double alpha_s_ice, 
        double alpha_k_ice
    ){
    double tendency_ice = qi_after - qi_before;
    double tendency_iso;

    // deposition grouth happened, kinetic fractionation
    if (tendency_ice >= 0.0){
        tendency_iso = alpha_s_ice * alpha_k_ice * tendency_ice * R_v;
    }
    // sublimation happened, no fractionation
    else{
        tendency_iso = tendency_ice*(qi_iso/qi_before);
    }
    return qi_iso + tendency_iso;
}

void iso_mix_phase_fractionation(
        const struct DimStruct *dims, struct LookupStruct *LT, 
        double (*lam_fp)(double), 
        double (*L_fp)(double, double), 
        double* restrict temperature, 
        double* restrict s,
        double* restrict p0,
        double* restrict qt_std,
        double* restrict qv_std,
        double* restrict ql_std,
        double* restrict qi_std,
        double* restrict qt_O18,
        double* restrict qv_O18,
        double* restrict ql_O18,
        double* restrict qi_O18,
        double* restrict qt_HDO,
        double* restrict qv_HDO,
        double* restrict ql_HDO,
        double* restrict qi_HDO
        ){ 

    ssize_t i,j,k;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];

    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
                for (k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    double qv_std_tmp, ql_std_tmp, qi_std_tmp;
                    double qv_O18_tmp, ql_O18_tmp, qi_O18_tmp;
                    double qv_HDO_tmp, ql_HDO_tmp, qi_HDO_tmp;
                    double t_tmp = temperature[ijk];
                    
                    double diff_O18 = DVAPOR*DIFF_O18_RATIO;
                    double diff_HDO = DVAPOR*DIFF_HDO_RATIO;
                    
                    eos_c(LT, lam_fp, L_fp, p0[k], s[ijk], qt_std[ijk], 
                        &t_tmp, &qv_std_tmp, &ql_std_tmp, &qi_std_tmp);

                    // isotope fractionation process
                    double alpha_eq_lv_O18 = equilibrium_fractionation_factor_O18_liquid(temperature[ijk]);
                    double alpha_eq_lv_HDO = equilibrium_fractionation_factor_HDO_liquid(temperature[ijk]);

                    double alpha_s_ice_O18 = equilibrium_fractionation_factor_O18_ice(temperature[ijk]);
                    double alpha_s_ice_HDO = equilibrium_fractionation_factor_HDO_ice(temperature[ijk]);

                    double alpha_k_ice_O18 = alpha_k_ice_equation_Blossey_Arc1M(LT, lam_fp, L_fp, temperature[ijk], 
                            p0[k], qt_std[ijk], alpha_s_ice_O18, DVAPOR, diff_O18);
                    double alpha_k_ice_HDO = alpha_k_ice_equation_Blossey_Arc1M(LT, lam_fp, L_fp, temperature[ijk], 
                            p0[k], qt_std[ijk], alpha_s_ice_HDO, DVAPOR, diff_HDO);

                    ql_O18[ijk] = fmax(ql_O18[ijk],0.0);
                    qi_O18[ijk] = fmax(qi_O18[ijk],0.0);
                    double qvl_O18 = qt_O18[ijk] - qi_O18[ijk];

                    qv_O18_tmp = eq_frac_function(qvl_O18, qv_std_tmp, ql_std_tmp, alpha_eq_lv_O18);
                    ql_O18_tmp = qvl_O18 - qv_O18_tmp;

                    double R_v_O18 = qv_O18_tmp/qv_std_tmp;
                    qi_O18_tmp  = ice_kinetic_frac_function(qi_std[ijk], qi_O18[ijk], qi_std_tmp, 
                            R_v_O18, alpha_s_ice_O18, alpha_k_ice_O18);


                    ql_HDO[ijk] = fmax(ql_HDO[ijk],0.0);
                    qi_HDO[ijk] = fmax(qi_HDO[ijk],0.0);
                    double qvl_HDO = qt_HDO[ijk] - qi_HDO[ijk];

                    qv_HDO_tmp = eq_frac_function(qvl_HDO, qv_std_tmp, ql_std_tmp, alpha_eq_lv_HDO);
                    ql_HDO_tmp = qvl_HDO - qv_HDO_tmp;

                    double R_v_HDO = qv_HDO_tmp/qv_std_tmp;
                    qi_HDO_tmp  = ice_kinetic_frac_function(qi_std[ijk], qi_HDO[ijk], qi_std_tmp, 
                            R_v_HDO, alpha_s_ice_HDO, alpha_k_ice_HDO);
    
                    qv_std[ijk] = qv_std_tmp;
                    ql_std[ijk] = ql_std_tmp;
                    qi_std[ijk] = qi_std_tmp;

                    qv_O18[ijk] = qv_O18_tmp;
                    qv_HDO[ijk] = qv_HDO_tmp;

                    ql_O18[ijk] = ql_O18_tmp;
                    ql_HDO[ijk] = ql_HDO_tmp;

                    qi_O18[ijk] = qi_O18_tmp;
                    qi_HDO[ijk] = qi_HDO_tmp;
                } // End k loop
            } // End j loop
        } // End i loop
    return;
}

// ===========<<< 1M tracer scheme for Arctic_1M Microphysics scheme >>> ============

void tracer_arctic1m_microphysics_sources(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double),
    double (*L_fp)(double, double), double* restrict density, double* restrict p0, double ccn, double n0_ice, double dt,
    double* restrict temperature,  double* restrict qt, double* restrict qv, double* restrict ql, double* restrict qi, 
    double* restrict qrain, double* restrict nrain, double* restrict qsnow, double* restrict nsnow, 
    double* restrict ql_std, double* restrict qi_std, double* restrict qrain_tendency_micro, double* restrict qrain_tendency,
    double* restrict qsnow_tendency_micro, double* restrict qsnow_tendency,
    double* restrict precip_rate, double* restrict evap_rate, double* restrict melt_rate,
    double* restrict qt_O18, double* restrict qv_O18, double* restrict ql_O18, double* restrict qi_O18, 
    double* restrict qrain_O18, double* restrict qsnow_O18, double* restrict qrain_O18_tendency, 
    double* restrict qrain_O18_tendency_micro, double* restrict qsnow_O18_tendency, double* restrict qsnow_O18_tendency_micro,
    double* restrict precip_O18_rate, double* restrict evap_O18_rate, 
    double* restrict qt_HDO, double* restrict qv_HDO, double* restrict ql_HDO, double* restrict qi_HDO, 
    double* restrict qrain_HDO, double* restrict qsnow_HDO, double* restrict qrain_HDO_tendency, 
    double* restrict qrain_HDO_tendency_micro, double* restrict qsnow_HDO_tendency, double* restrict qsnow_HDO_tendency_micro,
    double* restrict precip_HDO_rate, double* restrict evap_HDO_rate){

    const double b1 = 650.1466922699631;
    const double b2 = -1.222222222222222;
    const double y1 = 5.62e7;
    const double y2 = 0.63;

    double iwc, ni;
    double qrain_tendency_aut, qrain_tendency_acc, qrain_tendency_evp;
    double qsnow_tendency_aut, qsnow_tendency_acc, qsnow_tendency_evp, qsnow_tendency_melt;
    double ql_tendency_acc, qi_tendency_acc;
    double ql_tendency_tmp, qi_tendency_tmp, qrain_tendency_tmp, qsnow_tendency_tmp;
    double qt_tmp, qv_tmp, ql_tmp, qi_tmp, qrain_tmp, qsnow_tmp;
    double precip_tmp, evap_tmp;
    double qt_O18_tmp, qv_O18_tmp, ql_O18_tmp, qi_O18_tmp, qrain_O18_tmp, qsnow_O18_tmp;
    double qrain_O18_tendency_tmp, qrain_O18_tendency_aut, qrain_O18_tendency_acc, qrain_O18_tendency_evp;
    double qsnow_O18_tendency_tmp, qsnow_O18_tendency_aut, qsnow_O18_tendency_acc, qsnow_O18_tendency_evp, qsnow_O18_tendency_melt;
    double ql_O18_tendency_tmp, ql_O18_tendency_acc, qi_O18_tendency_tmp, qi_O18_tendency_acc;
    double precip_O18_tmp, evap_O18_tmp;
    double qt_HDO_tmp, qv_HDO_tmp, ql_HDO_tmp, qi_HDO_tmp, qrain_HDO_tmp, qsnow_HDO_tmp;
    double qrain_HDO_tendency_tmp, qrain_HDO_tendency_aut, qrain_HDO_tendency_acc, qrain_HDO_tendency_evp;
    double qsnow_HDO_tendency_tmp, qsnow_HDO_tendency_aut, qsnow_HDO_tendency_acc, qsnow_HDO_tendency_evp, qsnow_HDO_tendency_melt;
    double ql_HDO_tendency_tmp, ql_HDO_tendency_acc, qi_HDO_tendency_tmp, qi_HDO_tendency_acc;
    double precip_HDO_tmp, evap_HDO_tmp;
    double ql_std_tmp, qi_std_tmp, R_ql, R_qi, R_qrain, R_qsnow;

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin    = dims->gw;
    const ssize_t jmin    = dims->gw;
    const ssize_t kmin    = dims->gw;
    const ssize_t imax    = dims->nlg[0]-dims->gw;
    const ssize_t jmax    = dims->nlg[1]-dims->gw;
    const ssize_t kmax    = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                // First get number concentartion N_0 for micro species
                qi_tmp           = fmax(qi[ijk], 0.0);
                iwc              = fmax(qi_tmp * density[k], SMALL);
                ni               = fmax(fmin(n0_ice, iwc*N_MAX_ICE),iwc*N_MIN_ICE);

                qrain_tmp        = fmax(qrain[ijk],0.0); //clipping
                qsnow_tmp        = fmax(qsnow[ijk],0.0); //clipping
                qt_tmp           = qt[ijk];
                ql_tmp           = fmax(ql[ijk],0.0);
                ql_std_tmp       = fmax(ql_std[ijk],0.0);
                qi_std_tmp       = fmax(qi_std[ijk],0.0);
                qv_tmp           = qt_tmp - ql_std_tmp;

                precip_rate[ijk] = 0.0;
                evap_rate[ijk]   = 0.0;
                melt_rate[ijk]   = 0.0;

                // First get initial values of isotope tracers
                qi_O18_tmp           = fmax(qi_O18[ijk], 0.0);
                qt_O18_tmp           = qt_O18[ijk];
                qv_O18_tmp           = fmax(qv_O18[ijk], 0.0);
                qrain_O18_tmp        = fmax(qrain_O18[ijk],0.0);
                ql_O18_tmp           = fmax(ql_O18[ijk], 0.0);
                qsnow_O18_tmp        = fmax(qsnow_O18[ijk],0.0);

                precip_O18_rate[ijk] = 0.0;
                evap_O18_rate[ijk]   = 0.0;
                
                qi_HDO_tmp           = fmax(qi_HDO[ijk], 0.0);
                qt_HDO_tmp           = qt_HDO[ijk];
                qv_HDO_tmp           = fmax(qv_HDO[ijk], 0.0);
                qrain_HDO_tmp        = fmax(qrain_HDO[ijk],0.0);
                ql_HDO_tmp           = fmax(ql_HDO[ijk], 0.0);
                qsnow_HDO_tmp        = fmax(qsnow_HDO[ijk],0.0);

                precip_HDO_rate[ijk] = 0.0;
                evap_HDO_rate[ijk]   = 0.0;
                
                // Now do sub-timestepping
                double time_added = 0.0, dt_, rate;
                ssize_t iter_count = 0;
                do{
                    iter_count += 1;
                    dt_ = dt - time_added;

                    if ((ql_tmp + qi_tmp) < SMALL && (qrain_tmp + qsnow_tmp) < SMALL)
                        break;

                    qsnow_tendency_aut  = 0.0;
                    qsnow_tendency_acc  = 0.0;
                    qsnow_tendency_evp  = 0.0;
                    qsnow_tendency_melt = 0.0;

                    qrain_tendency_aut  = 0.0;
                    qrain_tendency_acc  = 0.0;
                    qrain_tendency_evp  = 0.0;

                    ql_tendency_acc     = 0.0;
                    qi_tendency_acc     = 0.0;

                    precip_tmp          = 0.0;
                    evap_tmp            = 0.0;

                    autoconversion_rain(density[k], ccn, ql_tmp, qrain_tmp, nrain[ijk], &qrain_tendency_aut);
                    autoconversion_snow(LT, lam_fp, L_fp, density[k], p0[k], temperature[ijk], qt_tmp, qi_tmp, ni, &qsnow_tendency_aut);
                    accretion_all(density[k], p0[k], temperature[ijk], ccn, ql_tmp, qi_tmp, ni,
                                  qrain_tmp, nrain[ijk], qsnow_tmp, nsnow[ijk],
                                  &ql_tendency_acc, &qi_tendency_acc, &qrain_tendency_acc, &qsnow_tendency_acc);
                    evaporation_rain(LT, lam_fp, L_fp, density[k], p0[k], temperature[ijk], qt_tmp, qrain_tmp, nrain[ijk],
                                     &qrain_tendency_evp);
                    evaporation_snow(LT, lam_fp, L_fp, density[k], p0[k], temperature[ijk], qt_tmp, qsnow_tmp,
                                     nsnow[ijk], &qsnow_tendency_evp);
                    melt_snow(density[k], temperature[ijk], qsnow_tmp, nsnow[ijk], &qsnow_tendency_melt);

                    qrain_tendency_tmp = qrain_tendency_aut + qrain_tendency_acc + qrain_tendency_evp - qsnow_tendency_melt;
                    qsnow_tendency_tmp = qsnow_tendency_aut + qsnow_tendency_acc + qsnow_tendency_evp + qsnow_tendency_melt;
                    ql_tendency_tmp    = ql_tendency_acc - qrain_tendency_aut;
                    qi_tendency_tmp    = qi_tendency_acc - qsnow_tendency_aut;

                    // ===========<<< IsotopeTracer calculation components >>> ============

                    qrain_O18_tendency_aut  = 0.0;
                    qrain_O18_tendency_acc  = 0.0;
                    qrain_O18_tendency_evp  = 0.0;

                    qsnow_O18_tendency_aut  = 0.0;
                    qsnow_O18_tendency_acc  = 0.0;
                    qsnow_O18_tendency_evp  = 0.0;
                    qsnow_O18_tendency_melt = 0.0;

                    ql_O18_tendency_acc     = 0.0;
                    qi_O18_tendency_acc     = 0.0;

                    qrain_HDO_tendency_aut  = 0.0;
                    qrain_HDO_tendency_acc  = 0.0;
                    qrain_HDO_tendency_evp  = 0.0;

                    qsnow_HDO_tendency_aut  = 0.0;
                    qsnow_HDO_tendency_acc  = 0.0;
                    qsnow_HDO_tendency_evp  = 0.0;
                    qsnow_HDO_tendency_melt = 0.0;

                    ql_HDO_tendency_acc     = 0.0;
                    qi_HDO_tendency_acc     = 0.0;

                    // the following sections are non-fractionational processes
                    arc1m_iso_autoconversion_rain(qrain_tendency_aut, ql_std_tmp, ql_O18_tmp, &qrain_O18_tendency_aut);
                    arc1m_iso_autoconversion_rain(qrain_tendency_aut, ql_std_tmp, ql_HDO_tmp, &qrain_HDO_tendency_aut);

                    arc1m_iso_autoconversion_snow(qsnow_tendency_aut, qi_std_tmp, qi_O18_tmp, &qsnow_O18_tendency_aut);
                    arc1m_iso_autoconversion_snow(qsnow_tendency_aut, qi_std_tmp, qi_HDO_tmp, &qsnow_HDO_tendency_aut);

                    arc1m_iso_accretion_all(density[k], p0[k], temperature[ijk], ccn, ql_tmp, qi_tmp, ni,
                        qrain_tmp, nrain[ijk], qsnow_tmp, nsnow[ijk],
                        ql_O18_tmp, qi_O18_tmp, qrain_O18_tmp, qsnow_O18_tmp,
                        &ql_O18_tendency_acc, &qi_O18_tendency_acc, &qrain_O18_tendency_acc, &qsnow_O18_tendency_acc);
                    arc1m_iso_accretion_all(density[k], p0[k], temperature[ijk], ccn, ql_tmp, qi_tmp, ni,
                        qrain_tmp, nrain[ijk], qsnow_tmp, nsnow[ijk],
                        ql_HDO_tmp, qi_HDO_tmp, qrain_HDO_tmp, qsnow_HDO_tmp,
                        &ql_HDO_tendency_acc, &qi_HDO_tendency_acc, &qrain_HDO_tendency_acc, &qsnow_HDO_tendency_acc);

                    // the following sections are kinetic fractionational processes 
                    // defination of thermo variables
                    double vapor_diff = vapor_diffusivity(temperature[ijk], p0[k]);
                    double therm_cond = thermal_conductivity(temperature[ijk]);
                    double diff_O18 = vapor_diff*DIFF_O18_RATIO;
                    double diff_HDO = vapor_diff*DIFF_HDO_RATIO;

                    double gtherm_O18_liq, gtherm_O18_ice, gtherm_HDO_liq, gtherm_HDO_ice;

                    double alpha_eq_lv_O18 = equilibrium_fractionation_factor_O18_liquid(temperature[ijk]);
                    double alpha_eq_lv_HDO = equilibrium_fractionation_factor_HDO_liquid(temperature[ijk]);
                    
                    double sat_ratio = microphysics_saturation_ratio(LT, temperature[ijk], p0[k], qt_tmp);

                    gtherm_O18_liq = microphysics_g_iso_Arc1M(LT, lam_fp, L_fp, temperature[ijk], p0[k], qrain_tmp, qrain_O18_tmp, 
                        qv_tmp, qv_O18_tmp, sat_ratio, alpha_eq_lv_O18, diff_O18, therm_cond);
                    arc1m_iso_evap_rain(LT, lam_fp, L_fp, density[k], p0[k], temperature[ijk], sat_ratio, qt_tmp, qv_tmp, qrain_tmp, 
                        nrain[ijk], gtherm_O18_liq, qv_O18_tmp, qrain_O18_tmp, &qrain_O18_tendency_evp);
                    
                    gtherm_HDO_liq = microphysics_g_iso_Arc1M(LT, lam_fp, L_fp, temperature[ijk], p0[k], qrain_tmp, qrain_HDO_tmp, 
                        qv_tmp, qv_HDO_tmp, sat_ratio, alpha_eq_lv_HDO, diff_HDO, therm_cond);
                    arc1m_iso_evap_rain(LT, lam_fp, L_fp, density[k], p0[k], temperature[ijk], sat_ratio, qt_tmp, qv_tmp, qrain_tmp, 
                        nrain[ijk], gtherm_HDO_liq, qv_HDO_tmp, qrain_HDO_tmp, &qrain_HDO_tendency_evp);
                    
                    // ===========<<< ice phase kinetic fractionation >>> ============

                    double alpha_s_ice_O18 = equilibrium_fractionation_factor_O18_ice(temperature[ijk]);
                    double alpha_s_ice_HDO = equilibrium_fractionation_factor_HDO_ice(temperature[ijk]);

                    double alpha_k_ice_O18 = alpha_k_ice_equation_Blossey_Arc1M(LT, lam_fp, L_fp, temperature[ijk], 
                            p0[k], qt_tmp, alpha_s_ice_O18, DVAPOR, diff_O18);
                    double alpha_k_ice_HDO = alpha_k_ice_equation_Blossey_Arc1M(LT, lam_fp, L_fp, temperature[ijk], 
                            p0[k], qt_tmp, alpha_s_ice_HDO, DVAPOR, diff_HDO);

                    arc1m_iso_evap_snow_kinetic(qsnow_tendency_evp, qsnow_tmp, qsnow_O18_tmp, 
                            qv_tmp, qv_O18_tmp, alpha_s_ice_O18, alpha_k_ice_O18, &qsnow_O18_tendency_evp);
                    arc1m_iso_evap_snow_kinetic(qsnow_tendency_evp, qsnow_tmp, qsnow_HDO_tmp, 
                            qv_tmp, qv_HDO_tmp, alpha_s_ice_HDO, alpha_k_ice_HDO, &qsnow_HDO_tendency_evp);

                    arc1m_iso_melt_snow(qsnow_tendency_melt, qsnow_tmp, qsnow_O18_tmp, &qsnow_O18_tendency_melt);
                    arc1m_iso_melt_snow(qsnow_tendency_melt, qsnow_tmp, qsnow_HDO_tmp, &qsnow_HDO_tendency_melt);
                    //
                    qrain_O18_tendency_tmp  = qrain_O18_tendency_aut + qrain_O18_tendency_acc + qrain_O18_tendency_evp - qsnow_O18_tendency_melt;
                    qsnow_O18_tendency_tmp  = qsnow_O18_tendency_aut + qsnow_O18_tendency_acc + qsnow_O18_tendency_evp + qsnow_O18_tendency_melt;
                    ql_O18_tendency_tmp     = ql_O18_tendency_acc - qrain_O18_tendency_aut;
                    qi_O18_tendency_tmp     = qi_O18_tendency_acc - qsnow_O18_tendency_aut;

                    qrain_HDO_tendency_tmp  = qrain_HDO_tendency_aut + qrain_HDO_tendency_acc + qrain_HDO_tendency_evp - qsnow_HDO_tendency_melt;
                    qsnow_HDO_tendency_tmp  = qsnow_HDO_tendency_aut + qsnow_HDO_tendency_acc + qsnow_HDO_tendency_evp + qsnow_HDO_tendency_melt;
                    ql_HDO_tendency_tmp     = ql_HDO_tendency_acc - qrain_HDO_tendency_aut;
                    qi_HDO_tendency_tmp     = qi_HDO_tendency_acc - qsnow_HDO_tendency_aut;

                    rate = 1.05 * qrain_tendency_tmp * dt_ / (-fmax(qrain_tmp, SMALL));
                    rate = fmax(1.05 * qsnow_tendency_tmp * dt_ /(-fmax(qsnow_tmp, SMALL)), rate);
                    rate = fmax(1.05 * ql_tendency_tmp * dt_ /(-fmax(ql_tmp, SMALL)), rate);
                    rate = fmax(1.05 * qi_tendency_tmp * dt_ /(-fmax(qi_tmp, SMALL)), rate);

                    if(rate > 1.0 && iter_count < MAX_ITER){
                        //Limit the timestep, but don't allow it to become vanishingly small
                        //Don't adjust if we have reached the maximum iteration number
                        dt_ = fmax(dt_/rate, 1.0e-3);
                    }

                    // precip_tmp is NEGATIVE if rain/snow forms (+precip_tmp is to remove qt via precip formation);
                    // evap_tmp is NEGATIVE if rain/snow evaporate/sublimate (-evap_tmp is to add qt via evap/subl);
                    precip_tmp        = -qrain_tendency_aut + ql_tendency_acc - qsnow_tendency_aut + qi_tendency_acc;
                    evap_tmp          = qrain_tendency_evp + qsnow_tendency_evp;
                    
                    precip_rate[ijk] += precip_tmp * dt_;
                    evap_rate[ijk]   += evap_tmp * dt_;
                    melt_rate[ijk]   += qsnow_tendency_melt * dt_; // NEGATIVE if snow melts to rain

                    // IsotopeTracer precip_rate and evap_rate source calculation
                    precip_O18_tmp = -qrain_O18_tendency_aut + ql_O18_tendency_acc - qsnow_O18_tendency_aut + qi_O18_tendency_acc;
                    evap_O18_tmp   = qrain_O18_tendency_evp + qsnow_O18_tendency_evp;

                    precip_O18_rate[ijk] += precip_O18_tmp * dt_;
                    evap_O18_rate[ijk]   += evap_O18_tmp * dt_;
                    
                    precip_HDO_tmp = -qrain_HDO_tendency_aut + ql_HDO_tendency_acc - qsnow_HDO_tendency_aut + qi_HDO_tendency_acc;
                    evap_HDO_tmp   = qrain_HDO_tendency_evp + qsnow_HDO_tendency_evp;

                    precip_HDO_rate[ijk] += precip_HDO_tmp * dt_;
                    evap_HDO_rate[ijk]   += evap_HDO_tmp * dt_;

                    //Integrate forward in time
                    ql_tmp    += ql_tendency_tmp * dt_;
                    qi_tmp    += qi_tendency_tmp * dt_;
                    qrain_tmp += qrain_tendency_tmp * dt_;
                    qsnow_tmp += qsnow_tendency_tmp * dt_;
                    qt_tmp    += (precip_tmp - evap_tmp) * dt_;

                    qrain_tmp  = fmax(qrain_tmp, 0.0);
                    qsnow_tmp  = fmax(qsnow_tmp, 0.0);
                    ql_tmp     = fmax(ql_tmp, 0.0);
                    qi_tmp     = fmax(qi_tmp, 0.0);
                    qt_tmp     = fmax(qt_tmp, 0.0);
                    double qv_ = qt_tmp - ql_tmp - qi_tmp;
                    qv_tmp     = fmax(qv_, 0.0);
                    ql_std_tmp = ql_tmp;
                    qi_std_tmp = qi_tmp;

                    // IsotopeTracer Intergrate forward in time
                    ql_O18_tmp    += ql_O18_tendency_tmp * dt_;
                    qi_O18_tmp    += qi_O18_tendency_tmp * dt_;
                    qrain_O18_tmp += qrain_O18_tendency_tmp *dt_;
                    qsnow_O18_tmp += qsnow_O18_tendency_tmp *dt_;
                    qt_O18_tmp    += (precip_O18_tmp - evap_O18_tmp) * dt_;

                    qt_O18_tmp     = fmax(qt_O18_tmp, 0.0);
                    ql_O18_tmp     = fmax(ql_O18_tmp, 0.0);
                    qi_O18_tmp     = fmax(qi_O18_tmp, 0.0);
                    qrain_O18_tmp  = fmax(qrain_O18_tmp, 0.0);
                    qsnow_O18_tmp  = fmax(qsnow_O18_tmp, 0.0);
                    double qv_O18_ = qt_O18_tmp - ql_O18_tmp - qi_O18_tmp;
                    qv_O18_tmp     = fmax(qv_O18_, 0.0);

                    ql_HDO_tmp    += ql_HDO_tendency_tmp * dt_;
                    qi_HDO_tmp    += qi_HDO_tendency_tmp * dt_;
                    qrain_HDO_tmp += qrain_HDO_tendency_tmp *dt_;
                    qsnow_HDO_tmp += qsnow_HDO_tendency_tmp *dt_;
                    qt_HDO_tmp    += (precip_HDO_tmp - evap_HDO_tmp) * dt_;

                    qt_HDO_tmp     = fmax(qt_HDO_tmp, 0.0);
                    ql_HDO_tmp     = fmax(ql_HDO_tmp, 0.0);
                    qi_HDO_tmp     = fmax(qi_HDO_tmp, 0.0);
                    qrain_HDO_tmp  = fmax(qrain_HDO_tmp, 0.0);
                    qsnow_HDO_tmp  = fmax(qsnow_HDO_tmp, 0.0);
                    double qv_HDO_ = qt_HDO_tmp - ql_HDO_tmp - qi_HDO_tmp;
                    qv_HDO_tmp     = fmax(qv_HDO_, 0.0);

                    time_added += dt_;
                    }while(time_added < dt && iter_count < MAX_ITER);

                qrain_tendency_micro[ijk]  = (qrain_tmp - qrain[ijk])/dt;
                qrain_tendency[ijk]       += qrain_tendency_micro[ijk];
                qsnow_tendency_micro[ijk]  = (qsnow_tmp - qsnow[ijk])/dt;
                qsnow_tendency[ijk]       += qsnow_tendency_micro[ijk];

                precip_rate[ijk] = precip_rate[ijk]/dt;
                evap_rate[ijk]   = evap_rate[ijk]/dt;
                melt_rate[ijk]   = melt_rate[ijk]/dt;

                qrain_O18_tendency_micro[ijk]  = (qrain_O18_tmp - qrain_O18[ijk])/dt;
                qrain_O18_tendency[ijk]       += qrain_O18_tendency_micro[ijk];
                qsnow_O18_tendency_micro[ijk]  = (qsnow_O18_tmp - qsnow_O18[ijk])/dt;
                qsnow_O18_tendency[ijk]       += qsnow_O18_tendency_micro[ijk];

                precip_O18_rate[ijk] = precip_O18_rate[ijk]/dt;
                evap_O18_rate[ijk]   = evap_O18_rate[ijk]/dt;
                
                qrain_HDO_tendency_micro[ijk]  = (qrain_HDO_tmp - qrain_HDO[ijk])/dt;
                qrain_HDO_tendency[ijk]       += qrain_HDO_tendency_micro[ijk];
                qsnow_HDO_tendency_micro[ijk]  = (qsnow_HDO_tmp - qsnow_HDO[ijk])/dt;
                qsnow_HDO_tendency[ijk]       += qsnow_HDO_tendency_micro[ijk];

                precip_HDO_rate[ijk] = precip_HDO_rate[ijk]/dt;
                evap_HDO_rate[ijk]   = evap_HDO_rate[ijk]/dt;
            }
        }
    }
    return;
};

// ===========<<< SB_2M_Ice two moment microphysics scheme Coupled IsotopeTracer Scheme>>> ============

void tracer_sb_cloud_fractionation(struct DimStruct *dims, 
    struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
    double* restrict p0, 
    double IN,
    double CCN, 
    double* restrict saturation_ratio,
    double dt,
    double* restrict s, 
    double* restrict w,
    double* restrict qt, 
    double* restrict T,
    double* restrict qv, 
    double* restrict ql, 
    double* restrict nl, 
    double* restrict qi, 
    double* restrict ni,
    double* restrict qt_O18,
    double* restrict qv_O18,
    double* restrict ql_O18,
    double* restrict qi_O18,
    double* restrict qt_HDO,
    double* restrict qv_HDO,
    double* restrict ql_HDO,
    double* restrict qi_HDO,
    double* restrict ql_tendency, double* restrict nl_tendency,
    double* restrict qi_tendency, double* restrict ni_tendency,
    double* restrict ql_O18_tendency, double* restrict qi_O18_tendency,
    double* restrict ql_HDO_tendency, double* restrict qi_HDO_tendency
    ){
    
    ssize_t i,j,k;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];
    const double dzi = 1.0/dims->dx[2];

    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
                for (k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    double nl_tmp, ql_tmp, qi_tmp, ni_tmp, qv_tmp;
                    double ql_activation, nl_activation;
                    double t_tmp = T[ijk];
                    
                    ql[ijk] = fmax(ql[ijk],0.0);
                    qi[ijk] = fmax(qi[ijk],0.0);
                    qv[ijk] = qt[ijk] - ql[ijk] - qi[ijk];

                    // only update T[ijk] here
                    // TODO: it missed the CCN in here
                
                    double dS = saturation_ratio[ijk +1] - saturation_ratio[ijk];
                    sb_ccn(CCN, saturation_ratio[ijk], dS, dzi, w[ijk],
                            &ql_tendency[ijk], &nl_tendency[ijk]);
                    // sb_cloud_activation_hdcp(p0[k], qv[ijk], 
                    //         ql[ijk], nl[ijk], w[ijk], dt, saturation_ratio[ijk],
                    //         &ql_tendency[ijk], &nl_tendency[ijk]);
                    
                    //TODO: what if change qt[ijk] into qvl = qt-qi;
                    double qvl = qt[ijk] - qi[ijk];
                    eos_c(LT, lam_fp, L_fp, p0[k], s[ijk], qvl, 
                        &t_tmp, &qv_tmp, &ql_tmp, &qi_tmp);

                    ql_tendency[ijk] += (ql_tmp - ql[ijk])/dt;
                    nl_tendency[ijk] += (ql_tmp/1.0e-12 - nl[ijk])/dt;

                    // ------------ Ice particle Nucleation --------
                    double qi_tend_nuc, ni_tend_nuc;
                    sb_ice_nucleation_mayer(LT, IN,
                        t_tmp, qt[ijk], p0[k], 
                        qv[ijk], ni[ijk], dt,
                        &qi_tend_nuc, &ni_tend_nuc);
                    qi_tendency[ijk] += qi_tend_nuc;
                    ni_tendency[ijk] += ni_tend_nuc;
                   
                    // ------------ Iso O18 Computation ------------

                    double iso_type_O18 = 1.0;
                    double qv_O18_tmp, ql_O18_tmp, qi_O18_tend_nuc;

                    ql_O18[ijk] = fmax(ql_O18[ijk],0.0);
                    qi_O18[ijk] = fmax(qi_O18[ijk],0.0);
                    qv_O18[ijk] = qt_O18[ijk] - ql_O18[ijk] - qi_O18[ijk];
                    double qvl_O18 = qt_O18[ijk] - qi_O18[ijk];

                    iso_sb_2m_cloud_liquid_fraction(iso_type_O18,
                        t_tmp, qv[ijk], ql[ijk],
                        qvl_O18, qv_O18[ijk], ql_O18[ijk],
                        &qv_O18_tmp, &ql_O18_tmp);
                    
                    iso_sb_2m_cloud_ice_fraction(iso_type_O18,
                        t_tmp, qi_tend_nuc, qv[ijk], qv_O18[ijk],
                        &qi_O18_tend_nuc);
                    
                    ql_O18_tendency[ijk] += (ql_O18_tmp - ql_O18[ijk])/dt;
                    qi_O18_tendency[ijk] += qi_O18_tend_nuc;

                    // ------------ Iso HDO Computation ------------
                    double iso_type_HDO = 2.0;
                    double qv_HDO_tmp, ql_HDO_tmp, qi_HDO_tend_nuc;

                    ql_HDO[ijk] = fmax(ql_HDO[ijk],0.0);
                    qi_HDO[ijk] = fmax(qi_HDO[ijk],0.0);
                    qv_HDO[ijk] = qt_HDO[ijk] - ql_HDO[ijk] - qi_HDO[ijk];
                    double qvl_HDO = qt_HDO[ijk] - qi_HDO[ijk];

                    iso_sb_2m_cloud_liquid_fraction(iso_type_HDO,
                        t_tmp, qv[ijk], ql[ijk],
                        qvl_HDO, qv_HDO[ijk], ql_HDO[ijk],
                        &qv_HDO_tmp, &ql_HDO_tmp);
                    
                    iso_sb_2m_cloud_ice_fraction(iso_type_HDO,
                        t_tmp, qi_tend_nuc, qv[ijk], qv_HDO[ijk],
                        &qi_HDO_tend_nuc);
                    
                    ql_HDO_tendency[ijk] += (ql_HDO_tmp - ql_HDO[ijk])/dt;
                    qi_HDO_tendency[ijk] += qi_HDO_tend_nuc;

                } // End k loop
            } // End j loop
        } // End i loop
    return;
}

void tracer_sb_ice_microphysics_sources(const struct DimStruct *dims, 
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
        double* restrict s, // specific entropy
        double* restrict w, // vertical velocity
        // INPUT STD VARIABLES ARRAY
        double* restrict qt, // total water specific humidity
        double* restrict qv, // total water specific humidity
        double* restrict nl, // cloud liquid number density
        double* restrict ql, // cloud liquid water specific humidity
        double* restrict ni, // cloud ice number density
        double* restrict qi, // cloud ice water specific humidity
        double* restrict nr, // rain droplet number density
        double* restrict qr, // rain droplet specific humidity
        double* restrict ns, // snow number density
        double* restrict qs, // snow specific humidity
        // OUTPUT STD ARRAYS: diagnose variables
        double* restrict Dm, 
        double* restrict mass,
        double* restrict diagnose_1, 
        double* restrict diagnose_2,
        double* restrict diagnose_3, 
        double* restrict diagnose_4, 
        double* restrict diagnose_5,
        // OUTPUT STD ARRAYS: q and n tendency
        double* restrict nl_tendency, double* restrict ql_tendency,
        double* restrict ni_tendency, double* restrict qi_tendency,
        double* restrict nr_tendency_micro, double* restrict qr_tendency_micro, 
        double* restrict nr_tendency, double* restrict qr_tendency, 
        double* restrict ns_tendency_micro, double* restrict qs_tendency_micro, 
        double* restrict ns_tendency, double* restrict qs_tendency,
        double* restrict precip_rate, double* restrict evap_rate, double* restrict melt_rate,
        // INPUT ISO VARIABLES ARRAY:
        double* restrict qt_O18, // total water specific humidity of O18
        double* restrict qv_O18, // total water specific humidity of O18
        double* restrict ql_O18, // cloud liquid water specific humidity of O18
        double* restrict qi_O18, // cloud ice water specific humidity of O18
        double* restrict qs_O18, // snow specific humidity of O18
        double* restrict qr_O18, // rain droplet specific humidity of O18
        double* restrict qt_HDO, // total water specific humidity of HDO
        double* restrict qv_HDO, // total water specific humidity of HDO
        double* restrict ql_HDO, // cloud liquid water specific humidity of HDO
        double* restrict qi_HDO, // cloud ice water specific humidity of HDO
        double* restrict qs_HDO, // snow specific humidity of HDO
        double* restrict qr_HDO, // rain droplet specific humidity of HDO
        // OUTPUT ISO VARIABLES TENDENCY 
        double* restrict ql_O18_tendency,
        double* restrict qi_O18_tendency,
        double* restrict qs_O18_tend,
        double* restrict qr_O18_tend,
        double* restrict qs_O18_tendency,
        double* restrict qr_O18_tendency,
        double* restrict precip_O18_rate,
        double* restrict evap_O18_rate,
        double* restrict melt_O18_rate,
        double* restrict ql_HDO_tendency,
        double* restrict qi_HDO_tendency,
        double* restrict qs_HDO_tend,
        double* restrict qr_HDO_tend,
        double* restrict qs_HDO_tendency,
        double* restrict qr_HDO_tendency,
        double* restrict precip_HDO_rate,
        double* restrict evap_HDO_rate,
        double* restrict melt_HDO_rate
    ){

    //Here we compute the source terms for nr, qr and ns, qs (number and mass of rain and snow)
    //Temporal substepping is used to help ensure boundedness of moments

    // vapor tendency in loop
    double qv_tendency_tmp;
    
    // ---------Warm Process Tendency------------------
    double nr_tendency_tmp, qr_tendency_tmp, ql_tendency_tmp, nl_tendency_tmp;

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

    // ============ Define IsotopeTracer Tmp Tendency ==============
    double qv_O18_tendency_tmp, ql_O18_tendency_tmp, qi_O18_tendency_tmp, 
           qr_O18_tendency_tmp, qs_O18_tendency_tmp;
    // liquid cloud O18 tendency
    double ql_O18_tendency_au, ql_O18_tendency_ac, ql_O18_tendency_rime;
    // ice cloud O18 tendency
    double qi_O18_tendency_dep, qi_O18_tendency_col, qi_O18_tendency_sub;
    double qi_O18_tendency_frz;
    // rain O18 tendency
    double qr_O18_tendency_au, qr_O18_tendency_ac, qr_O18_tendency_rime, 
           qr_O18_tendency_evap, qr_O18_tendency_melt;
    // snow O18 tendency
    double qs_O18_tendency_col, qs_O18_tendency_rime, qs_O18_tendency_dep, 
           qs_O18_tendency_sub, qs_O18_tendency_melt;

    double qv_HDO_tendency_tmp, ql_HDO_tendency_tmp, qi_HDO_tendency_tmp, 
           qr_HDO_tendency_tmp, qs_HDO_tendency_tmp;
    // liquid cloud HDO tendency
    double ql_HDO_tendency_au, ql_HDO_tendency_ac, ql_HDO_tendency_rime;
    // ice cloud HDO tendency
    double qi_HDO_tendency_dep, qi_HDO_tendency_col, qi_HDO_tendency_sub;
    double qi_HDO_tendency_frz;
    // rain HDO tendency
    double qr_HDO_tendency_au, qr_HDO_tendency_ac, qr_HDO_tendency_rime, 
           qr_HDO_tendency_evap, qr_HDO_tendency_melt;
    // snow HDO tendency
    double qs_HDO_tendency_col, qs_HDO_tendency_rime, qs_HDO_tendency_dep, 
           qs_HDO_tendency_sub, qs_HDO_tendency_melt;

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

                double dep_tend_ice, sub_tend_ice;
                double dep_tend_snow, sub_tend_snow;
                
                // isotope input variable settings
                
                double qt_O18_tmp = fmax(qt_O18[ijk],0.0);
                double qv_O18_tmp = fmax(qv_O18[ijk],0.0);
                double ql_O18_tmp = fmax(ql_O18[ijk],0.0);
                double qi_O18_tmp = fmax(qi_O18[ijk],0.0);
                double qvl_O18_tmp = qt_O18_tmp - qi_O18_tmp;
                double qr_O18_tmp = fmax(qr_O18[ijk],0.0);
                double qs_O18_tmp = fmax(qs_O18[ijk],0.0);
                double qt_HDO_tmp = fmax(qt_HDO[ijk],0.0);
                double qv_HDO_tmp = fmax(qv_HDO[ijk],0.0);
                double ql_HDO_tmp = fmax(ql_HDO[ijk],0.0);
                double qi_HDO_tmp = fmax(qi_HDO[ijk],0.0);
                double qvl_HDO_tmp = qt_HDO_tmp - qi_HDO_tmp;
                double qr_HDO_tmp = fmax(qr_HDO[ijk],0.0);
                double qs_HDO_tmp = fmax(qs_HDO[ijk],0.0);

                double precip_O18_tmp = 0.0;
                double precip_O18_tend = 0.0;
                double evap_O18_tmp = 0.0;
                double evap_O18_tend = 0.0;
                double melt_O18_tmp= 0.0;
                
                double precip_HDO_tmp = 0.0;
                double precip_HDO_tend = 0.0;
                double evap_HDO_tmp = 0.0;
                double evap_HDO_tend = 0.0;
                double melt_HDO_tmp= 0.0;

                do{
                    iter_count += 1;
                    // sat_ratio   = microphysics_saturation_ratio(LT, temperature[ijk], p0[k], qt_tmp);
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

                    dep_tend_ice = 0.0;
                    sub_tend_ice = 0.0;
                    dep_tend_snow = 0.0;
                    sub_tend_snow = 0.0;

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
                    
                    // compute the source terms of warm phase process: rain
                    sb_autoconversion_rain_tmp(droplet_nu, density[k], nl_tmp, ql_tmp, qr_tmp, 
                            &nr_tendency_au, &qr_tendency_au, &nl_tendency_au, &ql_tendency_au);
                    sb_accretion_rain_tmp(density[k], ql_tmp, qr_tmp, liquid_mass,
                            &qr_tendency_ac, &ql_tendency_ac, &nl_tendency_ac);
                    sb_selfcollection_breakup_rain(density[k], nr_tmp, qr_tmp, mu, rain_mass, Dm_r, 
                            &nr_tendency_scbk);
                    sb_evaporation_rain_tmp(g_therm_rain, sat_ratio_liq, nr_tmp, qr_tmp, 
                            mu, rain_mass, Dp, Dm_r, 
                            &nr_tendency_evp, &qr_tendency_evp, &qv_tendency_evp);

                    sb_ice_deposition(LT, lam_fp, L_fp, 
                            temperature[ijk], qt_tmp, p0[k], qi_tmp, ni_tmp, 
                            Dm_i, ice_mass, velocity_ice, dt_, sat_ratio_ice,
                            &qi_tendency_dep, &ni_tendency_dep,
                            &dep_tend_ice, &sub_tend_ice,
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
                            &dep_tend_snow, &sub_tend_snow,
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
                    ns_tendency_tmp = ns_tendency_ice_selcol + ns_tendency_snow_selcol + 
                                      ns_tendency_rime + ns_tendency_dep + ns_tendency_melt;
                    qs_tendency_tmp = qs_tendency_ice_selcol + qs_tendency_si_col + 
                                      qs_tendency_rime + qs_tendency_dep + qs_tendency_melt;
                    
                    // cloud droplet tendency sum
                    ql_tendency_tmp = ql_tendency_au + ql_tendency_ac + ql_tendency_snow_rime - ql_tendency_frz;
                    nl_tendency_tmp = nl_tendency_au + nl_tendency_ac + nl_tendency_snow_rime - nl_tendency_frz;

                    // ice particle tendency sum
                    qi_tendency_tmp = qi_tendency_dep + qi_tendency_ice_selcol + qi_tendency_si_col + 
                                      qi_tendency_snow_mult + ql_tendency_frz + qr_tendency_frz;
                    ni_tendency_tmp = ni_tendency_dep + ni_tendency_ice_selcol + ni_tendency_si_col + 
                                      ni_tendency_snow_mult + nl_tendency_frz + nr_tendency_frz;

                    // ================ Isotope Tracer Section ======================
                    //
                    // ------------ O18 calculation -------------
                    qv_O18_tendency_tmp  = 0.0;
                    ql_O18_tendency_tmp  = 0.0;
                    qi_O18_tendency_tmp  = 0.0;
                    qr_O18_tendency_tmp  = 0.0;
                    qs_O18_tendency_tmp  = 0.0;
                    ql_O18_tendency_au   = 0.0;
                    ql_O18_tendency_ac   = 0.0;
                    ql_O18_tendency_rime = 0.0;
                    qi_O18_tendency_dep  = 0.0;
                    qi_O18_tendency_col  = 0.0;
                    qi_O18_tendency_sub = 0.0;
                    qi_O18_tendency_frz = 0.0;
                    qr_O18_tendency_au   = 0.0;
                    qr_O18_tendency_ac   = 0.0;
                    qr_O18_tendency_rime = 0.0;
                    qr_O18_tendency_evap = 0.0;
                    qr_O18_tendency_melt = 0.0;
                    qs_O18_tendency_col  = 0.0;
                    qs_O18_tendency_rime = 0.0;
                    qs_O18_tendency_dep  = 0.0;
                    qs_O18_tendency_sub  = 0.0;
                    qs_O18_tendency_melt = 0.0;
                    
                    double diff_O18 = DVAPOR*DIFF_O18_RATIO;
                    double iso_type_O18 = 1.0; // 1.0 means O18;

                    // ice deposition and sublimation
                    iso_sb_2m_depostion(LT, lam_fp, L_fp, iso_type_O18,
                            temperature[ijk], p0[k], qt_tmp, qv_tmp, qv_O18_tmp,
                            DVAPOR, diff_O18, dep_tend_ice, 
                            &qv_O18_tendency_tmp, &qi_O18_tendency_dep);
                    iso_sb_2m_sublimation(qi_tmp, qi_O18_tmp, sub_tend_ice, 
                            &qv_O18_tendency_tmp, &qi_O18_tendency_sub);

                    sb_iso_rain_autoconversion(ql_tmp, ql_O18_tmp, qr_tendency_au, 
                            &qr_O18_tendency_au);
                    sb_iso_rain_accretion(ql_tmp, ql_O18_tmp, qr_tendency_ac, 
                            &qr_O18_tendency_ac);
                    ql_O18_tendency_au = -qr_O18_tendency_au;
                    ql_O18_tendency_au = -qr_O18_tendency_ac;

                    double g_therm_O18 = microphysics_g_iso_SB_Liquid(LT, lam_fp, L_fp, 
                            iso_type_O18,
                            temperature[ijk], p0[k], qr_tmp, qr_O18_tmp, 
                            qv_tmp, qv_O18_tmp, sat_ratio_liq, diff_O18, KT);
                    sb_iso_evaporation_rain(g_therm_O18, sat_ratio_liq, nr_tmp, qr_tmp, 
                            mu, qr_O18_tmp, rain_mass, Dp, Dm_r, &qr_O18_tendency_evap);

                    sb_iso_ice_collection_snow(qi_tmp, qi_O18_tmp, 
                            qs_tendency_ice_selcol, qs_tendency_si_col,
                            &qs_O18_tendency_col, &qi_O18_tendency_col);

                    sb_iso_frz_ice(ql_tmp, ql_O18_tmp, qr_tmp, qr_O18_tmp,
                            ql_tendency_frz, qr_tendency_frz, 
                            &qi_O18_tendency_frz);

                    sb_iso_riming_snow(ql_tmp, qr_tmp, ql_O18_tmp, qr_O18_tmp, 
                            ql_tendency_snow_rime, qr_tendency_snow_rime,
                            &ql_O18_tendency_rime, &qr_O18_tendency_rime,
                            &qs_O18_tendency_rime);

                    sb_iso_melt_snow(qs_tmp, qs_O18_tmp, qs_tendency_melt,
                            &qr_O18_tendency_melt, &qs_O18_tendency_melt);
                    
                    // snow deposition and sublimation
                    iso_sb_2m_depostion(LT, lam_fp, L_fp, iso_type_O18,
                            temperature[ijk], p0[k], qt_tmp, qv_tmp, qv_O18_tmp,
                            DVAPOR, diff_O18, dep_tend_snow, 
                            &qv_O18_tendency_tmp, &qs_O18_tendency_dep);
                    iso_sb_2m_sublimation(qs_tmp, qs_O18_tmp, sub_tend_snow, 
                            &qv_O18_tendency_tmp, &qs_O18_tendency_sub);

                    ql_O18_tendency_tmp = ql_O18_tendency_au + ql_O18_tendency_ac + ql_O18_tendency_rime;
                    qi_O18_tendency_tmp = qi_O18_tendency_dep + qi_O18_tendency_col + qi_O18_tendency_sub +
                                          qi_O18_tendency_frz;
                    qr_O18_tendency_tmp = qr_O18_tendency_au + qr_O18_tendency_ac + 
                                          qr_O18_tendency_rime + qr_O18_tendency_melt + qr_O18_tendency_evap;
                    qs_O18_tendency_tmp = qs_O18_tendency_col + qs_O18_tendency_rime + qs_O18_tendency_melt +
                                          qs_O18_tendency_dep;
                    qv_O18_tendency_tmp += -qr_O18_tendency_evap;


                    // ------------ HDO calculation -------------
                    qv_HDO_tendency_tmp  = 0.0;
                    ql_HDO_tendency_tmp  = 0.0;
                    qi_HDO_tendency_tmp  = 0.0;
                    qr_HDO_tendency_tmp  = 0.0;
                    qs_HDO_tendency_tmp  = 0.0;
                    ql_HDO_tendency_au   = 0.0;
                    ql_HDO_tendency_ac   = 0.0;
                    ql_HDO_tendency_rime = 0.0;
                    qi_HDO_tendency_dep  = 0.0;
                    qi_HDO_tendency_col  = 0.0;
                    qi_HDO_tendency_sub = 0.0;
                    qi_HDO_tendency_frz = 0.0;
                    qr_HDO_tendency_au   = 0.0;
                    qr_HDO_tendency_ac   = 0.0;
                    qr_HDO_tendency_rime = 0.0;
                    qr_HDO_tendency_evap = 0.0;
                    qr_HDO_tendency_melt = 0.0;
                    qs_HDO_tendency_col  = 0.0;
                    qs_HDO_tendency_rime = 0.0;
                    qs_HDO_tendency_dep  = 0.0;
                    qs_HDO_tendency_sub  = 0.0;
                    qs_HDO_tendency_melt = 0.0;
                    

                    double diff_HDO = DVAPOR*DIFF_HDO_RATIO;
                    double iso_type_HDO = 2.0;

                    // ice deposition and sublimation
                    iso_sb_2m_depostion(LT, lam_fp, L_fp, iso_type_HDO, 
                            temperature[ijk], p0[k], qt_tmp, qv_tmp, qv_HDO_tmp,
                            DVAPOR, diff_HDO, dep_tend_ice, 
                            &qv_HDO_tendency_tmp, &qi_HDO_tendency_dep);
                    iso_sb_2m_sublimation(qi_tmp, qi_HDO_tmp, sub_tend_ice, 
                            &qv_HDO_tendency_tmp, &qi_HDO_tendency_sub);

                    sb_iso_rain_autoconversion(ql_tmp, ql_HDO_tmp, qr_tendency_au, 
                            &qr_HDO_tendency_au);
                    sb_iso_rain_accretion(ql_tmp, ql_HDO_tmp, qr_tendency_ac, 
                            &qr_HDO_tendency_ac);
                    ql_HDO_tendency_au = -qr_HDO_tendency_au;
                    ql_HDO_tendency_au = -qr_HDO_tendency_ac;

                    double g_therm_HDO = microphysics_g_iso_SB_Liquid(LT, lam_fp, L_fp,
                            iso_type_HDO,
                            temperature[ijk], p0[k], qr_tmp, qr_HDO_tmp, 
                            qv_tmp, qv_HDO_tmp, sat_ratio_liq, diff_HDO, KT);
                    sb_iso_evaporation_rain(g_therm_HDO, sat_ratio_liq, nr_tmp, qr_tmp, 
                            mu, qr_HDO_tmp, rain_mass, Dp, Dm_r, &qr_HDO_tendency_evap);

                    sb_iso_ice_collection_snow(qi_tmp, qi_HDO_tmp, 
                            qs_tendency_ice_selcol, qs_tendency_si_col,
                            &qs_HDO_tendency_col, &qi_HDO_tendency_col);
                    
                    sb_iso_frz_ice(ql_tmp, ql_HDO_tmp, qr_tmp, qr_HDO_tmp,
                            ql_tendency_frz, qr_tendency_frz, 
                            &qi_HDO_tendency_frz);

                    sb_iso_riming_snow(ql_tmp, qr_tmp, ql_HDO_tmp, qr_HDO_tmp, 
                            ql_tendency_snow_rime, qr_tendency_snow_rime,
                            &ql_HDO_tendency_rime, &qr_HDO_tendency_rime,
                            &qs_HDO_tendency_rime);

                    sb_iso_melt_snow(qs_tmp, qs_HDO_tmp, qs_tendency_melt,
                            &qr_HDO_tendency_melt, &qs_HDO_tendency_melt);
                    
                    // snow deposition and sublimation
                    iso_sb_2m_depostion(LT, lam_fp, L_fp, iso_type_HDO,
                            temperature[ijk], p0[k], qt_tmp, qv_tmp, qv_HDO_tmp,
                            DVAPOR, diff_HDO, dep_tend_snow, 
                            &qv_HDO_tendency_tmp, &qs_HDO_tendency_dep);
                    iso_sb_2m_sublimation(qs_tmp, qs_HDO_tmp, sub_tend_snow, 
                            &qv_HDO_tendency_tmp, &qs_HDO_tendency_sub);
                    
                    ql_HDO_tendency_tmp = ql_HDO_tendency_au + ql_HDO_tendency_ac + ql_HDO_tendency_rime;
                    qi_HDO_tendency_tmp = qi_HDO_tendency_dep + qi_HDO_tendency_col + qi_HDO_tendency_sub +
                                          qi_HDO_tendency_frz;
                    qr_HDO_tendency_tmp = qr_HDO_tendency_au + qr_HDO_tendency_ac + qr_HDO_tendency_rime +
                                          qr_HDO_tendency_melt + qr_HDO_tendency_evap;
                    qs_HDO_tendency_tmp = qs_HDO_tendency_col + qs_HDO_tendency_rime + qs_HDO_tendency_melt +
                                          qs_HDO_tendency_dep;
                    qv_HDO_tendency_tmp = -qr_HDO_tendency_evap;
                    // -----------------------------------------------------------------------

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
                    
                    // keep POSITIVE when precipation formed
                    precip_tend = qr_tendency_au + qr_tendency_ac + qs_tendency_rime +
                                 qs_tendency_ice_selcol + qs_tendency_si_col + dep_tend_ice + dep_tend_snow; 
                    // keep POSITIVE when evap/sub formed
                    evap_tend = -(sub_tend_ice + sub_tend_snow + qv_tendency_evp); 
                    
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

                    // ========= Isotope Intergrate forward in time =======
                    
                    // keep POSITIVE when precipation formed
                    precip_O18_tend = qr_O18_tendency_au + qr_O18_tendency_ac +
                                      qs_O18_tendency_col + qs_O18_tendency_rime + qs_O18_tendency_dep;
                                 
                    // keep POSITIVE when evap/sub formed
                    evap_O18_tend = -(qs_O18_tendency_sub - qr_O18_tendency_evap); 
                    
                    precip_O18_tmp += precip_O18_tend * dt_;
                    evap_O18_tmp   += evap_O18_tend * dt_;
                    melt_O18_tmp   += qr_O18_tendency_melt * dt_;

                    qv_O18_tmp += qv_O18_tendency_tmp * dt_;
                    ql_O18_tmp += ql_O18_tendency_tmp * dt_;
                    qi_O18_tmp += qi_O18_tendency_tmp * dt_;
                    qr_O18_tmp += qr_O18_tendency_tmp * dt_;
                    qs_O18_tmp += qs_O18_tendency_tmp * dt_;
                    qt_O18_tmp = fmax(qt_O18_tmp,0.0);
                    qv_O18_tmp = fmax(qv_O18_tmp,0.0);
                    ql_O18_tmp = fmax(ql_O18_tmp,0.0);
                    qi_O18_tmp = fmax(qi_O18_tmp,0.0);
                    qr_O18_tmp = fmax(qr_O18_tmp,0.0);
                    qs_O18_tmp = fmax(qs_O18_tmp,0.0);
                    
                    // keep POSITIVE when precipation formed
                    precip_HDO_tend = qr_HDO_tendency_au + qr_HDO_tendency_ac +
                                      qs_HDO_tendency_col + qs_HDO_tendency_rime + qs_HDO_tendency_dep;
                                 
                    // keep POSITIVE when evap/sub formed
                    evap_HDO_tend = -(qs_HDO_tendency_sub - qr_HDO_tendency_evap); 
                    
                    precip_HDO_tmp += precip_HDO_tend * dt_;
                    evap_HDO_tmp   += evap_HDO_tend * dt_;
                    melt_HDO_tmp   += qr_HDO_tendency_melt * dt_;

                    qv_HDO_tmp += qv_HDO_tendency_tmp * dt_;
                    ql_HDO_tmp += ql_HDO_tendency_tmp * dt_;
                    qi_HDO_tmp += qi_HDO_tendency_tmp * dt_;
                    qr_HDO_tmp += qr_HDO_tendency_tmp * dt_;
                    qs_HDO_tmp += qs_HDO_tendency_tmp * dt_;
                    qt_HDO_tmp = fmax(qt_HDO_tmp,0.0);
                    qv_HDO_tmp = fmax(qv_HDO_tmp,0.0);
                    ql_HDO_tmp = fmax(ql_HDO_tmp,0.0);
                    qi_HDO_tmp = fmax(qi_HDO_tmp,0.0);
                    qr_HDO_tmp = fmax(qr_HDO_tmp,0.0);
                    qs_HDO_tmp = fmax(qs_HDO_tmp,0.0);

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
                Dm[ijk] = dep_tend_ice;
                mass[ijk] = sub_tend_ice;
                diagnose_1[ijk] = qi_O18_tendency_dep;
                diagnose_2[ijk] = qi_O18_tendency_sub;
                diagnose_3[ijk] = qi_HDO_tendency_dep;
                diagnose_4[ijk] = qi_HDO_tendency_sub;

                precip_rate[ijk] = precip_tmp/dt;
                evap_rate[ijk]   = evap_tmp/dt;
                melt_rate[ijk]   = melt_tmp/dt;

                // isotope tracer tendency calculation
                ql_O18_tendency[ijk] += (ql_O18_tmp - ql_O18[ijk])/dt;
                qi_O18_tendency[ijk] += (qi_O18_tmp - qi_O18[ijk])/dt;
                qr_O18_tend[ijk] += (qr_O18_tmp - qr_O18[ijk])/dt;
                qs_O18_tend[ijk] += (qs_O18_tmp - qs_O18[ijk])/dt;
                qr_O18_tendency[ijk] += qr_O18_tend[ijk];
                qs_O18_tendency[ijk] += qs_O18_tend[ijk];
                precip_O18_rate[ijk] = precip_O18_tmp/dt;
                evap_O18_rate[ijk]   = evap_O18_tmp/dt;
                melt_O18_rate[ijk]   = melt_O18_tmp/dt;

                ql_HDO_tendency[ijk] += (ql_HDO_tmp - ql_HDO[ijk])/dt;
                qi_HDO_tendency[ijk] += (qi_HDO_tmp - qi_HDO[ijk])/dt;
                qr_HDO_tend[ijk] += (qr_HDO_tmp - qr_HDO[ijk])/dt;
                qs_HDO_tend[ijk] += (qs_HDO_tmp - qs_HDO[ijk])/dt;
                qr_HDO_tendency[ijk] += qr_HDO_tend[ijk];
                qs_HDO_tendency[ijk] += qs_HDO_tend[ijk];
                precip_HDO_rate[ijk] = precip_HDO_tmp/dt;
                evap_HDO_rate[ijk]   = evap_HDO_tmp/dt;
                melt_HDO_rate[ijk]   = melt_HDO_tmp/dt;
            }
        }
    }
    return;
}

// ================ To Facilitate Output ================

void sb_iso_rain_evaporation_wrapper(
        const struct DimStruct *dims, 
        struct LookupStruct *LT, 
        double (*lam_fp)(double), 
        double (*L_fp)(double, double),
        double (*rain_mu)(double,double,double), 
        double (*droplet_nu)(double,double),
        double* restrict density, 
        double* restrict p0, 
        double* restrict temperature,
        double* restrict qt,
        double* restrict qv,
        double* restrict qr,
        double* restrict nr,
        double* restrict qv_O18, 
        double* restrict qr_O18,
        double* restrict qv_HDO, 
        double* restrict qr_HDO,
        double* restrict dpfv_stats,
        double* restrict Dp_stats,
        double* restrict g_therm_stats,
        double* restrict qr_tend_evap, 
        double* restrict nr_tend_evap, 
        double* restrict qr_O18_tend_evap, 
        double* restrict qr_HDO_tend_evap
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

                double qv_tmp = qv[ijk];
                double qt_tmp = qt[ijk];
                double qr_tmp = fmax(qr[ijk],0.0);
                double nr_tmp = fmax(fmin(nr[ijk], qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                double g_therm = microphysics_g(LT, lam_fp, L_fp, temperature[ijk]);
                
                double qr_O18_tmp = fmax(qr_O18[ijk], 0.0);
                double qv_O18_tmp = qv_O18[ijk];

                double qr_HDO_tmp = fmax(qr_HDO[ijk], 0.0);
                double qv_HDO_tmp = qv_HDO[ijk];
                    
                //obtain some parameters
                double rain_mass = microphysics_mean_mass(nr_tmp, qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS);
                double Dm = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi);
                double mu = rain_mu(density[k], qr_tmp, Dm);
                double Dp = sb_Dp(Dm, mu);
                double sat_ratio = microphysics_saturation_ratio(LT, temperature[ijk], p0[k], qt_tmp);

                double nr_tendency_evp, qr_tendency_evp, qr_O18_evap_tendency, qr_HDO_evap_tendency;

                sb_evaporation_rain_debug(LT, lam_fp, L_fp, temperature[ijk], sat_ratio, nr_tmp, qr_tmp, mu, 
                            rain_mass, Dp, Dm, &dpfv_stats[ijk], &nr_tendency_evp, &qr_tendency_evp);
                double diff_O18 = DVAPOR*DIFF_O18_RATIO;
                double diff_HDO = DVAPOR*DIFF_HDO_RATIO;
                double g_therm_O18 = microphysics_g_iso_SB_Liquid(LT, lam_fp, L_fp, 1.0, temperature[ijk], 
                        p0[k], qr_tmp, qr_O18_tmp, qv_tmp, qv_O18_tmp, sat_ratio, diff_O18, KT);
                double g_therm_HDO = microphysics_g_iso_SB_Liquid(LT, lam_fp, L_fp, 2.0, temperature[ijk], 
                        p0[k], qr_tmp, qr_HDO_tmp, qv_tmp, qv_HDO_tmp, sat_ratio, diff_HDO, KT);

                sb_iso_evaporation_rain(g_therm_O18, sat_ratio, nr_tmp, qr_tmp, mu, qr_O18_tmp, 
                        rain_mass, Dp, Dm, &qr_O18_evap_tendency);
                sb_iso_evaporation_rain(g_therm_HDO, sat_ratio, nr_tmp, qr_tmp, mu, qr_HDO_tmp, 
                        rain_mass, Dp, Dm, &qr_HDO_evap_tendency);

                qr_tend_evap[ijk] = qr_tendency_evp;
                nr_tend_evap[ijk] = nr_tendency_evp;
                qr_O18_tend_evap[ijk] = qr_O18_evap_tendency;
                qr_HDO_tend_evap[ijk] = qr_HDO_evap_tendency;
                Dp_stats[ijk] = Dp;
                g_therm_stats[ijk] = g_therm;
            }
        }
    }
    return;
}

void sb_iso_ice_nucleation_wrapper(
        const struct DimStruct *dims, 
        struct LookupStruct *LT,
        double ice_in,
        double* restrict temperature, 
        double* restrict qt,
        double* restrict p0,
        double* restrict qv,
        double* restrict ni,
        double* restrict qv_O18,
        double* restrict qv_HDO,
        double dt,
        double* restrict qi_tendency,
        double* restrict qi_O18_tendency,
        double* restrict qi_HDO_tendency){

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
                double qi_tend_nuc, ni_tend_nuc;

                    sb_ice_nucleation_mayer(LT, ice_in,
                        temperature[ijk], qt[ijk], p0[k], 
                        qv[ijk], ni[ijk], dt,
                        &qi_tend_nuc, &ni_tend_nuc);
                    qi_tendency[ijk] = qi_tend_nuc;
                    // O18 ice Nucleation tend
                    iso_sb_2m_cloud_ice_fraction(1.0,
                        temperature[ijk], qi_tend_nuc, qv[ijk], qv_O18[ijk],
                        &qi_O18_tendency[ijk]);
                    // HDO ice Nucleation tend
                    iso_sb_2m_cloud_ice_fraction(2.0,
                        temperature[ijk], qi_tend_nuc, qv[ijk], qv_HDO[ijk],
                        &qi_HDO_tendency[ijk]);
            }
        }
    }
    return;
}

void sb_iso_ice_deposition_wrapper(
        const struct DimStruct *dims, 
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
        // INPUT VARIABLES
        double* restrict temperature, 
        double* restrict qt,         // total water specific humidity
        double* restrict p0,         // air pressure
        double* restrict density,
        double dt, 
        double* restrict qi,         // ice specific humidity
        double* restrict ni,         // ice number density
        double* restrict qv,
        double* restrict qv_O18,
        double* restrict qv_HDO,
        double* restrict qi_O18,         // ice specific humidity
        double* restrict qi_HDO,         // ice number density
        double* restrict qi_O18_tend_dep,
        double* restrict qi_O18_tend_sub,
        double* restrict qi_HDO_tend_dep,
        double* restrict qi_HDO_tend_sub
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
                double qi_tend, ni_tend, qi_tend_dep, qi_tend_sub, qv_tend;

                sb_ice_deposition(LT, lam_fp, L_fp, 
                    temperature[ijk], qt[ijk], p0[k], qi_tmp, ni_tmp, 
                    Dm_i, ice_mass, velocity_ice, dt, sat_ratio_ice,
                    &qi_tend, &ni_tend,
                    &qi_tend_dep, &qi_tend_sub, &qv_tend);

                double qi_O18_tmp = fmax(qi_O18[ijk],0.0);
                double qi_HDO_tmp = fmax(qi_HDO[ijk],0.0);
                double diff_O18 = DVAPOR*DIFF_O18_RATIO;
                double diff_HDO = DVAPOR*DIFF_HDO_RATIO;
                double qv_O18_tend_dep, qv_HDO_tend_dep, qv_O18_tend_sub, qv_HDO_tend_sub;

                iso_sb_2m_depostion(LT, lam_fp, L_fp, 1.0,
                    temperature[ijk], p0[k], qt[ijk], qv[ijk], qv_O18[ijk],
                    DVAPOR, diff_O18, qi_tend_dep, 
                    &qv_O18_tend_dep, &qi_O18_tend_dep[ijk]);

                iso_sb_2m_depostion(LT, lam_fp, L_fp, 2.0,
                    temperature[ijk], p0[k], qt[ijk], qv[ijk], qv_HDO[ijk],
                    DVAPOR, diff_HDO, qi_tend_dep, 
                    &qv_HDO_tend_dep, &qi_HDO_tend_dep[ijk]);

                iso_sb_2m_sublimation(qi_tmp, qi_O18_tmp, qi_tend_sub, 
                    &qv_O18_tend_sub, &qi_O18_tend_sub[ijk]);
                iso_sb_2m_sublimation(qi_tmp, qi_HDO_tmp, qi_tend_sub, 
                    &qv_HDO_tend_sub, &qi_HDO_tend_sub[ijk]);
            }
        }
    }
    return;
}

void sb_iso_snow_deposition_wrapper(
        const struct DimStruct *dims, 
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
        // INPUT VARIABLES
        double* restrict temperature, 
        double* restrict qt, 
        double* restrict p0, 
        double* restrict density,
        double dt, 
        double* restrict qs, 
        double* restrict ns, 
        double* restrict qv,
        double* restrict qv_O18, 
        double* restrict qv_HDO, 
        double* restrict qs_O18, 
        double* restrict qs_HDO, 
        double* restrict qs_O18_tend_dep,
        double* restrict qs_O18_tend_sub,
        double* restrict qs_HDO_tend_dep,
        double* restrict qs_HDO_tend_sub
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
                double qs_tend, ns_tend, qs_tend_dep, qs_tend_sub, qv_tend;
                sb_ice_deposition(LT, lam_fp, L_fp, 
                    temperature[ijk], qt[ijk], p0[k], qs_tmp, ns_tmp, 
                    Dm_s, snow_mass, velocity_snow, dt, sat_ratio_ice,
                    &qs_tend, &ns_tend, &qs_tend_dep, &qs_tend_sub, &qv_tend);

                double qs_O18_tmp = fmax(qs_O18[ijk],0.0);
                double qs_HDO_tmp = fmax(qs_HDO[ijk],0.0);
                double diff_O18 = DVAPOR*DIFF_O18_RATIO;
                double diff_HDO = DVAPOR*DIFF_HDO_RATIO;
                double qv_O18_tend_dep, qv_HDO_tend_dep, qv_O18_tend_sub, qv_HDO_tend_sub;

                iso_sb_2m_depostion(LT, lam_fp, L_fp, 1.0,
                    temperature[ijk], p0[k], qt[ijk], qv[ijk], qv_O18[ijk],
                    DVAPOR, diff_O18, qs_tend_dep, 
                    &qv_O18_tend_dep, &qs_O18_tend_dep[ijk]);

                iso_sb_2m_depostion(LT, lam_fp, L_fp, 2.0,
                    temperature[ijk], p0[k], qt[ijk], qv[ijk], qv_HDO[ijk],
                    DVAPOR, diff_HDO, qs_tend_dep, 
                    &qv_HDO_tend_dep, &qs_HDO_tend_dep[ijk]);
                
                iso_sb_2m_sublimation(qs_tmp, qs_O18_tmp, qs_tend_sub, 
                    &qv_O18_tend_sub, &qs_O18_tend_sub[ijk]);
                iso_sb_2m_sublimation(qs_tmp, qs_HDO_tmp, qs_tend_sub, 
                    &qv_HDO_tend_sub, &qs_HDO_tend_sub[ijk]);
            }
        }
    }
    return;
}

void cloud_liquid_wrapper(struct DimStruct *dims, 
    struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
    double* restrict p0, 
    double IN,
    double dt,
    double* restrict s, 
    double* restrict qt, 
    double* restrict T,
    double* restrict qv, 
    double* restrict ql, 
    // double* restrict nl, 
    double* restrict qi, 
    double* restrict ni,
    double* restrict qt_O18,
    double* restrict qv_O18,
    double* restrict ql_O18,
    double* restrict qi_O18,
    double* restrict qt_HDO,
    double* restrict qv_HDO,
    double* restrict ql_HDO,
    double* restrict qi_HDO,
    double* restrict ql_cond,
    double* restrict ql_evap,
    double* restrict ql_O18_cond,
    double* restrict ql_O18_evap,
    double* restrict ql_HDO_cond,
    double* restrict ql_HDO_evap
    // double* restrict ql_tendency, double* restrict nl_tendency,
    // double* restrict qi_tendency, double* restrict ni_tendency,
    // double* restrict ql_O18_tendency, double* restrict qi_O18_tendency,
    // double* restrict ql_HDO_tendency, double* restrict qi_HDO_tendency
    ){
    
    ssize_t i,j,k;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];


    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
                for (k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    double nl_tmp, ql_tmp, qi_tmp, ni_tmp, qv_tmp;
                    double ql_tend;
                    double t_tmp = T[ijk];
                    
                    ql[ijk] = fmax(ql[ijk],0.0);
                    qi[ijk] = fmax(qi[ijk],0.0);
                    qv[ijk] = qt[ijk] - ql[ijk] - qi[ijk];

                    // only update T[ijk] here
                    
                    double qvl = qt[ijk] - qi[ijk];
                     
                    eos_c(LT, lam_fp, L_fp, p0[k], s[ijk], qvl, 
                        &t_tmp, &qv_tmp, &ql_tmp, &qi_tmp);
                    
                    ql_tend = (ql_tmp - ql[ijk])/dt;

                    if (ql_tend > 0.0){
                        ql_cond[ijk] = ql_tend;
                    }
                    else if(ql_tend < 0.0){
                        ql_evap[ijk] = ql_tend;
                    }
                   
                    // ------------ Iso O18 Computation ------------

                    double iso_type_O18 = 1.0;
                    double qv_O18_tmp, ql_O18_tmp, ql_O18_tend;

                    ql_O18[ijk] = fmax(ql_O18[ijk],0.0);
                    qi_O18[ijk] = fmax(qi_O18[ijk],0.0);
                    qv_O18[ijk] = qt_O18[ijk] - ql_O18[ijk] - qi_O18[ijk];
                    double qvl_O18 = qt_O18[ijk] - qi_O18[ijk];

                    iso_sb_2m_cloud_liquid_fraction(iso_type_O18,
                        t_tmp, qv_tmp, ql_tmp,
                        qvl_O18, qv_O18[ijk], ql_O18[ijk],
                        &qv_O18_tmp, &ql_O18_tmp);
                    
                    ql_O18_tend = (ql_O18_tmp - ql_O18[ijk])/dt;

                    if (ql_O18_tend > 0.0){
                        ql_O18_cond[ijk] = ql_O18_tend;
                        // ql_O18_evap[ijk] = 0.0;
                    }
                    else if(ql_O18_tend < 0.0){
                        ql_O18_evap[ijk] = ql_O18_tend;
                        // ql_O18_cond[ijk] = 0.0;
                    }

                    // ------------ Iso HDO Computation ------------
                    double iso_type_HDO = 2.0;
                    double qv_HDO_tmp, ql_HDO_tmp, ql_HDO_tend;

                    ql_HDO[ijk] = fmax(ql_HDO[ijk],0.0);
                    qi_HDO[ijk] = fmax(qi_HDO[ijk],0.0);
                    qv_HDO[ijk] = qt_HDO[ijk] - ql_HDO[ijk] - qi_HDO[ijk];
                    double qvl_HDO = qt_HDO[ijk] - qi_HDO[ijk];

                    iso_sb_2m_cloud_liquid_fraction(iso_type_HDO,
                        t_tmp, qv_tmp, ql_tmp, 
                        qvl_HDO, qv_HDO[ijk], ql_HDO[ijk],
                        &qv_HDO_tmp, &ql_HDO_tmp);
                    
                    ql_HDO_tend = (ql_HDO_tmp - ql_HDO[ijk])/dt;

                    if (ql_HDO_tend > 0.0){
                        ql_HDO_cond[ijk] = ql_HDO_tend;
                    }
                    else if(ql_HDO_tend < 0.0){
                        ql_HDO_evap[ijk] = ql_HDO_tend;
                    }

                } // End k loop
            } // End j loop
        } // End i loop
    return;
}
