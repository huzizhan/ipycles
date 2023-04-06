#pragma once
#include "parameters.h"
#include "isotope_functions.h"
#include "thermodynamics_sa.h"
#include "microphysics.h"
#include "microphysics_sb.h"
#include "microphysics_sb_si.h"
#include "microphysics_sb_liquid.h"
#include "entropies.h"
#include "thermodynamic_functions.h"
#include "microphysics_arctic_1m.h"
#include <math.h>
// #define SB_EPS 1.0e-13

void iso_equilibrium_fractionation_No_Microphysics(struct DimStruct *dims, double* restrict temperature,
    double* restrict qt, double* restrict qv_DV, double* restrict ql_DV, double* restrict qv_std, double* restrict ql_std, 
    double* restrict qt_iso_O18, double* restrict qv_iso_O18, double* restrict ql_iso_O18, 
    double* restrict qt_iso_HDO, double* restrict qv_iso_HDO, double* restrict ql_iso_HDO){

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
                    double qv_std_tmp, ql_std_tmp, qv_iso_O18_tmp, ql_iso_O18_tmp, qv_iso_HDO_tmp, ql_iso_HDO_tmp;
                    double alpha_eq_lv_O18 = equilibrium_fractionation_factor_O18_liquid(temperature[ijk]);
                    double alpha_eq_lv_HDO = equilibrium_fractionation_factor_HDO_liquid(temperature[ijk]);

                    qv_std_tmp     = qv_DV[ijk];
                    qv_iso_O18_tmp = eq_frac_function(qt_iso_O18[ijk], qv_DV[ijk], ql_DV[ijk], alpha_eq_lv_O18);
                    qv_iso_HDO_tmp = eq_frac_function(qt_iso_HDO[ijk], qv_DV[ijk], ql_DV[ijk], alpha_eq_lv_HDO);

                    ql_std_tmp     = ql_DV[ijk];
                    ql_iso_O18_tmp = qt_iso_O18[ijk] - qv_iso_O18_tmp;
                    ql_iso_HDO_tmp = qt_iso_HDO[ijk] - qv_iso_HDO_tmp;
                    
                    qv_std[ijk]     = qv_std_tmp;
                    ql_std[ijk]     = ql_std_tmp;
                    qv_iso_O18[ijk] = qv_iso_O18_tmp;
                    ql_iso_O18[ijk] = ql_iso_O18_tmp;
                    qv_iso_HDO[ijk] = qv_iso_HDO_tmp;
                    ql_iso_HDO[ijk] = ql_iso_HDO_tmp;
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
    double* restrict density, double* restrict p0,  double* restrict temperature,  double* restrict qt, double ccn,
    double* restrict ql, double* restrict nr, double* restrict qr, double dt,
    double* restrict nr_tendency_micro, double* restrict qr_tendency_micro, 
    double* restrict nr_std_tendency, double* restrict qr_std_tendency,
    double* restrict qr_iso_O18, double* restrict qt_iso_O18, double* restrict qv_iso_O18, double* restrict ql_iso_O18,
    double* restrict qr_iso_HDO, double* restrict qt_iso_HDO, double* restrict qv_iso_HDO, double* restrict ql_iso_HDO,
    double* restrict qr_iso_O18_tendency_micro, double* restrict qr_iso_O18_tendency, 
    double* restrict qr_iso_HDO_tendency_micro, double* restrict qr_iso_HDO_tendency){

    //Here we compute the source terms for nr and qr (number and mass of rain)
    //Temporal substepping is used to help ensure boundedness of moments
    double rain_mass, Dm, mu, Dp, nr_tendency_tmp, qr_tendency_tmp, ql_tendency_tmp;
    double nr_tendency_au, nr_tendency_scbk, nr_tendency_evp;
    double qr_tendency_au, qr_tendency_ac,  qr_tendency_evp;
    double sat_ratio;
    double qr_iso_O18_tmp, qr_iso_O18_tend, qr_iso_O18_tendency_tmp, qt_iso_O18_tendency_tmp, qv_iso_O18_tendency_tmp, ql_iso_O18_tendency_tmp;
    double qr_iso_HDO_tmp, qr_iso_HDO_tend, qr_iso_HDO_tendency_tmp, qt_iso_HDO_tendency_tmp, qv_iso_HDO_tendency_tmp, ql_iso_HDO_tendency_tmp;
    double qr_iso_O18_auto_tendency, qr_iso_O18_accre_tendency, qr_iso_O18_evap_tendency;
    double qr_iso_HDO_auto_tendency, qr_iso_HDO_accre_tendency, qr_iso_HDO_evap_tendency;

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

                double ql_iso_O18_tmp = fmax(ql_iso_O18[ijk], SB_EPS);
                double qr_iso_O18_tmp = fmax(qr_iso_O18[ijk], SB_EPS);
                double qv_iso_O18_tmp = qv_iso_O18[ijk];

                double ql_iso_HDO_tmp = fmax(ql_iso_HDO[ijk], SB_EPS);
                double qr_iso_HDO_tmp = fmax(qr_iso_HDO[ijk], SB_EPS);
                double qv_iso_HDO_tmp = qv_iso_HDO[ijk];

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
                    sb_evaporation_rain(g_therm, sat_ratio, nr_tmp, qr_tmp, mu, rain_mass, Dp, Dm, &nr_tendency_evp, &qr_tendency_evp);
                    //find the maximum substep time
                    dt_ = dt - time_added;
                    //check the source term magnitudes
                    nr_tendency_tmp = nr_tendency_au + nr_tendency_scbk + nr_tendency_evp;
                    qr_tendency_tmp = qr_tendency_au + qr_tendency_ac + qr_tendency_evp;
                    ql_tendency_tmp = -qr_tendency_au - qr_tendency_ac;

                    // //iso_tendencies initilize
                    qr_iso_O18_auto_tendency  = 0.0;
                    qr_iso_O18_accre_tendency = 0.0;
                    qr_iso_O18_evap_tendency  = 0.0;

                    qr_iso_HDO_auto_tendency  = 0.0;
                    qr_iso_HDO_accre_tendency = 0.0;
                    qr_iso_HDO_evap_tendency  = 0.0;

                    // iso_tendencies calculations
                    // the autoconversion and accretion processes are isotope non-fractionational
                    sb_iso_rain_autoconversion(ql_tmp, ql_iso_O18_tmp, qr_tendency_au, &qr_iso_O18_auto_tendency);
                    sb_iso_rain_autoconversion(ql_tmp, ql_iso_HDO_tmp, qr_tendency_au, &qr_iso_HDO_auto_tendency);
                    sb_iso_rain_accretion(ql_tmp, ql_iso_O18_tmp, qr_tendency_ac, &qr_iso_O18_accre_tendency);
                    sb_iso_rain_accretion(ql_tmp, ql_iso_HDO_tmp, qr_tendency_ac, &qr_iso_HDO_accre_tendency);

                    double diff_O18 = DVAPOR*0.9723;
                    // ================================================
                    // ToDo: give the defination of diff_HDO based on the actual physical value of the diffusivity of HDO
                    // ================================================
                    double diff_HDO = DVAPOR*0.9723;
                    double g_therm_iso_O18 = microphysics_g_iso_SB_Liquid(LT, lam_fp, L_fp, temperature[ijk], p0[k], qr_tmp, qr_iso_O18_tmp, qv_tmp, qv_iso_O18_tmp, sat_ratio, diff_O18, KT);
                    double g_therm_iso_HDO = microphysics_g_iso_SB_Liquid(LT, lam_fp, L_fp, temperature[ijk], p0[k], qr_tmp, qr_iso_HDO_tmp, qv_tmp, qv_iso_HDO_tmp, sat_ratio, diff_HDO, KT);
                    sb_iso_evaporation_rain(g_therm_iso_O18, sat_ratio, nr_tmp, qr_tmp, mu, qr_iso_O18_tmp, rain_mass, Dp, Dm, &qr_iso_O18_evap_tendency);
                    sb_iso_evaporation_rain(g_therm_iso_O18, sat_ratio, nr_tmp, qr_tmp, mu, qr_iso_HDO_tmp, rain_mass, Dp, Dm, &qr_iso_HDO_evap_tendency);
                    
                    // iso_tendencies add
                    qr_iso_O18_tendency_tmp = qr_iso_O18_auto_tendency + qr_iso_O18_accre_tendency + qr_iso_O18_evap_tendency;
                    ql_iso_O18_tendency_tmp = -qr_iso_O18_auto_tendency - qr_iso_O18_accre_tendency;

                    qr_iso_HDO_tendency_tmp = qr_iso_HDO_auto_tendency + qr_iso_HDO_accre_tendency + qr_iso_HDO_evap_tendency;
                    ql_iso_HDO_tendency_tmp = -qr_iso_HDO_auto_tendency - qr_iso_HDO_accre_tendency;
                    
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
                    qr_iso_O18_tmp += qr_iso_O18_tendency_tmp * dt_;
                    ql_iso_O18_tmp += ql_iso_O18_tendency_tmp * dt_;
                    qv_iso_O18_tmp += -qr_iso_O18_evap_tendency * dt_;
                    qr_iso_HDO_tmp += qr_iso_HDO_tendency_tmp * dt_;
                    ql_iso_HDO_tmp += ql_iso_HDO_tendency_tmp * dt_;
                    qv_iso_HDO_tmp += -qr_iso_HDO_evap_tendency * dt_;

                    qr_iso_O18_tmp  = fmax(qr_iso_O18_tmp, 0.0);
                    ql_iso_O18_tmp  = fmax(ql_iso_O18_tmp, 0.0);
                    qr_iso_HDO_tmp  = fmax(qr_iso_HDO_tmp, 0.0);
                    ql_iso_HDO_tmp  = fmax(ql_iso_HDO_tmp, 0.0);

                    time_added += dt_ ;
                }while(time_added < dt);
                nr_tendency_micro[ijk]  = (nr_tmp - nr[ijk] )/dt;
                qr_tendency_micro[ijk]  = (qr_tmp - qr[ijk])/dt;
                nr_std_tendency[ijk]   += nr_tendency_micro[ijk];
                qr_std_tendency[ijk]       += qr_tendency_micro[ijk];

                qr_iso_O18_tendency_micro[ijk]  = (qr_iso_O18_tmp - qr_iso_O18[ijk])/dt;
                qr_iso_O18_tendency[ijk]       += qr_iso_O18_tendency_micro[ijk];
                qr_iso_HDO_tendency_micro[ijk]  = (qr_iso_HDO_tmp - qr_iso_HDO[ijk])/dt;
                qr_iso_HDO_tendency[ijk]       += qr_iso_HDO_tendency_micro[ijk];
            }
        }
    }
    return;
}

// ===========<<< iso 1_m ice scheme for wbf >>> ============

double ice_kinetic_frac_function(double qi_std, double qi_iso, double qi, double alpha_s_ice, double alpha_k_ice){
    double tendency_std = qi - qi_std;
    double tendency_iso = alpha_s_ice * alpha_k_ice * tendency_std;
    double qi_iso_tmp   = qi_iso + tendency_iso;
    return qi_iso_tmp;
}

void iso_mix_phase_fractionation(const struct DimStruct *dims, struct LookupStruct *LT, 
    double (*lam_fp)(double), double (*L_fp)(double, double),double* restrict temperature, double* restrict p0,
    double* restrict qt_std, double* restrict qv_std, double* restrict ql_std, double* restrict qi_std, 
    double* restrict qt_iso_O18, double* restrict qv_iso_O18, double* restrict ql_iso_O18, double* restrict qi_iso_O18, 
    double* restrict qt_iso_HDO, double* restrict qv_iso_HDO, double* restrict ql_iso_HDO, double* restrict qi_iso_HDO, 
    double* restrict qv_DV, double* restrict ql_DV, double* restrict qi_DV){ 

    ssize_t i,j,k;
    double alpha_s_ice_O18, alpha_k_ice_O18, alpha_eq_lv_O18;
    double alpha_s_ice_HDO, alpha_k_ice_HDO, alpha_eq_lv_HDO;
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
                    double qv_iso_O18_tmp, ql_iso_O18_tmp, qi_iso_O18_tmp;
                    double qv_iso_HDO_tmp, ql_iso_HDO_tmp, qi_iso_HDO_tmp;
                    
                    double diff_O18 = DVAPOR*0.9723;
                    double diff_HDO = DVAPOR*0.9723;

                    alpha_eq_lv_O18 = equilibrium_fractionation_factor_O18_liquid(temperature[ijk]);
                    alpha_eq_lv_HDO = equilibrium_fractionation_factor_HDO_liquid(temperature[ijk]);

                    alpha_s_ice_O18 = 1.0 / equilibrium_fractionation_factor_O18_ice(temperature[ijk]);
                    alpha_s_ice_HDO = 1.0 / equilibrium_fractionation_factor_HDO_ice(temperature[ijk]);

                    alpha_k_ice_O18 = alpha_k_ice_equation_Blossey_Arc1M(LT, lam_fp, L_fp, temperature[ijk], p0[k], qt_std[ijk], alpha_s_ice_O18, DVAPOR, diff_O18);
                    alpha_k_ice_HDO = alpha_k_ice_equation_Blossey_Arc1M(LT, lam_fp, L_fp, temperature[ijk], p0[k], qt_std[ijk], alpha_s_ice_HDO, DVAPOR, diff_HDO);

                    qv_iso_O18_tmp  = eq_frac_function(qt_iso_O18[ijk], qv_DV[ijk], ql_DV[ijk], alpha_eq_lv_O18);
                    qv_iso_HDO_tmp  = eq_frac_function(qt_iso_HDO[ijk], qv_DV[ijk], ql_DV[ijk], alpha_eq_lv_HDO);

                    qi_iso_O18_tmp  = ice_kinetic_frac_function(qi_std[ijk], qi_iso_O18[ijk], qi_DV[ijk], alpha_s_ice_O18, alpha_k_ice_O18);
                    qi_iso_HDO_tmp  = ice_kinetic_frac_function(qi_std[ijk], qi_iso_HDO[ijk], qi_DV[ijk], alpha_s_ice_HDO, alpha_k_ice_HDO);


                    qv_std_tmp  = qv_DV[ijk];
                    ql_std_tmp  = ql_DV[ijk];
                    qi_std_tmp  = qi_DV[ijk];

                    ql_iso_O18_tmp  = qt_iso_O18[ijk] - qv_iso_O18_tmp - qi_iso_O18_tmp;
                    ql_iso_HDO_tmp  = qt_iso_HDO[ijk] - qv_iso_HDO_tmp - qi_iso_HDO_tmp;
                    
                    qv_std[ijk] = qv_std_tmp;
                    ql_std[ijk] = ql_std_tmp;
                    qi_std[ijk] = qi_std_tmp;

                    qv_iso_O18[ijk] = qv_iso_O18_tmp;
                    ql_iso_O18[ijk] = ql_iso_O18_tmp;
                    qi_iso_O18[ijk] = qi_iso_O18_tmp;
                    
                    qv_iso_HDO[ijk] = qv_iso_HDO_tmp;
                    ql_iso_HDO[ijk] = ql_iso_HDO_tmp;
                    qi_iso_HDO[ijk] = qi_iso_HDO_tmp;
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
    double* restrict qt_iso_O18, double* restrict qv_iso_O18, double* restrict ql_iso_O18, double* restrict qi_iso_O18, 
    double* restrict qrain_iso_O18, double* restrict qsnow_iso_O18, double* restrict qrain_iso_O18_tendency, 
    double* restrict qrain_iso_O18_tendency_micro, double* restrict qsnow_iso_O18_tendency, double* restrict qsnow_iso_O18_tendency_micro,
    double* restrict precip_iso_O18_rate, double* restrict evap_iso_O18_rate, 
    double* restrict qt_iso_HDO, double* restrict qv_iso_HDO, double* restrict ql_iso_HDO, double* restrict qi_iso_HDO, 
    double* restrict qrain_iso_HDO, double* restrict qsnow_iso_HDO, double* restrict qrain_iso_HDO_tendency, 
    double* restrict qrain_iso_HDO_tendency_micro, double* restrict qsnow_iso_HDO_tendency, double* restrict qsnow_iso_HDO_tendency_micro,
    double* restrict precip_iso_HDO_rate, double* restrict evap_iso_HDO_rate){

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
    double qt_iso_O18_tmp, qv_iso_O18_tmp, ql_iso_O18_tmp, qi_iso_O18_tmp, qrain_iso_O18_tmp, qsnow_iso_O18_tmp;
    double qrain_iso_O18_tendency_tmp, qrain_iso_O18_tendency_aut, qrain_iso_O18_tendency_acc, qrain_iso_O18_tendency_evp;
    double qsnow_iso_O18_tendency_tmp, qsnow_iso_O18_tendency_aut, qsnow_iso_O18_tendency_acc, qsnow_iso_O18_tendency_evp, qsnow_iso_O18_tendency_melt;
    double ql_iso_O18_tendency_tmp, ql_iso_O18_tendency_acc, qi_iso_O18_tendency_tmp, qi_iso_O18_tendency_acc;
    double precip_iso_O18_tmp, evap_iso_O18_tmp;
    double qt_iso_HDO_tmp, qv_iso_HDO_tmp, ql_iso_HDO_tmp, qi_iso_HDO_tmp, qrain_iso_HDO_tmp, qsnow_iso_HDO_tmp;
    double qrain_iso_HDO_tendency_tmp, qrain_iso_HDO_tendency_aut, qrain_iso_HDO_tendency_acc, qrain_iso_HDO_tendency_evp;
    double qsnow_iso_HDO_tendency_tmp, qsnow_iso_HDO_tendency_aut, qsnow_iso_HDO_tendency_acc, qsnow_iso_HDO_tendency_evp, qsnow_iso_HDO_tendency_melt;
    double ql_iso_HDO_tendency_tmp, ql_iso_HDO_tendency_acc, qi_iso_HDO_tendency_tmp, qi_iso_HDO_tendency_acc;
    double precip_iso_HDO_tmp, evap_iso_HDO_tmp;
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
                qi_iso_O18_tmp           = fmax(qi_iso_O18[ijk], 0.0);
                qt_iso_O18_tmp           = qt_iso_O18[ijk];
                qv_iso_O18_tmp           = fmax(qv_iso_O18[ijk], 0.0);
                qrain_iso_O18_tmp        = fmax(qrain_iso_O18[ijk],0.0);
                ql_iso_O18_tmp           = fmax(ql_iso_O18[ijk], 0.0);
                qsnow_iso_O18_tmp        = fmax(qsnow_iso_O18[ijk],0.0);

                precip_iso_O18_rate[ijk] = 0.0;
                evap_iso_O18_rate[ijk]   = 0.0;
                
                qi_iso_HDO_tmp           = fmax(qi_iso_HDO[ijk], 0.0);
                qt_iso_HDO_tmp           = qt_iso_HDO[ijk];
                qv_iso_HDO_tmp           = fmax(qv_iso_HDO[ijk], 0.0);
                qrain_iso_HDO_tmp        = fmax(qrain_iso_HDO[ijk],0.0);
                ql_iso_HDO_tmp           = fmax(ql_iso_HDO[ijk], 0.0);
                qsnow_iso_HDO_tmp        = fmax(qsnow_iso_HDO[ijk],0.0);

                precip_iso_HDO_rate[ijk] = 0.0;
                evap_iso_HDO_rate[ijk]   = 0.0;
                
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

                    qrain_iso_O18_tendency_aut  = 0.0;
                    qrain_iso_O18_tendency_acc  = 0.0;
                    qrain_iso_O18_tendency_evp  = 0.0;

                    qsnow_iso_O18_tendency_aut  = 0.0;
                    qsnow_iso_O18_tendency_acc  = 0.0;
                    qsnow_iso_O18_tendency_evp  = 0.0;
                    qsnow_iso_O18_tendency_melt = 0.0;

                    ql_iso_O18_tendency_acc     = 0.0;
                    qi_iso_O18_tendency_acc     = 0.0;

                    qrain_iso_HDO_tendency_aut  = 0.0;
                    qrain_iso_HDO_tendency_acc  = 0.0;
                    qrain_iso_HDO_tendency_evp  = 0.0;

                    qsnow_iso_HDO_tendency_aut  = 0.0;
                    qsnow_iso_HDO_tendency_acc  = 0.0;
                    qsnow_iso_HDO_tendency_evp  = 0.0;
                    qsnow_iso_HDO_tendency_melt = 0.0;

                    ql_iso_HDO_tendency_acc     = 0.0;
                    qi_iso_HDO_tendency_acc     = 0.0;

                    // the following sections are non-fractionational processes
                    arc1m_iso_autoconversion_rain(qrain_tendency_aut, ql_std_tmp, ql_iso_O18_tmp, &qrain_iso_O18_tendency_aut);
                    arc1m_iso_autoconversion_rain(qrain_tendency_aut, ql_std_tmp, ql_iso_HDO_tmp, &qrain_iso_HDO_tendency_aut);

                    arc1m_iso_autoconversion_snow(qsnow_tendency_aut, qi_std_tmp, qi_iso_O18_tmp, &qsnow_iso_O18_tendency_aut);
                    arc1m_iso_autoconversion_snow(qsnow_tendency_aut, qi_std_tmp, qi_iso_HDO_tmp, &qsnow_iso_HDO_tendency_aut);

                    arc1m_iso_accretion_all(density[k], p0[k], temperature[ijk], ccn, ql_tmp, qi_tmp, ni,
                        qrain_tmp, nrain[ijk], qsnow_tmp, nsnow[ijk],
                        ql_iso_O18_tmp, qi_iso_O18_tmp, qrain_iso_O18_tmp, qsnow_iso_O18_tmp,
                        &ql_iso_O18_tendency_acc, &qi_iso_O18_tendency_acc, &qrain_iso_O18_tendency_acc, &qsnow_iso_O18_tendency_acc);
                    arc1m_iso_accretion_all(density[k], p0[k], temperature[ijk], ccn, ql_tmp, qi_tmp, ni,
                        qrain_tmp, nrain[ijk], qsnow_tmp, nsnow[ijk],
                        ql_iso_HDO_tmp, qi_iso_HDO_tmp, qrain_iso_HDO_tmp, qsnow_iso_HDO_tmp,
                        &ql_iso_HDO_tendency_acc, &qi_iso_HDO_tendency_acc, &qrain_iso_HDO_tendency_acc, &qsnow_iso_HDO_tendency_acc);

                    // the following sections are kinetic fractionational processes 
                    // defination of thermo variables
                    double vapor_diff = vapor_diffusivity(temperature[ijk], p0[k]);
                    double therm_cond = thermal_conductivity(temperature[ijk]);
                    double diff_O18 = vapor_diff*0.9723;
                    // ================================================
                    // ToDo: give the defination of diff_HDO based on the actual physical value of the diffusivity of HDO
                    // ================================================
                    double diff_HDO = vapor_diff*0.9723;

                    double gtherm_iso_O18_liq, gtherm_iso_O18_ice, gtherm_iso_HDO_liq, gtherm_iso_HDO_ice;

                    double alpha_eq_lv_O18 = equilibrium_fractionation_factor_O18_liquid(temperature[ijk]);
                    double alpha_eq_lv_HDO = equilibrium_fractionation_factor_HDO_liquid(temperature[ijk]);
                    
                    double sat_ratio = microphysics_saturation_ratio(LT, temperature[ijk], p0[k], qt_tmp);

                    gtherm_iso_O18_liq = microphysics_g_iso_Arc1M(LT, lam_fp, L_fp, temperature[ijk], p0[k], qrain_tmp, qrain_iso_O18_tmp, 
                        qv_tmp, qv_iso_O18_tmp, sat_ratio, alpha_eq_lv_O18, diff_O18, therm_cond);
                    arc1m_iso_evap_rain(LT, lam_fp, L_fp, density[k], p0[k], temperature[ijk], sat_ratio, qt_tmp, qv_tmp, qrain_tmp, 
                        nrain[ijk], gtherm_iso_O18_liq, qv_iso_O18_tmp, qrain_iso_O18_tmp, &qrain_iso_O18_tendency_evp);
                    
                    gtherm_iso_HDO_liq = microphysics_g_iso_Arc1M(LT, lam_fp, L_fp, temperature[ijk], p0[k], qrain_tmp, qrain_iso_HDO_tmp, 
                        qv_tmp, qv_iso_HDO_tmp, sat_ratio, alpha_eq_lv_HDO, diff_HDO, therm_cond);
                    arc1m_iso_evap_rain(LT, lam_fp, L_fp, density[k], p0[k], temperature[ijk], sat_ratio, qt_tmp, qv_tmp, qrain_tmp, 
                        nrain[ijk], gtherm_iso_HDO_liq, qv_iso_HDO_tmp, qrain_iso_HDO_tmp, &qrain_iso_HDO_tendency_evp);
                    
                    // ===========<<< ice phase kinetic fractionation >>> ============
                    
                    double alpha_s_ice_O18 = 1.0 / equilibrium_fractionation_factor_O18_ice(temperature[ijk]);
                    double alpha_s_ice_HDO = 1.0 / equilibrium_fractionation_factor_HDO_ice(temperature[ijk]);

                    gtherm_iso_O18_ice = microphysics_g_iso_Arc1M(LT, lam_fp, L_fp, temperature[ijk], p0[k], qsnow_tmp, qsnow_iso_O18_tmp, 
                        qv_tmp, qv_iso_O18_tmp, sat_ratio, alpha_s_ice_O18, diff_O18, therm_cond);
                    arc1m_iso_evap_snow(LT, lam_fp, L_fp, density[k], p0[k], temperature[ijk], qt_tmp, qv_tmp, qsnow_tmp, nsnow[ijk], 
                        gtherm_iso_O18_ice, qv_iso_O18_tmp, qsnow_iso_O18_tmp, &qsnow_iso_O18_tendency_evp);

                    gtherm_iso_HDO_ice = microphysics_g_iso_Arc1M(LT, lam_fp, L_fp, temperature[ijk], p0[k], qsnow_tmp, qsnow_iso_HDO_tmp, 
                        qv_tmp, qv_iso_HDO_tmp, sat_ratio, alpha_s_ice_HDO, diff_HDO, therm_cond);
                    arc1m_iso_evap_snow(LT, lam_fp, L_fp, density[k], p0[k], temperature[ijk], qt_tmp, qv_tmp, qsnow_tmp, nsnow[ijk], 
                        gtherm_iso_HDO_ice, qv_iso_HDO_tmp, qsnow_iso_HDO_tmp, &qsnow_iso_HDO_tendency_evp);

                    arc1m_iso_melt_snow(qsnow_tendency_melt, qsnow_tmp, qsnow_iso_O18_tmp, &qsnow_iso_O18_tendency_melt);
                    arc1m_iso_melt_snow(qsnow_tendency_melt, qsnow_tmp, qsnow_iso_HDO_tmp, &qsnow_iso_HDO_tendency_melt);
                    //
                    qrain_iso_O18_tendency_tmp  = qrain_iso_O18_tendency_aut + qrain_iso_O18_tendency_acc + qrain_iso_O18_tendency_evp - qsnow_iso_O18_tendency_melt;
                    qsnow_iso_O18_tendency_tmp  = qsnow_iso_O18_tendency_aut + qsnow_iso_O18_tendency_acc + qsnow_iso_O18_tendency_evp + qsnow_iso_O18_tendency_melt;
                    ql_iso_O18_tendency_tmp     = ql_iso_O18_tendency_acc - qrain_iso_O18_tendency_aut;
                    qi_iso_O18_tendency_tmp     = qi_iso_O18_tendency_acc - qsnow_iso_O18_tendency_aut;

                    qrain_iso_HDO_tendency_tmp  = qrain_iso_HDO_tendency_aut + qrain_iso_HDO_tendency_acc + qrain_iso_HDO_tendency_evp - qsnow_iso_HDO_tendency_melt;
                    qsnow_iso_HDO_tendency_tmp  = qsnow_iso_HDO_tendency_aut + qsnow_iso_HDO_tendency_acc + qsnow_iso_HDO_tendency_evp + qsnow_iso_HDO_tendency_melt;
                    ql_iso_HDO_tendency_tmp     = ql_iso_HDO_tendency_acc - qrain_iso_HDO_tendency_aut;
                    qi_iso_HDO_tendency_tmp     = qi_iso_HDO_tendency_acc - qsnow_iso_HDO_tendency_aut;

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
                    precip_iso_O18_tmp = -qrain_iso_O18_tendency_aut + ql_iso_O18_tendency_acc - qsnow_iso_O18_tendency_aut + qi_iso_O18_tendency_acc;
                    evap_iso_O18_tmp   = qrain_iso_O18_tendency_evp + qsnow_iso_O18_tendency_evp;

                    precip_iso_O18_rate[ijk] += precip_iso_O18_tmp * dt_;
                    evap_iso_O18_rate[ijk]   += evap_iso_O18_tmp * dt_;
                    
                    precip_iso_HDO_tmp = -qrain_iso_HDO_tendency_aut + ql_iso_HDO_tendency_acc - qsnow_iso_HDO_tendency_aut + qi_iso_HDO_tendency_acc;
                    evap_iso_HDO_tmp   = qrain_iso_HDO_tendency_evp + qsnow_iso_HDO_tendency_evp;

                    precip_iso_HDO_rate[ijk] += precip_iso_HDO_tmp * dt_;
                    evap_iso_HDO_rate[ijk]   += evap_iso_HDO_tmp * dt_;

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
                    ql_iso_O18_tmp    += ql_iso_O18_tendency_tmp * dt_;
                    qi_iso_O18_tmp    += qi_iso_O18_tendency_tmp * dt_;
                    qrain_iso_O18_tmp += qrain_iso_O18_tendency_tmp *dt_;
                    qsnow_iso_O18_tmp += qsnow_iso_O18_tendency_tmp *dt_;
                    qt_iso_O18_tmp    += (precip_iso_O18_tmp - evap_iso_O18_tmp) * dt_;

                    qt_iso_O18_tmp     = fmax(qt_iso_O18_tmp, 0.0);
                    ql_iso_O18_tmp     = fmax(ql_iso_O18_tmp, 0.0);
                    qi_iso_O18_tmp     = fmax(qi_iso_O18_tmp, 0.0);
                    qrain_iso_O18_tmp  = fmax(qrain_iso_O18_tmp, 0.0);
                    qsnow_iso_O18_tmp  = fmax(qsnow_iso_O18_tmp, 0.0);
                    double qv_iso_O18_ = qt_iso_O18_tmp - ql_iso_O18_tmp - qi_iso_O18_tmp;
                    qv_iso_O18_tmp     = fmax(qv_iso_O18_, 0.0);

                    ql_iso_HDO_tmp    += ql_iso_HDO_tendency_tmp * dt_;
                    qi_iso_HDO_tmp    += qi_iso_HDO_tendency_tmp * dt_;
                    qrain_iso_HDO_tmp += qrain_iso_HDO_tendency_tmp *dt_;
                    qsnow_iso_HDO_tmp += qsnow_iso_HDO_tendency_tmp *dt_;
                    qt_iso_HDO_tmp    += (precip_iso_HDO_tmp - evap_iso_HDO_tmp) * dt_;

                    qt_iso_HDO_tmp     = fmax(qt_iso_HDO_tmp, 0.0);
                    ql_iso_HDO_tmp     = fmax(ql_iso_HDO_tmp, 0.0);
                    qi_iso_HDO_tmp     = fmax(qi_iso_HDO_tmp, 0.0);
                    qrain_iso_HDO_tmp  = fmax(qrain_iso_HDO_tmp, 0.0);
                    qsnow_iso_HDO_tmp  = fmax(qsnow_iso_HDO_tmp, 0.0);
                    double qv_iso_HDO_ = qt_iso_HDO_tmp - ql_iso_HDO_tmp - qi_iso_HDO_tmp;
                    qv_iso_HDO_tmp     = fmax(qv_iso_HDO_, 0.0);

                    time_added += dt_;
                    }while(time_added < dt && iter_count < MAX_ITER);

                qrain_tendency_micro[ijk]  = (qrain_tmp - qrain[ijk])/dt;
                qrain_tendency[ijk]       += qrain_tendency_micro[ijk];
                qsnow_tendency_micro[ijk]  = (qsnow_tmp - qsnow[ijk])/dt;
                qsnow_tendency[ijk]       += qsnow_tendency_micro[ijk];

                precip_rate[ijk] = precip_rate[ijk]/dt;
                evap_rate[ijk]   = evap_rate[ijk]/dt;
                melt_rate[ijk]   = melt_rate[ijk]/dt;

                qrain_iso_O18_tendency_micro[ijk]  = (qrain_iso_O18_tmp - qrain_iso_O18[ijk])/dt;
                qrain_iso_O18_tendency[ijk]       += qrain_iso_O18_tendency_micro[ijk];
                qsnow_iso_O18_tendency_micro[ijk]  = (qsnow_iso_O18_tmp - qsnow_iso_O18[ijk])/dt;
                qsnow_iso_O18_tendency[ijk]       += qsnow_iso_O18_tendency_micro[ijk];

                precip_iso_O18_rate[ijk] = precip_iso_O18_rate[ijk]/dt;
                evap_iso_O18_rate[ijk]   = evap_iso_O18_rate[ijk]/dt;
                
                qrain_iso_HDO_tendency_micro[ijk]  = (qrain_iso_HDO_tmp - qrain_iso_HDO[ijk])/dt;
                qrain_iso_HDO_tendency[ijk]       += qrain_iso_HDO_tendency_micro[ijk];
                qsnow_iso_HDO_tendency_micro[ijk]  = (qsnow_iso_HDO_tmp - qsnow_iso_HDO[ijk])/dt;
                qsnow_iso_HDO_tendency[ijk]       += qsnow_iso_HDO_tendency_micro[ijk];

                precip_iso_HDO_rate[ijk] = precip_iso_HDO_rate[ijk]/dt;
                evap_iso_HDO_rate[ijk]   = evap_iso_HDO_rate[ijk]/dt;
            }
        }
    }
    return;
};

// ===========<<< SBSI two moment microphysics scheme >>> ============

void tracer_sb_si_microphysics_sources(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
    double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
    double* restrict density, double* restrict p0,  double* restrict temperature,  double* restrict qt, double ccn,
    double* restrict ql, double* restrict nr, double* restrict qr, double* restrict qi, double* restrict ni, double dt, double* restrict ql_std,
    double* restrict nr_tendency_micro, double* restrict qr_tendency_micro, double* restrict nr_tendency, double* restrict qr_tendency, 
    double* restrict ni_tendency_micro, double* restrict qi_tendency_micro, double* restrict ni_tendency, double* restrict qi_tendency, 
    double* restrict precip_rate, double* restrict evap_rate, double* restrict melt_rate, 
    double* restrict qt_iso_O18, double* restrict ql_iso_O18, double* restrict qr_iso_O18, double* restrict qi_iso_O18, double* restrict qr_iso_O18_tendency, 
    double* restrict qi_iso_O18_tendency, double* restrict qr_iso_O18_tendency_micro, double* restrict qi_iso_O18_tendency_micro,
    double* restrict qt_iso_HDO, double* restrict ql_iso_HDO, double* restrict qr_iso_HDO, double* restrict qi_iso_HDO, double* restrict qr_iso_HDO_tendency, 
    double* restrict qi_iso_HDO_tendency, double* restrict qr_iso_HDO_tendency_micro, double* restrict qi_iso_HDO_tendency_micro){

    // Here all rain and ice related variables are std_tracers;
    // while qr_iso* and qi_iso* means isotope tracers.
    // all qi related variables are single ice variables.
    // Here we compute the source terms for nr, qr & ni, qr(number and mass of rain)
    // Temporal substepping is used to help ensure boundedness of moments
    double rain_mass, Dm_r, mu, Dp, nr_tendency_tmp, qr_tendency_tmp, ql_tendency_tmp, ql_tendency_frez;
    double liquid_mass, Dm_l, velocity_liquid;
    double nr_tendency_au, nr_tendency_scbk, nr_tendency_evap, nr_tendency_frez;
    double qr_tendency_au, qr_tendency_ac, qr_tendency_evap, qr_tendency_frez;
    double sat_ratio;
    // single ice tendency definition
    double qi_tendency_tmp, ni_tendency_tmp;
    double ni_tendency_nuc, ni_tendency_frez, ni_tendency_berg, ni_tendency_melt;
    double qi_tendency_nuc, qi_tendency_frez, qi_tendency_acc, qi_tendency_dep, qi_tendency_berg, qi_tendency_melt, qi_tendency_sub;
    // isotope tracers tendency definition
    double qv_iso_O18_tendency_tmp, ql_iso_O18_tendency_tmp, ql_iso_O18_tendency_frez;
    double qr_iso_O18_tendency_tmp, qr_iso_O18_tendency_auto, qr_iso_O18_tendency_acc, qr_iso_O18_tendency_evap, qr_iso_O18_tendency_frez;
    double qi_iso_O18_tendency_tmp, qi_iso_O18_tendency_nuc, qi_iso_O18_tendency_frez, qi_iso_O18_tendency_dep, qi_iso_O18_tendency_sub, qi_iso_O18_tendency_melt, qi_iso_O18_tendency_acc;

    double qv_iso_HDO_tendency_tmp, ql_iso_HDO_tendency_tmp, ql_iso_HDO_tendency_frez;
    double qr_iso_HDO_tendency_tmp, qr_iso_HDO_tendency_auto, qr_iso_HDO_tendency_acc, qr_iso_HDO_tendency_evap, qr_iso_HDO_tendency_frez;
    double qi_iso_HDO_tendency_tmp, qi_iso_HDO_tendency_nuc, qi_iso_HDO_tendency_frez, qi_iso_HDO_tendency_dep, qi_iso_HDO_tendency_sub, qi_iso_HDO_tendency_melt, qi_iso_HDO_tendency_acc;

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
                qi[ijk] = fmax(qi[ijk],0.0);
                ni[ijk] = fmax(fmin(ni[ijk], qi[ijk]/ICE_MIN_MASS),qi[ijk]/ICE_MAX_MASS);

                double qv_tmp = qt[ijk] - fmax(ql[ijk],0.0);
                double qt_tmp = qt[ijk];
                double nl     = ccn/density[k];
                double ql_tmp = fmax(ql[ijk],0.0);
                double ql_std_tmp  = fmax(ql_std[ijk],0.0);
                // holding nl fixed since it doesn't change between timesteps
                
                // get qr and nr values before the computation system begin
                double qr_tmp = fmax(qr[ijk],0.0);
                double nr_tmp = fmax(fmin(nr[ijk], qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                
                // get qi and ni values before the computation system begin
                double qi_tmp = fmax(qi[ijk],0.0);
                double ni_tmp = fmax(fmin(ni[ijk], qi_tmp/ICE_MIN_MASS),qi_tmp/ICE_MAX_MASS);

                double Dm_i, velocity_ice, sb_a_ice, sb_b_ice, sifi_av, sifi_bv, sb_beta_ice, ice_mass;
                
                precip_rate[ijk] = 0.0;
                evap_rate[ijk] = 0.0;
                melt_rate[ijk] = 0.0;
                
                // Get isotpe tracer values before the computation system begin
                double qt_iso_O18_tmp = qt_iso_O18[ijk];
                double ql_iso_O18_tmp = fmax(ql_iso_O18[ijk], 0.0);
                double qv_iso_O18_tmp = qt_iso_O18_tmp - ql_iso_O18_tmp;
                double qr_iso_O18_tmp = fmax(qr_iso_O18[ijk], 0.0);
                double qi_iso_O18_tmp = fmax(qi_iso_O18[ijk], 0.0);
                
                double qt_iso_HDO_tmp = qt_iso_HDO[ijk];
                double ql_iso_HDO_tmp = fmax(ql_iso_HDO[ijk], 0.0);
                double qv_iso_HDO_tmp = qt_iso_HDO_tmp - ql_iso_HDO_tmp;
                double qr_iso_HDO_tmp = fmax(qr_iso_HDO[ijk], 0.0);
                double qi_iso_HDO_tmp = fmax(qi_iso_HDO[ijk], 0.0);

                double time_added = 0.0, dt_, rate;
                ssize_t iter_count = 0;
                do{
                    qi_tendency_tmp   = 0.0;
                    ni_tendency_tmp   = 0.0;
                    qr_tendency_tmp   = 0.0;
                    nr_tendency_tmp   = 0.0;
                    ql_tendency_tmp   = 0.0;

                    iter_count       += 1;
                    double sat_ratio_liq    = microphysics_saturation_ratio_liq(temperature[ijk], p0[k], qt_tmp);
                    double sat_ratio_ice    = microphysics_saturation_ratio_ice(temperature[ijk], p0[k], qt_tmp);
                    double sat_ratio_lookup = microphysics_saturation_ratio(LT, temperature[ijk], p0[k], qt_tmp);
                    ql_tendency_frez  = 0.0;

                    nr_tendency_au    = 0.0;
                    nr_tendency_scbk  = 0.0;
                    nr_tendency_evap  = 0.0;
                    nr_tendency_frez  = 0.0;
                    qr_tendency_au    = 0.0;
                    qr_tendency_ac    = 0.0;
                    qr_tendency_evap  = 0.0;
                    qr_tendency_frez  = 0.0;

                    qi_tendency_nuc   = 0.0;
                    qi_tendency_acc   = 0.0;
                    qi_tendency_dep   = 0.0;
                    qi_tendency_frez  = 0.0;
                    qi_tendency_melt  = 0.0;
                    qi_tendency_berg  = 0.0;
                    qi_tendency_sub   = 0.0;
                    ni_tendency_nuc   = 0.0;
                    ni_tendency_frez  = 0.0;
                    ni_tendency_melt  = 0.0;
                    ni_tendency_berg  = 0.0;

                    //obtain some parameters of cloud droplets
                    liquid_mass = microphysics_mean_mass(nl, ql_tmp, LIQUID_MIN_MASS, LIQUID_MAX_MASS);// average mass of cloud droplets
                    Dm_l =  cbrt(liquid_mass * 6.0/DENSITY_LIQUID/pi);
                    velocity_liquid = 3.75e5 * cbrt(liquid_mass)*cbrt(liquid_mass) *(DENSITY_SB/density[ijk]);

                    //obtain some parameters of rain droplets
                    rain_mass = microphysics_mean_mass(nr_tmp, qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS); //average mass of rain droplet
                    Dm_r      = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi); // mass weighted diameter of rain droplets
                    Dp        = sb_Dp(Dm_r, mu);
                    mu        = rain_mu(density[k], qr_tmp, Dm_r);

                    //obtain some parameters of ice particle
                    // ================================================
                    // ToDo: set the right calculation processes of lwp and iwp, and following calculation of Ri 
                    // ================================================
                    double Ri;
                    ice_mass = microphysics_mean_mass(ni_tmp, qi_tmp, ICE_MIN_MASS, ICE_MAX_MASS);
                    sb_si_get_ice_parameters_SIFI(&sb_a_ice, &sb_b_ice, &sifi_av, &sifi_bv, &sb_beta_ice);
                    Dm_i     = sb_a_ice * pow(ice_mass, sb_b_ice);
                    velocity_ice  = sifi_av * pow(Dm_i, sifi_bv);

                    double g_therm_liq = microphysics_g_liq_SBSI(temperature[ijk], DVAPOR, KT);
                    double g_therm_ice = microphysics_g_ice_SBSI(temperature[ijk], DVAPOR, KT);

                    //compute the source terms
                    sb_autoconversion_rain(droplet_nu, density[k], nl, ql_tmp, qr_tmp, &nr_tendency_au, &qr_tendency_au);
                    sb_accretion_rain(density[k], ql_tmp, qr_tmp, &qr_tendency_ac); 
                    sb_selfcollection_breakup_rain(density[k], nr_tmp, qr_tmp, mu, rain_mass, Dm_r, &nr_tendency_scbk);
                    sb_evaporation_rain(g_therm_liq, sat_ratio_liq, nr_tmp, qr_tmp, mu, rain_mass, Dp, Dm_r, &nr_tendency_evap, &qr_tendency_evap);

                    sb_nucleation_ice(temperature[ijk], sat_ratio_ice, dt_, ni_tmp, density[k], &qi_tendency_nuc, &ni_tendency_nuc);
                    sb_freezing_ice(droplet_nu, density[k], temperature[ijk], liquid_mass, rain_mass, ql_tmp, nl, qr_tmp, 
                            nr_tmp, &ql_tendency_frez, &qr_tendency_frez, &nr_tendency_frez, &qi_tendency_frez, &ni_tendency_frez);
                    sb_accretion_cloud_ice(liquid_mass, Dm_l, velocity_liquid, ice_mass, Dm_i, velocity_ice, nl, ql_tmp, 
                            ni_tmp, qi_tmp, sb_a_ice, sb_b_ice, sb_beta_ice, &qi_tendency_acc);
                    sb_deposition_ice(g_therm_ice, temperature[ijk], Dm_i, sat_ratio_ice, ice_mass, velocity_ice,
                            qi_tmp, ni_tmp, sb_b_ice, sb_beta_ice, &qi_tendency_dep);   
                    sb_sublimation_ice(g_therm_ice, temperature[ijk], Dm_i, sat_ratio_ice, ice_mass, velocity_ice,
                            qi_tmp, ni_tmp, sb_b_ice, sb_beta_ice, &qi_tendency_sub);  
                    sb_melting_ice(LT, lam_fp, L_fp, temperature[ijk], ice_mass, Dm_i, qv_tmp, ni_tmp, qi_tmp, 
                            &ni_tendency_melt, &qi_tendency_melt);

                    // double check the source term magnitudes
                    // qi_tendency_sub is POSITIVE, qi_tendency_dep is POSITIVE;
                    // qr_tendency_evap is NEGATIVE;
                    // qi_tendency_melt and ni_tendency_melt are all POSITIVE
                    qi_tendency_tmp = qi_tendency_nuc + qi_tendency_frez + qi_tendency_acc + qi_tendency_dep + qi_tendency_berg + qi_tendency_sub - qi_tendency_melt;
                    ni_tendency_tmp = ni_tendency_nuc + ni_tendency_frez + ni_tendency_berg - ni_tendency_melt;

                    // double check the source term magnitude:
                    // qr_tendency_frez is POSITIVE
                    // ql_tendency_frez is POSITIVE
                    nr_tendency_tmp = nr_tendency_au + nr_tendency_scbk + nr_tendency_evap + ni_tendency_melt - nr_tendency_frez;
                    qr_tendency_tmp = qr_tendency_au + qr_tendency_ac + qr_tendency_evap + qi_tendency_melt - qr_tendency_frez;

                    ql_tendency_tmp = -qr_tendency_au - qr_tendency_ac - ql_tendency_frez - qi_tendency_acc;
                    
                    // ===========<<< isotope traces components computation >>> ============
                    // set all variables with initial values as 0.0 before loop start;
                    qv_iso_O18_tendency_tmp  = 0.0;
                    ql_iso_O18_tendency_tmp  = 0.0;
                    ql_iso_O18_tendency_frez = 0.0;
                    qr_iso_O18_tendency_tmp  = 0.0;
                    qr_iso_O18_tendency_auto = 0.0;
                    qr_iso_O18_tendency_acc  = 0.0;
                    qr_iso_O18_tendency_evap = 0.0;
                    qr_iso_O18_tendency_frez = 0.0;
                    qi_iso_O18_tendency_tmp  = 0.0;
                    qi_iso_O18_tendency_nuc  = 0.0;
                    qi_iso_O18_tendency_frez = 0.0;
                    qi_iso_O18_tendency_dep  = 0.0;
                    qi_iso_O18_tendency_sub  = 0.0;
                    qi_iso_O18_tendency_melt = 0.0;
                    qi_iso_O18_tendency_acc  = 0.0;
                    
                    qv_iso_HDO_tendency_tmp  = 0.0;
                    ql_iso_HDO_tendency_tmp  = 0.0;
                    ql_iso_HDO_tendency_frez = 0.0;
                    qr_iso_HDO_tendency_tmp  = 0.0;
                    qr_iso_HDO_tendency_auto = 0.0;
                    qr_iso_HDO_tendency_acc  = 0.0;
                    qr_iso_HDO_tendency_evap = 0.0;
                    qr_iso_HDO_tendency_frez = 0.0;
                    qi_iso_HDO_tendency_tmp  = 0.0;
                    qi_iso_HDO_tendency_nuc  = 0.0;
                    qi_iso_HDO_tendency_frez = 0.0;
                    qi_iso_HDO_tendency_dep  = 0.0;
                    qi_iso_HDO_tendency_sub  = 0.0;
                    qi_iso_HDO_tendency_melt = 0.0;
                    qi_iso_HDO_tendency_acc  = 0.0;

                    // iso_tendencies calculations
                    // non-fractionational processes of qr for O18 and HDO

                    sb_iso_rain_autoconversion(ql_tmp, ql_iso_O18_tmp, qr_tendency_au, &qr_iso_O18_tendency_auto);
                    sb_iso_rain_autoconversion(ql_tmp, ql_iso_HDO_tmp, qr_tendency_au, &qr_iso_HDO_tendency_auto);

                    sb_iso_rain_accretion(ql_tmp, ql_iso_O18_tmp, qr_tendency_ac, &qr_iso_O18_tendency_acc);
                    sb_iso_rain_accretion(ql_tmp, ql_iso_HDO_tmp, qr_tendency_ac, &qr_iso_HDO_tendency_acc);

                    sb_iso_ice_freezing(ql_tmp, qr_tmp, nr_tmp, ql_iso_O18_tmp, qr_iso_O18_tmp, ql_tendency_frez, qr_tendency_frez,
                            &qr_iso_O18_tendency_frez, &ql_iso_O18_tendency_frez, &qi_iso_O18_tendency_frez);
                    sb_iso_ice_freezing(ql_tmp, qr_tmp, nr_tmp, ql_iso_HDO_tmp, qr_iso_HDO_tmp, ql_tendency_frez, qr_tendency_frez,
                            &qr_iso_HDO_tendency_frez, &ql_iso_HDO_tendency_frez, &qi_iso_HDO_tendency_frez);

                    // fractionational processes of qi(qisi) for O18 and HDO
                    
                    // ================================================
                    // ToDo: give the defination of diff_HDO based on the actual physical value of the diffusivity of HDO
                    // ================================================
                    double diff_O18 = DVAPOR*0.9723;
                    double diff_HDO = DVAPOR*0.9723;

                    double gtherm_iso_O18_liq = microphysics_g_iso_rain_SBSI(LT, lam_fp, L_fp, temperature[ijk], p0[k], qr_tmp, qr_iso_O18_tmp, 
                        qv_tmp, qv_iso_O18_tmp, sat_ratio_liq, diff_O18, KT);
                    sb_iso_evaporation_rain(gtherm_iso_O18_liq, sat_ratio, nr_tmp, qr_tmp, mu, qr_iso_O18_tmp, rain_mass, Dp, Dm_r, &qr_iso_O18_tendency_evap);

                    double gtherm_iso_HDO_liq = microphysics_g_iso_rain_SBSI(LT, lam_fp, L_fp, temperature[ijk], p0[k], qr_tmp, qr_iso_HDO_tmp, 
                        qv_tmp, qv_iso_HDO_tmp, sat_ratio_liq, diff_HDO, KT);
                    sb_iso_evaporation_rain(gtherm_iso_HDO_liq, sat_ratio, nr_tmp, qr_tmp, mu, qr_iso_HDO_tmp, rain_mass, Dp, Dm_r, &qr_iso_HDO_tendency_evap);

                    double alpha_s_ice_O18 = 1.0 / equilibrium_fractionation_factor_O18_ice(temperature[ijk]);
                    double alpha_k_ice_O18 = alpha_k_ice_equation_Blossey_SBSI(LT, lam_fp, L_fp, temperature[ijk], p0[k], qt[ijk], alpha_s_ice_O18, DVAPOR, diff_O18);

                    double alpha_s_ice_HDO = 1.0 / equilibrium_fractionation_factor_HDO_ice(temperature[ijk]);
                    double alpha_k_ice_HDO = alpha_k_ice_equation_Blossey_SBSI(LT, lam_fp, L_fp, temperature[ijk], p0[k], qt[ijk], alpha_s_ice_HDO, DVAPOR, diff_HDO);

                    double F_ratio = 0.998;

                    sb_iso_ice_nucleation(qi_tendency_nuc, alpha_s_ice_O18, &qi_iso_O18_tendency_nuc);
                    sb_iso_ice_nucleation(qi_tendency_nuc, alpha_s_ice_HDO, &qi_iso_HDO_tendency_nuc);

                    sb_iso_ice_accretion_cloud(ql_tmp, qi_tmp, ni_tmp, qi_iso_O18_tmp, qi_tendency_acc, &qi_iso_O18_tendency_acc);
                    sb_iso_ice_accretion_cloud(ql_tmp, qi_tmp, ni_tmp, qi_iso_HDO_tmp, qi_tendency_acc, &qi_iso_HDO_tendency_acc);

                    sb_iso_ice_deposition(qi_tmp, ni_tmp, qi_iso_O18_tmp, sat_ratio, alpha_k_ice_O18, alpha_s_ice_O18, 
                            qi_tendency_dep, F_ratio, &qi_iso_O18_tendency_dep);
                    sb_iso_ice_deposition(qi_tmp, ni_tmp, qi_iso_HDO_tmp, sat_ratio, alpha_k_ice_HDO, alpha_s_ice_HDO, 
                            qi_tendency_dep, F_ratio, &qi_iso_HDO_tendency_dep);

                    sb_iso_ice_sublimation(qi_tmp, ni_tmp, qi_iso_O18_tmp, sat_ratio, qi_tendency_sub, &qi_iso_O18_tendency_sub);
                    sb_iso_ice_sublimation(qi_tmp, ni_tmp, qi_iso_HDO_tmp, sat_ratio, qi_tendency_sub, &qi_iso_HDO_tendency_sub);
                    // sb_iso_ice_melting(qi_tendency_melt, R_qi, &qi_iso_tendency_melt);

                    // iso_tendencies add
                    // qi_iso_tendency_sub is NEGATIVE because qi_tendency_sub is NEGATIVE
                    qi_iso_O18_tendency_tmp = qi_iso_O18_tendency_nuc + qi_iso_O18_tendency_frez + qi_iso_O18_tendency_dep + qi_iso_O18_tendency_acc + qi_iso_O18_tendency_sub - qi_iso_O18_tendency_melt;
                    qr_iso_O18_tendency_tmp = qr_iso_O18_tendency_auto + qr_iso_O18_tendency_acc + qr_iso_O18_tendency_evap - qr_iso_O18_tendency_frez;
                    ql_iso_O18_tendency_tmp = -qr_iso_O18_tendency_auto - qr_iso_O18_tendency_acc - ql_iso_O18_tendency_frez; 
                    qv_iso_O18_tendency_tmp = -qi_iso_O18_tendency_dep - qr_iso_O18_tendency_evap - qi_iso_O18_tendency_sub;

                    qi_iso_HDO_tendency_tmp = qi_iso_HDO_tendency_nuc + qi_iso_HDO_tendency_frez + qi_iso_HDO_tendency_dep + qi_iso_HDO_tendency_acc + qi_iso_HDO_tendency_sub - qi_iso_HDO_tendency_melt;
                    qr_iso_HDO_tendency_tmp = qr_iso_HDO_tendency_auto + qr_iso_HDO_tendency_acc + qr_iso_HDO_tendency_evap - qr_iso_HDO_tendency_frez;
                    ql_iso_HDO_tendency_tmp = -qr_iso_HDO_tendency_auto - qr_iso_HDO_tendency_acc - ql_iso_HDO_tendency_frez; 
                    qv_iso_HDO_tendency_tmp = -qi_iso_HDO_tendency_dep - qr_iso_HDO_tendency_evap - qi_iso_HDO_tendency_sub;

                    //Factor of 1.05 is ad-hoc
                    
                    //find the maximum substep time
                    dt_ = dt - time_added;
                    rate = 1.05 * ql_tendency_tmp * dt_ /(- fmax(ql_tmp,SB_EPS));
                    rate = fmax(1.05 * nr_tendency_tmp * dt_ /(-fmax(nr_tmp,SB_EPS)), rate);
                    rate = fmax(1.05 * qr_tendency_tmp * dt_ /(-fmax(qr_tmp,SB_EPS)), rate);
                    rate = fmax(1.05 * ni_tendency_tmp * dt_ /(-fmax(ni_tmp,SB_EPS)), rate);
                    rate = fmax(1.05 * qi_tendency_tmp * dt_ /(-fmax(qi_tmp,SB_EPS)), rate);
                    if(rate > 1.0 && iter_count < MAX_ITER){
                        //Limit the timestep, but don't allow it to become vanishingly small
                        //Don't adjust if we have reached the maximum iteration number
                        dt_ = fmax(dt_/rate, 1.0e-3);
                    }
                    
                    // precip_rate, evap_rate and melting_rate are calculated for entropy balance formula equations;
                    // precip_tmp is NEGATIVE if rain/snow forms (+precip_tmp is to remove qt via precip formation);
                    // evap_tmp is NEGATIVE if rain/snow evaporate/sublimate (-evap_tmp is to add qt via evap/subl);
                    // ================================================
                    // ToDo: Need Double Check the computation of precip_rate and evap_rate, be careful about the MAGNITUDE
                    // ================================================
                    double precip_tmp = - qr_tendency_au - qr_tendency_ac - ql_tendency_frez - qi_tendency_nuc - qi_tendency_acc - qi_tendency_dep;
                    double evap_tmp   = qr_tendency_evap - qi_tendency_sub;
                    
                    precip_rate[ijk] += precip_tmp * dt_;
                    evap_rate[ijk]   += evap_tmp * dt_;
                    melt_rate[ijk]   += qi_tendency_melt * dt_; // NEGATIVE if snow melts to rain
                    
                    //Integrate forward in time
                    ql_tmp += ql_tendency_tmp * dt_;
                    nr_tmp += nr_tendency_tmp * dt_;
                    qr_tmp += qr_tendency_tmp * dt_;
                    ni_tmp += ni_tendency_tmp * dt_;
                    qi_tmp += qi_tendency_tmp * dt_;

                    qv_tmp += -(qr_tendency_evap+qi_tendency_dep) * dt_;
                    qr_tmp = fmax(qr_tmp,0.0);
                    nr_tmp = fmax(fmin(nr_tmp, qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                    qi_tmp = fmax(qi_tmp,0.0);
                    ni_tmp = fmax(fmin(ni_tmp, qi_tmp/ICE_MIN_MASS),qi_tmp/ICE_MAX_MASS);
                    ni_tmp = fmax(ni_tmp,0.0);
                    ql_tmp = fmax(ql_tmp,0.0);
                    qt_tmp = ql_tmp + qv_tmp;

                    // isotope tracers Intergrate forward in time
                    qi_iso_O18_tmp += qi_iso_O18_tendency_tmp * dt_;
                    qr_iso_O18_tmp += qr_iso_O18_tendency_tmp * dt_;
                    ql_iso_O18_tmp += ql_iso_O18_tendency_tmp * dt_;
                    qv_iso_O18_tmp += qv_iso_O18_tendency_tmp * dt_;

                    qi_iso_O18_tmp = fmax(qi_iso_O18_tmp, 0.0);
                    qr_iso_O18_tmp = fmax(qr_iso_O18_tmp, 0.0);
                    ql_iso_O18_tmp = fmax(ql_iso_O18_tmp, 0.0);
                    
                    qi_iso_HDO_tmp += qi_iso_HDO_tendency_tmp * dt_;
                    qr_iso_HDO_tmp += qr_iso_HDO_tendency_tmp * dt_;
                    ql_iso_HDO_tmp += ql_iso_HDO_tendency_tmp * dt_;
                    qv_iso_HDO_tmp += qv_iso_HDO_tendency_tmp * dt_;

                    qi_iso_HDO_tmp = fmax(qi_iso_HDO_tmp, 0.0);
                    qr_iso_HDO_tmp = fmax(qr_iso_HDO_tmp, 0.0);
                    ql_iso_HDO_tmp = fmax(ql_iso_HDO_tmp, 0.0);

                    time_added += dt_ ;
                }while(time_added < dt);

                nr_tendency_micro[ijk] = (nr_tmp - nr[ijk])/dt;
                qr_tendency_micro[ijk] = (qr_tmp - qr[ijk])/dt;
                ni_tendency_micro[ijk] = (ni_tmp - ni[ijk])/dt;
                qi_tendency_micro[ijk] = (qi_tmp - qi[ijk])/dt;
                nr_tendency[ijk] += nr_tendency_micro[ijk];
                qr_tendency[ijk] += qr_tendency_micro[ijk];
                ni_tendency[ijk] += ni_tendency_micro[ijk];
                qi_tendency[ijk] += qi_tendency_micro[ijk];
                
                precip_rate[ijk] = precip_rate[ijk]/dt;
                evap_rate[ijk] = evap_rate[ijk]/dt;
                melt_rate[ijk] = melt_rate[ijk]/dt;
                
                // Isotope Tracer output
                qr_iso_O18_tendency_micro[ijk]  = (qr_iso_O18_tmp - qr_iso_O18[ijk])/dt;
                qr_iso_O18_tendency[ijk]       += qr_iso_O18_tendency_micro[ijk];
                qi_iso_O18_tendency_micro[ijk]  = (qi_iso_O18_tmp - qi_iso_O18[ijk])/dt;
                qi_iso_O18_tendency[ijk]       += qi_iso_O18_tendency_micro[ijk];

                qr_iso_HDO_tendency_micro[ijk]  = (qr_iso_HDO_tmp - qr_iso_HDO[ijk])/dt;
                qr_iso_HDO_tendency[ijk]       += qr_iso_HDO_tendency_micro[ijk];
                qi_iso_HDO_tendency_micro[ijk]  = (qi_iso_HDO_tmp - qi_iso_HDO[ijk])/dt;
                qi_iso_HDO_tendency[ijk]       += qi_iso_HDO_tendency_micro[ijk];
            }
        }
    }
    return;
}
