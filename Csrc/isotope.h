#pragma once
#include "parameters.h"
#include "isotope_functions.h"
#include <math.h>
#include "thermodynamics_sa.h"
#include "microphysics_sb.h"
#include "microphysics.h"
#include "entropies.h"
#include "thermodynamic_functions.h"
// #define SB_EPS 1.0e-13

void iso_equilibrium_fractionation_No_Microphysics(struct DimStruct *dims, double* restrict t,
    double* restrict qt_std, double* restrict qv_std, double* restrict ql_std, 
    double* restrict qt_iso, double* restrict qv_iso, double* restrict ql_iso, 
    double* restrict qv_DV, double* restrict ql_DV){ 
    ssize_t i,j,k;
    double alpha_eq_O18 = 0.0;
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
                    double qv_std_tmp, ql_std_tmp, qv_iso_tmp, ql_iso_tmp;
                    alpha_eq_O18 = equilibrium_fractionation_factor_H2O18(t[ijk]);
                    qv_std_tmp = eq_frac_function(qt_std[ijk], qv_DV[ijk], ql_DV[ijk], 1.0);
                    qv_iso_tmp = eq_frac_function(qt_iso[ijk], qv_DV[ijk], ql_DV[ijk], alpha_eq_O18);
                    ql_std_tmp = qt_std[ijk] - qv_std_tmp;
                    ql_iso_tmp = qt_iso[ijk] - qv_iso_tmp;
                    
                    qv_std[ijk] = qv_std_tmp;
                    ql_std[ijk] = ql_std_tmp;
                    qv_iso[ijk] = qv_iso_tmp;
                    ql_iso[ijk] = ql_iso_tmp;
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


void tracer_sb_microphysics_sources(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                             double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
                             double* restrict density, double* restrict p0,  double* restrict temperature,  double* restrict qt, double ccn,
                             double* restrict ql, double* restrict nr, double* restrict qr, double dt,
                             double* restrict nr_tendency_micro, double* restrict qr_tendency_micro, double* restrict nr_tendency, double* restrict qr_tendency,
                             double* restrict qr_std_tendency, double* restrict qr_std_tendency_micro, 
                             double* restrict qr_iso, double* restrict qt_iso, double* restrict qv_iso, double* restrict ql_iso,
                             double* restrict qr_iso_tendency_micro, double* restrict qr_iso_tendency){

    //Here we compute the source terms for nr and qr (number and mass of rain)
    //Temporal substepping is used to help ensure boundedness of moments
    double rain_mass, Dm, mu, Dp, nr_tendency_tmp, qr_tendency_tmp, ql_tendency_tmp;
    double nr_tendency_au, nr_tendency_scbk, nr_tendency_evp;
    double qr_tendency_au, qr_tendency_ac,  qr_tendency_evp;
    double sat_ratio;
    double qr_iso_tmp, qr_iso_tend, qr_iso_tendency_tmp, qt_iso_tendency_tmp, qv_iso_tendency_tmp, ql_iso_tendency_tmp;
    double qr_iso_auto_tendency, qr_iso_accre_tendency, qr_iso_evap_tendency;


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
                double qv_tmp = qt[ijk] - fmax(ql[ijk],0.0);
                double qt_tmp = qt[ijk];
                double nl = ccn/density[k];
                double ql_tmp = fmax(ql[ijk],0.0);
                double qr_tmp = fmax(qr[ijk],0.0);
                double nr_tmp = fmax(fmin(nr[ijk], qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                double g_therm = microphysics_g(LT, lam_fp, L_fp, temperature[ijk]);
                double ql_iso_tmp = fmax(ql_iso[ijk], 0.0);
                double qr_iso_tmp = fmax(qr_iso[ijk], 0.0);
                double qv_iso_tmp = qv_iso[ijk];

                //holding nl fixed since it doesn't change between timesteps
                double time_added = 0.0, dt_, rate;
                ssize_t iter_count = 0;
                do{
                    iter_count += 1;
                    sat_ratio = microphysics_saturation_ratio(LT, temperature[ijk], p0[k], qt_tmp);
                    nr_tendency_au = 0.0;
                    nr_tendency_scbk = 0.0;
                    nr_tendency_evp = 0.0;
                    qr_tendency_au = 0.0;
                    qr_tendency_ac = 0.0;
                    qr_tendency_evp = 0.0;
                    //obtain some parameters
                    rain_mass = microphysics_mean_mass(nr_tmp, qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS);
                    Dm = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi);
                    mu = rain_mu(density[k], qr_tmp, Dm);
                    Dp = Dm * cbrt(tgamma(mu + 1.0) / tgamma(mu + 4.0));
                    //compute the source terms
                    sb_autoconversion_rain(droplet_nu, density[k], nl, ql_tmp, qr_tmp, &nr_tendency_au, &qr_tendency_au);
                    sb_accretion_rain(density[k], ql_tmp, qr_tmp, &qr_tendency_ac);
                    sb_selfcollection_breakup_rain(density[k], nr_tmp, qr_tmp, mu, rain_mass, Dm, &nr_tendency_scbk);
                    sb_evaporation_rain( g_therm, sat_ratio, nr_tmp, qr_tmp, mu, rain_mass, Dp, Dm, &nr_tendency_evp, &qr_tendency_evp);
                    //find the maximum substep time
                    dt_ = dt - time_added;
                    //check the source term magnitudes
                    nr_tendency_tmp = nr_tendency_au + nr_tendency_scbk + nr_tendency_evp;
                    qr_tendency_tmp = qr_tendency_au + qr_tendency_ac + qr_tendency_evp;
                    ql_tendency_tmp = -qr_tendency_au - qr_tendency_ac;

                    //iso_tendencies initilize
                    qr_iso_auto_tendency = 0.0;
                    qr_iso_accre_tendency = 0.0;
                    qr_iso_evap_tendency = 0.0;

                    // iso_tendencies calculations
                    sb_iso_rain_autoconversion(ql_tmp, ql_iso_tmp, qr_tendency_au, &qr_iso_auto_tendency);
                    sb_iso_rain_accretion(ql_tmp, qr_tmp, ql_iso_tmp, qr_iso_tmp, qr_tendency_ac, &qr_iso_accre_tendency);
                    double g_therm_iso = microphysics_g_iso(LT, lam_fp, L_fp, temperature[ijk], p0[k], qr_tmp, qr_iso_tmp, qv_tmp, qv_iso_tmp, sat_ratio);
                    sb_iso_evaporation_rain(g_therm_iso, sat_ratio, nr_tmp, qr_tmp, mu, qr_iso_tmp, rain_mass, Dp, Dm, &qr_iso_evap_tendency);
                    
                    // iso_tendencies add
                    qr_iso_tendency_tmp = qr_iso_auto_tendency + qr_iso_accre_tendency + qr_iso_evap_tendency;
                    ql_iso_tendency_tmp = -qr_iso_auto_tendency - qr_iso_accre_tendency;

                    //Factor of 1.05 is ad-hoc
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
                    qr_tmp = fmax(qr_tmp,0.0);
                    nr_tmp = fmax(fmin(nr_tmp, qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                    ql_tmp = fmax(ql_tmp,0.0);
                    qt_tmp = ql_tmp + qv_tmp;
                    
                    // isotope_tracer Intergrate forward in time
                    qr_iso_tmp += qr_iso_tendency_tmp * dt_;
                    ql_iso_tmp += ql_iso_tendency_tmp * dt_;
                    qv_iso_tmp += -qr_iso_evap_tendency * dt_;

                    qr_iso_tmp = fmax(qr_iso_tmp, 0.0);
                    ql_iso_tmp = fmax(ql_iso_tmp, 0.0);

                    time_added += dt_ ;
                }while(time_added < dt);
                nr_tendency_micro[ijk] = (nr_tmp - nr[ijk] )/dt;
                qr_tendency_micro[ijk] = (qr_tmp - qr[ijk])/dt;
                qr_std_tendency_micro[ijk] = (qr_tmp - qr[ijk])/dt;
                qr_iso_tendency_micro[ijk] = (qr_iso_tmp - qr_iso[ijk])/dt;
               
                nr_tendency[ijk] += nr_tendency_micro[ijk];
                qr_tendency[ijk] += qr_tendency_micro[ijk];
                qr_std_tendency[ijk] += qr_std_tendency_micro[ijk];
                qr_iso_tendency[ijk] += qr_iso_tendency_micro[ijk];
                
            }
        }
    }
    return;
}
void tracer_sb_sedimentation_velocity_rain(const struct DimStruct *dims, double (*rain_mu)(double,double,double),
                                        double* restrict density, double* restrict nr, double* restrict qr, 
                                        double* restrict nr_velocity, double* restrict qr_velocity, double* restrict qr_std_velocity, double* restrict qr_iso_velocity){
    
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
                double qr_tmp = fmax(qr[ijk],0.0);
                double density_factor = sqrt(DENSITY_SB/density[k]);
                double rain_mass = microphysics_mean_mass(nr[ijk], qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS);
                double Dm = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi);
                double mu = rain_mu(density[k], qr_tmp, Dm);
                double Dp = Dm * cbrt(tgamma(mu + 1.0) / tgamma(mu + 4.0));

                nr_velocity[ijk] = -fmin(fmax( density_factor * (A_RAIN_SED - B_RAIN_SED * pow(1.0 + C_RAIN_SED * Dp, -mu - 1.0)) , 0.0),10.0);
                qr_velocity[ijk] = -fmin(fmax( density_factor * (A_RAIN_SED - B_RAIN_SED * pow(1.0 + C_RAIN_SED * Dp, -mu - 4.0)) , 0.0),10.0);
                qr_std_velocity[ijk] = -fmin(fmax( density_factor * (A_RAIN_SED - B_RAIN_SED * pow(1.0 + C_RAIN_SED * Dp, -mu - 4.0)) , 0.0),10.0);
                qr_iso_velocity[ijk] = -fmin(fmax( density_factor * (A_RAIN_SED - B_RAIN_SED * pow(1.0 + C_RAIN_SED * Dp, -mu - 4.0)) , 0.0),10.0);

            }
        }
    }


     for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax-1 ; k++){
                const ssize_t ijk = ishift + jshift + k;

                nr_velocity[ijk] = interp_2(nr_velocity[ijk], nr_velocity[ijk+1]) ;
                qr_velocity[ijk] = interp_2(qr_velocity[ijk], qr_velocity[ijk+1]) ;
                qr_std_velocity[ijk] = interp_2(qr_std_velocity[ijk], qr_std_velocity[ijk+1]) ;
                qr_iso_velocity[ijk] = interp_2(qr_iso_velocity[ijk], qr_iso_velocity[ijk+1]) ;
            }
        }
    }
    return;
}

void tracer_sb_qt_source_formation(const struct DimStruct *dims, double* restrict qr_tendency, double* restrict qr_iso_tendency, 
                                    double* restrict qt_tendency, double* restrict qt_std_tendency, double* restrict qt_iso_tendency){

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
                qt_tendency[ijk] += -qr_tendency[ijk];
                qt_std_tendency[ijk] += -qr_tendency[ijk];
                qt_iso_tendency[ijk] += -qr_iso_tendency[ijk];
            }
        }
    }
    return;
}