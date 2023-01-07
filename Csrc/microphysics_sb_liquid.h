#pragma once
#include "microphysics_arctic_1m.h"
#include "parameters.h"
#include "parameters_micro_sb.h"
#include "microphysics.h"
#include "microphysics_sb.h"
#include "advection_interpolation.h"
#include "entropies.h"
#include "thermodynamic_functions.h"
#include <math.h>

void sb_liquid_microphysics_sources(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
    double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
    double* restrict density, double* restrict p0,  double* restrict temperature,  double* restrict qt, double ccn,
    double* restrict ql, double* restrict nr, double* restrict qr, double dt,
    double* restrict nr_tendency_micro, double* restrict qr_tendency_micro, double* restrict nr_tendency, double* restrict qr_tendency,
    double* restrict precip_rate, double*restrict evap_rate){

    //Here we compute the source terms for nr and qr (number and mass of rain)
    //Temporal substepping is used to help ensure boundedness of moments
    double rain_mass, Dm, mu, Dp, nr_tendency_tmp, qr_tendency_tmp, ql_tendency_tmp;
    double nr_tendency_au, nr_tendency_scbk, nr_tendency_evp;
    double qr_tendency_au, qr_tendency_ac,  qr_tendency_evp;
    double sat_ratio;
    double precip_tmp, evap_tmp;

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
                double qv_tmp  = qt[ijk] - fmax(ql[ijk],0.0);
                double qt_tmp  = qt[ijk];
                double nl      = ccn/density[k];
                double ql_tmp  = fmax(ql[ijk],0.0);
                double qr_tmp  = fmax(qr[ijk],0.0);
                double nr_tmp  = fmax(fmin(nr[ijk], qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                double g_therm = microphysics_g(LT, lam_fp, L_fp, temperature[ijk]);

                //holding nl fixed since it doesn't change between timesteps

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
                    
                    precip_tmp = 0.0;
                    evap_tmp = 0.0;

                    //obtain some parameters
                    rain_mass = microphysics_mean_mass(nr_tmp, qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS);
                    Dm = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi);
                    mu = rain_mu(density[k], qr_tmp, Dm);
                    Dp = sb_Dp(Dm, mu);
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

                    //Factor of 1.05 is ad-hoc
                    rate = 1.05 * ql_tendency_tmp * dt_ /(- fmax(ql_tmp,SB_EPS));
                    rate = fmax(1.05 * nr_tendency_tmp * dt_ /(-fmax(nr_tmp,SB_EPS)), rate);
                    rate = fmax(1.05 * qr_tendency_tmp * dt_ /(-fmax(qr_tmp,SB_EPS)), rate);
                    if(rate > 1.0 && iter_count < MAX_ITER){
                        //Limit the timestep, but don't allow it to become vanishingly small
                        //Don't adjust if we have reached the maximum iteration number
                        dt_ = fmax(dt_/rate, 1.0e-3);
                    }
                    
                    // precip_tmp is NEGATIVE if rain forms (+precip_tmp is to remove qt via precip formation);
                    // evap_tmp is NEGATIVE if rain evaporate (-evap_tmp is to add qt via evap/subl);
                    precip_tmp = - qr_tendency_au - qr_tendency_ac;
                    evap_tmp = qr_tendency_evp;

                    precip_rate[ijk] += precip_tmp * dt_;
                    evap_rate[ijk] += evap_tmp * dt_;
                                                                 //
                    //Integrate forward in time
                    ql_tmp += ql_tendency_tmp * dt_;
                    nr_tmp += nr_tendency_tmp * dt_;
                    qr_tmp += qr_tendency_tmp * dt_;
                    qv_tmp += -qr_tendency_evp * dt_;
                    qr_tmp = fmax(qr_tmp,0.0);
                    nr_tmp = fmax(fmin(nr_tmp, qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                    ql_tmp = fmax(ql_tmp,0.0);
                    qt_tmp = ql_tmp + qv_tmp;
                    time_added += dt_ ;
                }while(time_added < dt);
                nr_tendency_micro[ijk] = (nr_tmp - nr[ijk] )/dt;
                qr_tendency_micro[ijk] = (qr_tmp - qr[ijk])/dt;
                nr_tendency[ijk] += nr_tendency_micro[ijk];
                qr_tendency[ijk] += qr_tendency_micro[ijk];

                precip_rate[ijk] = precip_rate[ijk]/dt;
                evap_rate[ijk] = evap_rate[ijk]/dt;
            }
        }
    }
    return;
}

void sb_qt_source_formation(const struct DimStruct *dims,double* restrict qr_tendency, double* restrict qt_tendency ){

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
            }
        }
    }
    return;
}

// ===========<<< microphysics process's effect on thermodynamic process >>> ============

void sb_entropy_source_formation(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
    double* restrict p0, double* restrict T, double* restrict Twet, double* restrict qt, double* restrict qv,
    double* restrict qr_tendency,  double* restrict entropy_tendency){

    //Here we compute the source terms of total water and entropy related to microphysics. See Pressel et al. 2015, Eq. 49-54
    //
    //Some simplifications are possible because there is only a single hydrometeor species (rain), so d(qr)/dt
    //can only represent a transfer to/from the equilibrium mixture. Furthermore, formation and evaporation of
    //rain cannot occur simultaneously at the same physical location so keeping track of each sub-tendency is not required

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    //entropy tendencies from formation or evaporation of precipitation
    //we use fact that P = d(qr)/dt > 0, E =  d(qr)/dt < 0
    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                const double lam_T = lam_fp(T[ijk]);
                const double L_fp_T = L_fp(T[ijk],lam_T);
                const double lam_Tw = lam_fp(Twet[ijk]);
                const double L_fp_Tw = L_fp(Twet[ijk],lam_Tw);
                const double pv_star_T = lookup(LT, T[ijk]);
                const double pv_star_Tw = lookup(LT,Twet[ijk]);
                const double pv = pv_c(p0[k], qt[ijk], qv[ijk]);
                const double pd = p0[k] - pv;
                const double sd_T = sd_c(pd, T[ijk]);
                const double sv_star_T = sv_c(pv_star_T,T[ijk] );
                const double sv_star_Tw = sv_c(pv_star_Tw, Twet[ijk]);
                const double S_P = sd_T - sv_star_T + L_fp_T/T[ijk]; //Pressel15, Equ 49
                const double S_E = sv_star_Tw - L_fp_Tw/Twet[ijk] - sd_T; //Pressel15, Equ 50  
                const double S_D = -Rv * log(pv/pv_star_T) + cpv * log(T[ijk]/Twet[ijk]); //Pressel15, Equ 51 
                // add SP SE and SD in Pressel15 
                entropy_tendency[ijk] += S_P * 0.5 * (qr_tendency[ijk] + fabs(qr_tendency[ijk])) - (S_E + S_D) * 0.5 *(qr_tendency[ijk] - fabs(qr_tendency[ijk]));
            }
        }
    }
    return;
}

void sb_entropy_source_heating(const struct DimStruct *dims, double* restrict T, double* restrict Twet, double* restrict qr,
    double* restrict w_qr, double* restrict w,  double* restrict entropy_tendency){

    //derivative of Twet is upwinded

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;
    const double dzi = 1.0/dims->dx[2];

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                // Q_P = qr[ijk]*(fabs(w_qr[ijk]) - w[ijk]) * cl * (Twet[ijk+1] - Twet[ijk])* dzi;
                // Q_P should following the Pressel15 Equ 52
                entropy_tendency[ijk]+= qr[ijk]*(fabs(w_qr[ijk]) - w[ijk]) * cl * (Twet[ijk+1] - Twet[ijk])* dzi / T[ijk];
            }
        }
    }
    return;
}

// following  Pressel15 Equ 54.
void sb_entropy_source_drag(const struct DimStruct *dims, double* restrict T,  double* restrict qr,
    double* restrict w_qr, double* restrict entropy_tendency){

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
                entropy_tendency[ijk]+= g * qr[ijk]* fabs(w_qr[ijk])/ T[ijk];
            }
        }
    }
    return;
}

void sb_liquid_entropy_source_precipitation(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double),
    double (*L_fp)(double, double), double* restrict p0, double* restrict temperature,
    double* restrict qt, double* restrict qv, double* precip_rate,
    double* restrict qr_tendency, double* restrict entropy_tendency){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    //entropy tendencies from formation of snow and rain
    //we use fact that P = d(qr)/dt > 0, E =  d(qr)/dt < 0
    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                double lam = lam_fp(temperature[ijk]);
                double L = L_fp(temperature[ijk],lam);
                double pv_star_T = lookup(LT, temperature[ijk]);

                // following function to calculate P is used in original Arc1m, where precip_rate is calculated during microphysics_source;
                // precip_rate[ijk] < 0.0; so need make -precip_rate[ijk] 
                // double precip_rate_tmp = -precip_rate[ijk];

                // following function to calculate P is used in original SB06;
                // precip_rate_tmp > 0.0;
                double precip_rate_tmp = 0.5 * (qr_tendency[ijk] + fabs(qr_tendency[ijk]));
                entropy_tendency[ijk] += entropy_src_precipitation_c(pv_star_T, p0[k], temperature[ijk], qt[ijk], qv[ijk], L, precip_rate_tmp);

            }
        }
    }
    return;
};

// entropy source functions which are adopted from arctic 1m scheme
void sb_liquid_entropy_source_evaporation(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double),
    double (*L_fp)(double, double), double* restrict p0, double* restrict temperature,
    double* restrict Twet, double* restrict qt, double* restrict qv, 
    double* evap_rate, double* restrict qr_tendency, double* restrict entropy_tendency){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    //entropy tendencies from evaporation of rain and sublimation of snow
    //we use fact that P = d(qr)/dt > 0, E =  d(qr)/dt < 0
    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                double lam_Tw = lam_fp(Twet[ijk]);
                double L_Tw = L_fp(Twet[ijk],lam_Tw);
                const double pv_star_T = lookup(LT, temperature[ijk]);
                const double pv_star_Tw = lookup(LT, Twet[ijk]); 
                // following function to calculate E is used in original Arc1m, where evap_rate is calculated during microphysics_source;
                // evaporate rate E < 0.0;
                // double evap_rate_tmp = - evap_rate[ijk];

                // following function to calculate P is used in original SB06;
                // evap_rate_tmp > 0.0;
                double evap_rate_tmp = 0.5 *(qr_tendency[ijk] - fabs(qr_tendency[ijk]));
                entropy_tendency[ijk] -= entropy_src_evaporation_c(pv_star_T, pv_star_Tw, p0[k], temperature[ijk], Twet[ijk], qt[ijk], qv[ijk], L_Tw, evap_rate_tmp);
            }
        }
    }
    return;
};

void sb_liquid_entropy_source_heating_rain(const struct DimStruct *dims, double* restrict temperature, double* restrict Twet, double* restrict qrain,
    double* restrict w_qrain, double* restrict w,  double* restrict entropy_tendency){

    //derivative of Twet is upwinded

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;
    const double dzi = 1.0/dims->dx[2];

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                entropy_tendency[ijk]+= qrain[ijk]*(fabs(w_qrain[ijk]) - w[ijk]) * cl * (Twet[ijk+1] - Twet[ijk])* dzi/temperature[ijk];
            }
        }
    }
    return;
};

void sb_liquid_entropy_source_drag(const struct DimStruct *dims, double* restrict temperature,  double* restrict qprec,
    double* restrict w_qprec, double* restrict entropy_tendency){

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
                entropy_tendency[ijk]+= g * qprec[ijk]* fabs(w_qprec[ijk])/ temperature[ijk];
            }
        }
    }
    return;
};
