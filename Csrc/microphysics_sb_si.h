#pragma once
#include "parameters.h"
#include "parameters_micro_sb.h"
#include "microphysics.h"
#include "microphysics_sb.h"
#include "advection_interpolation.h"
#include "entropies.h"
#include "thermodynamic_functions.h"
#include <math.h>

#define C1_AM 4e-3
#define C2_AM 6e-5
#define C3_AM 0.15
#define C1_BM 1.85
#define C3_BM 1.25
#define D1_AA 1.28
#define C2_BM 3e-3
#define D2_AA -0.012
#define D3_AA -0.6
#define D1_BA 1.5
#define D2_BA 7.5e-4
#define D3_BA 0.5
#define E1 1.08
#define E2 0.449
#define DENSITY_SNOW 250

// ===========<<< single ice parameters adopted from Zhao etc. 2017 >>> ============
// Zhao etc. 2017, Equ(1)
double riming_intensity(double lwc, double iwc){
    return 1.0/(1.0 + 6e-5/(lwc*pow(iwc, 0.17)));
}

// Zhao etc. 2017, Equ(2), falling velocity related parameters
double power_law_parameters_aa_SI(double Ri, double temperature){
    return D1_AA + D2_AA*temperature + D3_AA*Ri;
}
double power_law_parameters_ba_SI(double Ri, double temperature){
    return D1_BA + D2_BA*temperature + D3_BA*Ri;
}

// Zhao etc. 2017. Equ(3)
double power_law_parameters_am_SI(double Ri, double temperature){
    return C1_AM + C2_AM*temperature + C3_AM*Ri*Ri;
}
double power_law_parameters_bm_SI(double Ri, double temperature){
    return C1_BM + C2_BM*temperature + C3_BM*Ri;
}

// Zhao etc. 2017 Equ (4)
double power_law_parameters_av_SI(double am, double aa){
    double var_base = 2.0 * g * am/DENSITY_SNOW * KIN_VISC_AIR * KIN_VISC_AIR * aa;
    return E1 * KIN_VISC_AIR * pow(var_base, E2);
}
double power_law_parameters_bv_SI(double bm, double ba){
    return E2 * (bm - ba + 2.0) - 1.0;
}

void sb_si_get_ice_parameters_SI(double Ri, double temperature, double ice_mass, 
        double* sb_a_ice, double* sb_b_ice, double* sb_alpha_ice, double* sb_beta_ice){
    double am = power_law_parameters_am_SI(Ri, temperature);
    double bm = power_law_parameters_bm_SI(Ri, temperature);
    double aa = power_law_parameters_am_SI(Ri, temperature);
    double ba = power_law_parameters_bm_SI(Ri, temperature);
    double av = power_law_parameters_av_SI(am, aa);
    double bv = power_law_parameters_bv_SI(bm, ba);

    double ice_dm_exponent = 1.0/bm;
    double ice_dm_prefactor = pow(1.0/am, ice_dm_exponent);
    *sb_a_ice = ice_dm_prefactor;
    *sb_b_ice = ice_dm_exponent;
    *sb_alpha_ice = av*pow(am, bv);
    *sb_beta_ice = bm*bv;
    return;
}

// another scheme used in Zhao etc. 2017, SIFI 
// SIFI scheme is mostly based on MG08 scheme
double power_law_parameters_am_SIFI(void){
    return pi/6*DENSITY_SNOW;
}
double power_law_parameters_bm_SIFI(void){
    return 3.0;
}
double power_law_parameters_aa_SIFI(void){
    return pi/4;
}
double power_law_parameters_ba_SIFI(void){
    return 2.0;
}
double power_law_parameters_av_SIFI(void){
    return 11.72; 
}
double power_law_parameters_bv_SIFI(void){
    return 0.41;
}

void sb_si_get_ice_parameters_SIFI(double* sb_a_ice, double* sb_b_ice, 
        double* sb_alpha_ice, double* sb_beta_ice){
    double am = power_law_parameters_am_SIFI();
    double bm = power_law_parameters_bm_SIFI();
    double av = power_law_parameters_av_SIFI();
    double bv = power_law_parameters_bv_SIFI();

    double ice_dm_exponent = 1.0/bm;
    double ice_dm_prefactor = pow(1.0/am, ice_dm_exponent);

    *sb_a_ice = ice_dm_prefactor;
    *sb_b_ice = ice_dm_exponent;
    *sb_alpha_ice = av*pow(am, bv);
    *sb_beta_ice = bm*bv;
    return;
}

void sb_si_microphysics_sources(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                             double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
                             double* restrict density, double* restrict p0,  double* restrict temperature,  double* restrict qt, double ccn,
                             double* restrict ql, double* restrict nr, double* restrict qr, double* restrict qi, double* restrict ni, double dt,
                             double* restrict nr_tendency_micro, double* restrict qr_tendency_micro, double* restrict nr_tendency, double* restrict qr_tendency, 
                             double* restrict ni_tendency_micro, double* restrict qi_tendency_micro, double* restrict ni_tendency, double* restrict qi_tendency, 
                             double* restrict precip_rate, double* restrict evap_rate, double* restrict melt_rate){

    //Here we compute the source terms for nr and qr (number and mass of rain)
    //Temporal substepping is used to help ensure boundedness of moments
    double rain_mass, Dm_r, mu, Dp, nr_tendency_tmp, qr_tendency_tmp, ql_tendency_tmp, ql_tendency_frez, nl_tendency_acc;
    double liquid_mass, Dm_l;
    double nr_tendency_au, nr_tendency_scbk, nr_tendency_evap, nr_tendency_frez;
    double qr_tendency_au, qr_tendency_ac, qr_tendency_evap, qr_tendency_frez;
    // single ice tendency definition
    double Dm_i, ice_vel, sb_a_ice, sb_b_ice, sb_alpha_ice, sb_beta_ice, ice_mass;
    double qi_tendency_tmp, ni_tendency_tmp;
    double ni_tendency_nuc, ni_tendency_frez, ni_tendency_berg, ni_tendency_melt;
    double qi_tendency_nuc, qi_tendency_frez, qi_tendency_acc, qi_tendency_dep, qi_tendency_berg, qi_tendency_melt, qi_tendency_sub;

    double sat_ratio;
    

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
                // holding nl fixed since it doesn't change between timesteps

                // double nr_org = fmax(fmin(nr[ijk], qr_org/RAIN_MIN_MASS),qr_org/RAIN_MAX_MASS);
                double qr_tmp = fmax(qr[ijk],0.0);
                double nr_tmp = fmax(fmin(nr[ijk], qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                
                double qi_tmp = fmax(qi[ijk],0.0);
                double ni_tmp = fmax(fmin(ni[ijk], qi_tmp/ICE_MIN_MASS),qi_tmp/ICE_MAX_MASS);

                double g_therm = microphysics_g(LT, lam_fp, L_fp, temperature[ijk]);

                
                precip_rate[ijk] = 0.0;
                evap_rate[ijk] = 0.0;
                melt_rate[ijk] = 0.0;

                double time_added = 0.0, dt_, rate;
                ssize_t iter_count = 0;
                do{
                    qi_tendency_tmp   = 0.0;
                    ni_tendency_tmp   = 0.0;
                    qr_tendency_tmp   = 0.0;
                    nr_tendency_tmp   = 0.0;
                    ql_tendency_tmp   = 0.0;

                    iter_count       += 1;
                    sat_ratio         = microphysics_saturation_ratio(LT, temperature[ijk], p0[k], qt_tmp);
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
                    sb_si_get_ice_parameters_SIFI(&sb_a_ice, &sb_b_ice, &sb_alpha_ice, &sb_beta_ice);
                    Dm_i     = sb_a_ice * pow(ice_mass, sb_b_ice);
                    ice_vel  = sb_alpha_ice * pow(ice_mass, sb_beta_ice);

                    //compute the source terms
                    sb_nucleation_ice(temperature[ijk], sat_ratio, dt_, ni_tmp, &qi_tendency_nuc, &ni_tendency_nuc);
                    sb_autoconversion_rain(droplet_nu, density[k], nl, ql_tmp, qr_tmp, &nr_tendency_au, &qr_tendency_au);
                    sb_deposition_ice(LT, lam_fp, L_fp, temperature[ijk], Dm_i, sat_ratio, ice_mass, ice_vel,
                            qi_tmp, ni_tmp, &qi_tendency_dep);
                    sb_sublimation_ice(LT, lam_fp, L_fp, temperature[ijk], Dm_i, sat_ratio, ice_mass, ice_vel,
                            qi_tmp, ni_tmp, &qi_tendency_sub);
                    // sb_freezing_ice(droplet_nu, temperature[ijk], density[k], liquid_mass, rain_mass, ql_tmp, nl, qr_tmp, nr_tmp,  
                    //         &ql_tendency_frez, &qr_tendency_frez, &nr_tendency_frez, &ni_tendency_frez, &qi_tendency_frez);
                    sb_accretion_rain(density[k], ql_tmp, qr_tmp, &qr_tendency_ac); 
                    sb_accretion_cloud_ice(liquid_mass, Dm_l, ice_mass, Dm_i, ice_vel, nl, ql_tmp, ni_tmp, qi_tmp, 
                            sb_a_ice, sb_b_ice, sb_beta_ice, &nl_tendency_acc, &qi_tendency_acc);
                    sb_selfcollection_breakup_rain(density[k], nr_tmp, qr_tmp, mu, rain_mass, Dm_r, &nr_tendency_scbk);
                    sb_evaporation_rain(g_therm, sat_ratio, nr_tmp, qr_tmp, mu, rain_mass, Dp, Dm_r, &nr_tendency_evap, &qr_tendency_evap);
                    sb_melting_ice(LT, lam_fp, L_fp, temperature[ijk], ice_mass, Dm_i, qv_tmp, ni_tmp, qi_tmp, &ni_tendency_melt, &qi_tendency_melt);

                    //find the maximum substep time
                    dt_ = dt - time_added;

                    // check the source term magnitudes
                    // qi_tendency_sub is POSITIVE, qi_tendency_dep is POSITIVE;
                    // qr_tendency_evap is NEGATIVE;
                    // qi_tendency_melt and ni_tendency_melt are all POSITIVE
                    qi_tendency_tmp = qi_tendency_nuc + qi_tendency_frez + qi_tendency_acc + qi_tendency_dep + qi_tendency_berg + qi_tendency_sub - qi_tendency_melt;
                    ni_tendency_tmp = ni_tendency_nuc + ni_tendency_frez + ni_tendency_berg - ni_tendency_melt;

                    nr_tendency_tmp = nr_tendency_au + nr_tendency_scbk + nr_tendency_evap + ni_tendency_melt - nr_tendency_frez;
                    qr_tendency_tmp = qr_tendency_au + qr_tendency_ac + qr_tendency_evap + qi_tendency_melt - qr_tendency_frez;

                    ql_tendency_tmp = -qr_tendency_au - qr_tendency_ac - ql_tendency_frez - qi_tendency_acc;

                    //Factor of 1.05 is ad-hoc
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
            }
        }
    }
    return;
}

void sb_si_qt_source_formation(const struct DimStruct *dims, double* restrict qi_tendency,
                               double* restrict qr_tendency, double* restrict qt_tendency){

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
                qt_tendency[ijk] += -qr_tendency[ijk] - qi_tendency[ijk];
            }
        }
    }
    return;
}

void sb_sedimentation_velocity_ice(const struct DimStruct *dims, double* restrict ni, double* restrict qi,
        double* restrict ni_velocity, double* restrict qi_velocity){

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
                double sb_a_ice, sb_b_ice, sb_alpha_ice, sb_beta_ice;
                sb_si_get_ice_parameters_SIFI(&sb_a_ice, &sb_b_ice, &sb_alpha_ice, &sb_beta_ice);
                double ice_mass = microphysics_mean_mass(ni[ijk], qi[ijk], ICE_MIN_MASS, ICE_MAX_MASS);
                double vel_tmp_1 = gamma(6.0)/gamma(9.0);
                double ni_vel_tmp = sb_alpha_ice * gamma(6.0 + 3.0*sb_beta_ice)/gamma(6.0) * pow(vel_tmp_1, sb_beta_ice) * pow(ice_mass, sb_beta_ice);
                double qi_vel_tmp = sb_alpha_ice * gamma(9.0 + 3.0*sb_beta_ice)/gamma(9.0) * pow(vel_tmp_1, sb_beta_ice) * pow(ice_mass, sb_beta_ice);
                ni_velocity[ijk] = -fmin(fmax( ni_vel_tmp, 0.0),10.0);
                qi_velocity[ijk] = -fmin(fmax( qi_vel_tmp, 0.0),10.0);

            }
        }
    }
     for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax-1 ; k++){
                const ssize_t ijk = ishift + jshift + k;
                ni_velocity[ijk] = interp_2(ni_velocity[ijk], ni_velocity[ijk+1]) ;
                qi_velocity[ijk] = interp_2(qi_velocity[ijk], qi_velocity[ijk+1]) ;
            }
        }
    }
    return;
}

void sb_si_entropy_source_precipitation(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double),
                              double (*L_fp)(double, double), double* restrict p0, double* restrict temperature,
                              double* restrict qt, double* restrict qv, double* restrict qr_tendency,
                              double* restrict precip_rate, double* restrict entropy_tendency){

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

void sb_si_entropy_source_evaporation(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double),
                              double (*L_fp)(double, double), double* restrict p0, double* restrict temperature,
                              double* restrict Twet, double* restrict qt, double* restrict qv, double* restrict qr_tendency,
                              double* restrict evap_rate, double* restrict entropy_tendency){

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

void sb_si_entropy_source_melt(const struct DimStruct *dims, double* restrict temperature,
                         double* restrict melt_rate, double* restrict entropy_tendency){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    //entropy tendencies from snow melt
    //we use fact that melt_rate is negative when snow becomes rain
    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                entropy_tendency[ijk] += melt_rate[ijk] * LHF / temperature[ijk];
            }
        }
    }
    return;
};

void sb_si_entropy_source_heating_rain(const struct DimStruct *dims, double* restrict temperature, double* restrict Twet, double* restrict qrain,
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

void sb_si_entropy_source_heating_snow(const struct DimStruct *dims, double* restrict temperature, double* restrict Twet, double* restrict qsnow,
                               double* restrict w_qsnow, double* restrict w,  double* restrict entropy_tendency){

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
                entropy_tendency[ijk]+= qsnow[ijk]*(fabs(w_qsnow[ijk]) - w[ijk]) * ci * (Twet[ijk+1] - Twet[ijk])* dzi/temperature[ijk];
            }
        }
    }
    return;

};

void sb_si_entropy_source_drag(const struct DimStruct *dims, double* restrict temperature,  double* restrict qprec,
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
