#pragma once
#include "parameters.h"
#include "microphysics.h"
#include "microphysics_sb.h"
#include "advection_interpolation.h"
#include "entropies.h"
#include "thermodynamic_functions.h"
// #include <cmath>
#include <cmath>
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
}

// ===========<<< SB06 accretion of ice and cloud droplets parameters >>> ============
// adopted in sb_accretion_cloud_ice()
// Seifert & Beheng 2006: Equ
void microphysics_sb_collision_parameters(double sb_a_ice, double sb_b_ice, double sb_beta_ice, double k,
        double* delta_li, double* delta_l, double* delta_i, double* vartheta_l, double* vartheta_li){
    // k: k-th moment
    double ice_mu_        = 3.0; // 1/mu_ice, and mu_ice =1/3.0
    double liquid_mu_     = 1.0; // liquid_mu_ = 1.0
    double nu             = 1.0; // both ice and cloud droplets
    double var_ice_1      = gamma(6.0); // Γ((nu+1)/ice_mu)
    double var_ice_2      = gamma(9.0); // Γ((nu+2)/ice_mu)
    double var_ice_3      = var_ice_1/var_ice_2;
    double var_ice_4      = (2*sb_b_ice + nu + 1.0 + k) * ice_mu_;
    double var_ice_5      = (sb_b_ice + nu + 1.0 +k) * ice_mu_;
    double var_ice_6      = (sb_beta_ice + sb_b_ice + nu + 1.0 + k) * ice_mu_;
    double var_ice_7      = (sb_b_ice + nu + 1.0 + k) * ice_mu_;

    double var_liquid_1   = gamma(2.0); // Γ((nu+1)/liquid_mu)
    double var_liquid_2   = gamma(3.0); // Γ((nu+2)/liquid_mu)
    double var_liquid_3   = var_liquid_1/var_liquid_2;
    double var_liquid_4   = 8.0 + 3.0*k; // (2*sb_b_liquid + nu + 1.0 + k)*ice_mu_
    double sb_b_liquid    = 1.0/3.0;
    double sb_beta_liquid = 2.0/3.0;
    double var_liquid_5   = 7.0 + 3.0*k; // (sb_b_liquid + nu + 1.0 + k)*ice_mu_
    double var_liquid_6   = 12.0 + 3.0*k; // (2*sb_beta_liquid + 2*sb_b_liquid + nu + 1.0 + k)*ice_mu_
   
    *delta_i     = gamma(var_ice_4)/var_ice_1 * pow(var_ice_3, (2*sb_b_ice+k));
    *delta_l     = gamma(var_liquid_4)/var_liquid_1 * pow(var_liquid_3, (2*sb_b_liquid+k));
    *delta_li    = 2.0 * gamma(var_ice_5)/var_ice_1 * gamma(7.0)/var_liquid_1 * pow(var_ice_3, (sb_b_ice+k)) * cbrt(var_liquid_3);
    
    *vartheta_l  = gamma(var_liquid_6)/gamma(var_liquid_4)*pow(var_liquid_3, 2*sb_beta_liquid);
    *vartheta_li = 2.0 * gamma(var_ice_6)/gamma(var_ice_7) * gamma(9.0)/gamma(7.0) * pow(var_ice_3, sb_beta_ice) * pow(var_liquid_3, sb_beta_liquid);
}

double microphysics_sb_E_il(double Dm_l, double Dm_i){
    double E_l, E_i;
    // calculation of E_l
    if(Dm_l < D_L0){
        E_l = 0.0;
    }
    else if(Dm_l <= D_L1){
        E_l = (Dm_l - D_L0)/(D_L1 - D_L0);
    }
    else{
        E_l = 1.0;
    }
    // calculation of E_i
    if(Dm_i <= D_I0){
        E_i = 0.0;
    }
    else{
        E_i = 0.8;
    }
    return E_l*E_i;
}

// another scheme used in Zhao etc. 2017, SIFI 
// SIFI scheme is mostly based on MG08 scheme
double power_law_parameters_am_SIFI(){
    return pi/6*DENSITY_SNOW;
}
double power_law_parameters_bm_SIFI(){
    return 3.0;
}
double power_law_parameters_aa_SIFI(){
    return pi/4;
}
double power_law_parameters_ba_SIFI(){
    return 2.0;
}
double power_law_parameters_av_SIFI(){
    return 11.72; 
}
double power_law_parameters_bv_SIFI(){
    return 0.41;
}

void sb_si_get_ice_parameters_SIFI(double ice_mass, double* sb_a_ice, double* sb_b_ice, 
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
}

void sb_si_microphysics_sources(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                             double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
                             double* restrict density, double* restrict p0,  double* restrict temperature,  double* restrict qt, double ccn,
                             double* restrict ql, double* restrict nr, double* restrict qr, double* restrict qi, double* restrict ni, double dt,
                             double* restrict nr_tendency_micro, double* restrict qr_tendency_micro, double* restrict nr_tendency, double* restrict qr_tendency, 
                             double* restrict ni_tendency_micro, double* restrict qi_tendency_micro, double* restrict ni_tendency, double* restrict ni_tendency){

    //Here we compute the source terms for nr and qr (number and mass of rain)
    //Temporal substepping is used to help ensure boundedness of moments
    double rain_mass, Dm_r, mu, Dp, nr_tendency_tmp, qr_tendency_tmp, ql_tendency_tmp, ql_tendency_frez, nl_tendency_acc;
    double liquid_mass, Dm_l;
    double nr_tendency_au, nr_tendency_scbk, nr_tendency_evp, nr_tendency_frez;
    double qr_tendency_au, qr_tendency_ac, qr_tendency_evp, qr_tendency_frez;
    // single ice tendency definition
    double Dm_i, ice_vel, sb_a_ice, sb_b_ice, sb_alpha_ice, sb_beta_ice, ice_mass;
    double qi_tendency_tmp, ni_tendency_tmp;
    double ni_tendency_nuc, ni_tendency_frez, ni_tendency_dep, ni_tendency_berg, ni_tendency_melt;
    double qi_tendency_nuc, qi_tendency_frez, qi_tendency_acc, qi_tendency_dep, qi_tendency_berg, qi_tendency_melt;

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

                double qv_tmp = qt[ijk] - fmax(ql[ijk],0.0);
                double qt_tmp = qt[ijk];
                double nl     = ccn/density[k];
                double ql_tmp = fmax(ql[ijk],0.0);

                double qr_org = fmax(qr[ijk],0.0);
                double nr_org = fmax(fmin(nr[ijk], qr_org/RAIN_MIN_MASS),qr_org/RAIN_MAX_MASS);
                double qr_tmp = fmax(qr[ijk],0.0);
                double nr_tmp = fmax(fmin(nr[ijk], qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                
                double qi_org = fmax(qi[ijk],0.0);
                double ni_org = fmax(fmin(ni[ijk], qi_org/RAIN_MIN_MASS),qi_org/RAIN_MAX_MASS);
                double qi_tmp = fmax(qi[ijk],0.0);
                double ni_tmp = fmax(fmin(ni[ijk], qi_tmp/RAIN_MIN_MASS),qi_tmp/RAIN_MAX_MASS);

                double g_therm = microphysics_g(LT, lam_fp, L_fp, temperature[ijk]);

                //holding nl fixed since it doesn't change between timesteps

                double time_added = 0.0, dt_, rate;
                ssize_t iter_count = 0;
                do{
                    iter_count       += 1;
                    sat_ratio         = microphysics_saturation_ratio(LT, temperature[ijk], p0[k], qt_tmp);
                    ql_tendency_frez  = 0.0;

                    nr_tendency_au    = 0.0;
                    nr_tendency_scbk  = 0.0;
                    nr_tendency_evp   = 0.0;
                    nr_tendency_frez  = 0.0;
                    qr_tendency_au    = 0.0;
                    qr_tendency_ac    = 0.0;
                    qr_tendency_evp   = 0.0;
                    qr_tendency_frez  = 0.0;

                    qi_tendency_nuc   = 0.0;
                    qi_tendency_acc   = 0.0;
                    qi_tendency_dep   = 0.0;
                    qi_tendency_frez  = 0.0;
                    qi_tendency_melt  = 0.0;
                    qi_tendency_berg  = 0.0;
                    ni_tendency_nuc   = 0.0;
                    ni_tendency_dep   = 0.0;
                    ni_tendency_frez  = 0.0;
                    ni_tendency_melt  = 0.0;
                    ni_tendency_berg  = 0.0;

                    //obtain some parameters of cloud droplets
                    liquid_mass = microphysics_mean_mass(nl, ql_tmp, LIQUID_MIN_MASS, LIQUID_MAX_MASS);// average mass of cloud droplets
                    Dm_l =  cbrt(liquid_mass * 6.0/DENSITY_LIQUID/pi)                    
                    
                    //obtain some parameters of rain droplets
                    rain_mass = microphysics_mean_mass(nr_tmp, qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS); //average mass of rain droplet
                    Dm_r      = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi); // mass weighted diameter of rain droplets
                    mu        = rain_mu(density[k], qr_tmp, Dm_r);
                    Dp        = sb_Dp(Dm_r, mu);

                    //obtain some parameters of ice particle
                    double Ri;
                    ice_mass = microphysics_mean_mass(ni_tmp, qi_tmp, ICE_MIN_MASS, ICE_MAX_MASS);
                    sb_si_get_ice_parameters_SIFI(&sb_a_ice, &sb_b_ice, &sb_alpha_ice, &sb_beta_ice);
                    Dm_i = sb_a_ice * pow(ice_mass, sb_b_ice);
                    ice_vel = sb_alpha_ice * pow(ice_mass, sb_beta_ice);

                    //compute the source terms
                    sb_nucleation_ice(sat_ratio, dt_, ni_tmp, &qi_tendency_nuc, &ni_tendency_nuc);
                    sb_autoconversion_rain(droplet_nu, density[k], nl_tmp, ql_tmp, qr_tmp, &nr_tendency_au, &qr_tendency_au);
                    sb_freezing_ice(droplet_nu, temperature[ijk], density[k], liquid_mass, rain_mass, ql_tmp, nl_tmp, qr_tmp, nr_tmp,  
                                    &ql_tendency_frez, &qr_tendency_frez, &nr_tendency_frez, &ni_tendency_frez, &qi_tendency_frez);
                    sb_deposition_ice(LT, lam_fp, L_fp, temperature[ijk], Dm_i, sat_ratio, ice_mass, ice_vel,
                                      qi_tmp, ni_tmp, &qi_tendency_dep, &ni_tendency_dep);
                    sb_accretion_rain(density[k], ql_tmp, qr_tmp, &qr_tendency_ac);
                    sb_accretion_cloud_ice(liquid_mass, Dm_l, ice_mass, Dm_i, ice_vel, nl_tmp, ql_tmp, ni_tmp, qi_tmp, &nl_tendency_acc, &qi_tendency_acc);
                    sb_selfcollection_breakup_rain(density[k], nr_tmp, qr_tmp, mu, rain_mass, Dm_r, &nr_tendency_scbk);
                    sb_evaporation_rain(g_therm, sat_ratio, nr_tmp, qr_tmp, mu, rain_mass, Dp, Dm_r, &nr_tendency_evp, &qr_tendency_evp);
                    sb_melting_ice(temperature[ijk], ice_mass, Dm_i, qv_tmp, ni_tmp, qi_tmp, &qi_tendency_melt, &qi_tendency_acc);

                    //find the maximum substep time
                    dt_ = dt - time_added;

                    // check the source term magnitudes
                    // 
                    qi_tendency_tep = qi_tendency_nuc + qi_tendency_frez + qi_tendency_acc + qi_tendency_dep + qi_tendency_berg - qi_tendency_melt;
                    ni_tendency_tep = ni_tendency_nuc + ni_tendency_frez + ni_tendency_dep + ni_tendency_berg - ni_tendency_melt;

                    nr_tendency_tmp = nr_tendency_au + nr_tendency_scbk + nr_tendency_evp + ni_tendency_melt - nr_tendency_frez;
                    qr_tendency_tmp = qr_tendency_au + qr_tendency_ac + qr_tendency_evp + qi_tendency_melt - qr_tendency_frez;

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
                    //Integrate forward in time
                    ql_tmp += ql_tendency_tmp * dt_;
                    nr_tmp += nr_tendency_tmp * dt_;
                    qr_tmp += qr_tendency_tmp * dt_;
                    ni_tmp += ni_tendency_tmp * dt_;
                    qi_tmp += qi_tendency_tmp * dt_;

                    qv_tmp += -(qr_tendency_evp+qi_tendency_dep) * dt_;
                    qr_tmp = fmax(qr_tmp,0.0);
                    nr_tmp = fmax(fmin(nr_tmp, qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                    qi_tmp = fmax(qi_tmp,0.0);
                    ni_tmp = fmax(fmin(ni_tmp, qi_tmp/ICE_MIN_MASS),qi_tmp/ICE_MAX_MASS);
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
            }
        }
    }
    return;
}
