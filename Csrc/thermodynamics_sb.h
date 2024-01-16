#pragma once
#include "parameters.h"
#include "parameters_micro_sb.h"
#include "grid.h"
#include "thermodynamic_functions.h"
#include "advection_interpolation.h"
#include "lookup.h"
#include "entropies.h"
#include "microphysics_sb_ice.h"
#include <stdio.h>

void saturation_ratio(const struct DimStruct *dims, 
        // thermodynamic settings
        struct LookupStruct *LT,
        double* restrict p0, // reference air pressure
        double* restrict temperature,  // temperature of air parcel
        double* restrict qt, // total water specific humidity
        double* restrict S_lookup,
        double* restrict S_liq,
        double* restrict S_ice
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
                
                // lookup table method 
                S_lookup[ijk] = microphysics_saturation_ratio(LT, temperature[ijk], p0[k], qt[ijk]);
                S_liq[ijk] = microphysics_saturation_ratio_liq(temperature[ijk], p0[k], qt[ijk]);
                S_ice[ijk] = microphysics_saturation_ratio_ice(temperature[ijk], p0[k], qt[ijk]);
            }
        }
    }
    return;
}

void liquid_saturation_adjustment(
        struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
        const double p0, 
        const double dt,
        const double s, 
        const double qt, 
        const double ql,
        const double nl,
        const double qi,
        // OUTPUT variable
        double* qv,
        double* T, 
        double* ql_tendency, 
        double* nl_tendency 
    ){
    // this section will calculate following variables:
    // * ql tendency comes from evap/cond saturation adjustment;
    // * nl tendency comes from evap/cond saturation adjustment;
    // * T: temperature changed after saturation adjustment;

    double pv_1 = pv_c(p0,qt,qt);
    double pd_1 = p0 - pv_1;
    double T_1 = temperature_no_ql(pd_1,pv_1,s,qt);
    double pv_star_1 = lookup(LT, T_1);
    double qv_star_1 = qv_star_c(p0,qt,pv_star_1);
    double ql_cond, nl_cond;

    /// If not saturated
    if(qt <= qv_star_1){
        // all cloud droplet evaporate
        *ql_tendency += -ql/dt;
        *nl_tendency += -nl/dt;
        *T = T_1;
        return;
    }
    else{
        double sigma_1 = qt - qv_star_1;
        double lam_1 = lam_fp(T_1);
        double L_1 = L_fp(T_1,lam_1);
        double s_1 = sd_c(pd_1,T_1) * (1.0 - qt) + sv_c(pv_1,T_1) * qt + sc_c(L_1,T_1)*sigma_1;
        double f_1 = s - s_1;
        double T_2 = T_1 + sigma_1 * L_1 /((1.0 - qt)*cpd + qv_star_1 * cpv);
        double delta_T  = fabs(T_2 - T_1);
        double qv_star_2;
        double sigma_2;
        double lam_2;
        do{
            double pv_star_2 = lookup(LT, T_2);
            qv_star_2 = qv_star_c(p0,qt,pv_star_2);
            double pv_2 = pv_c(p0,qt,qv_star_2);
            double pd_2 = p0 - pv_2;
            sigma_2 = qt - qv_star_2;
            lam_2 = lam_fp(T_2);
            double L_2 = L_fp(T_2,lam_2);
            double s_2 = sd_c(pd_2,T_2) * (1.0 - qt) + sv_c(pv_2,T_2) * qt + sc_c(L_2,T_2)*sigma_2;
            double f_2 = s - s_2;
            double T_n = T_2 - f_2*(T_2 - T_1)/(f_2 - f_1);
            T_1 = T_2;
            T_2 = T_n;
            f_1 = f_2;
            delta_T  = fabs(T_2 - T_1);
        } while(delta_T >= 1.0e-3 || sigma_2 < 0.0 );
        // *qv = qv_star_2;
        // after saturation adjustment, cloud condensation reach:

        ql_cond = (qt - qv_star_2);
        nl_cond = ql_cond/LIQUID_MIN_MASS;
        
        *T  = T_2;
        *qv = qt - ql - qi;
        // *qv = qt - ql - qi;
        // TODO: whether calculate the qv use the updated ql contend.

        *ql_tendency += (ql_cond - ql)/dt;
        *nl_tendency += (nl_cond - nl)/dt;
    }
    return;
}

void eos_sb_update(struct DimStruct *dims, 
    struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
    double* restrict p0, 
    double dt,
    double IN, // given ice nuclei
    double CCN, 
    double* restrict saturation_ratio,
    double* restrict s, 
    double* restrict w,
    double* restrict qt, 
    double* restrict T,
    double* restrict qv, 
    double* restrict ql, 
    double* restrict nl, 
    double* restrict qi, 
    double* restrict ni,
    double* restrict alpha,
    double* restrict ql_tendency, double* restrict nl_tendency,
    double* restrict qi_tendency, double* restrict ni_tendency){

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
                    double t_tmp = T[ijk];
                    
                    ql[ijk] = fmax(ql[ijk],0.0);
                    qi[ijk] = fmax(qi[ijk],0.0);
                    // *qv = qt - ql - qi;
                    // TODO: whether calculate the qv use the updated ql contend.
                    qv[ijk] = qt[ijk] - ql[ijk] - qi[ijk];

                    double dS = saturation_ratio[ijk +1] - saturation_ratio[ijk];
                    sb_ccn(CCN, saturation_ratio[ijk], dS, dzi, w[ijk],
                            &ql_tendency[ijk], &nl_tendency[ijk]);
                    // sb_cloud_activation_hdcp(p0[k], qv[ijk], 
                    //         ql[ijk], nl[ijk], w[ijk], dt, saturation_ratio[ijk],
                    //         &ql_tendency[ijk], &nl_tendency[ijk]);

                    // only update T[ijk] here
                    double qvl = qt[ijk] - qi[ijk];
                    eos_c(LT, lam_fp, L_fp, p0[k], s[ijk], qvl, 
                            &T[ijk], &qv_tmp, &ql_tmp, &qi_tmp);
                    alpha[ijk] = alpha_c(p0[k], T[ijk],qt[ijk],qv[ijk]);
                    // 
                    ql_tendency[ijk] += (ql_tmp - ql[ijk])/dt;
                    nl_tendency[ijk] += (ql_tmp/1.0e-12 - nl[ijk])/dt;
                    
                    // ------------ Ice particle Nucleation --------
                    double qi_tend_nuc, ni_tend_nuc;
                    sb_ice_nucleation_mayer(LT, IN,
                        T[ijk], qt[ijk], p0[k], 
                        qv[ijk], ni[ijk], dt, 
                        &qi_tend_nuc, &ni_tend_nuc);

                    qi_tendency[ijk] += qi_tend_nuc;
                    ni_tendency[ijk] += ni_tend_nuc;

                } // End k loop
            } // End j loop
        } // End i loop
    return;
}

void thetali_sb_update(struct DimStruct *dims, double (*lam_fp)(double), double (*L_fp)(double, double), 
        double* restrict p0, double* restrict T, double* restrict qt, 
        double* restrict ql, double* restrict qi, double* restrict thetali){

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
                double Lv=L_fp(T[ijk],lam_fp(T[ijk]));
                ql[ijk] = fmax(ql[ijk],0.0);
                qi[ijk] = fmax(qi[ijk],0.0);
                
                thetali[ijk] =  thetali_c(p0[k], T[ijk], qt[ijk], ql[ijk], qi[ijk], Lv);
            } // End k loop
        } // End j loop
    } // End i loop
    return;
}
