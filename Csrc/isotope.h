#pragma once
#include "parameters.h"
#include "isotope_functions.h"
#include <math.h>
#include "thermodynamics_sa.h"
#include "microphysics_sb.h"
#include "microphysics.h"
#include "entropies.h"
#include "thermodynamic_functions.h"

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

