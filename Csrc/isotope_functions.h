#pragma once
#include "parameters.h"
#include "thermodynamic_functions.h"
#include "microphysics.h"
#include "microphysics_sb.h"
#include "lookup.h"
#include <math.h>

static inline double equilibrium_fractionation_factor_H2O18(double t){
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
    double alpha_eq = 1.0 / equilibrium_fractionation_factor_H2O18(temperature);
    double R_sur_evap = alpha_eq*alpha_k*R_std_O18/((1-RH)+alpha_k*RH);
    return R_sur_evap;
}