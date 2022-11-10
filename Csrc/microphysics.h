#pragma once
// #include "microphysics_arctic_1m.h"
// #include "microphysics_sb.h"
#include "parameters.h"
#include "lookup.h"
#include "parameters_micro_sb.h"
#include "thermodynamic_functions.h"
#include "advection_interpolation.h"
#include "entropies.h"

#define DENSITY_LIQUID  1000.0 // density of liquid water, kg/m^3
#define MICRO_EPS  1.0e-13
#define C_STOKES_VEL 1.19e8 //(m s)^-1, Rogers 1979, Ackerman 2009
#define SIGMA_G 1.5 //1.2 // geometric standard deviation of droplet psdf.  Ackerman 2009
                   
// Here, only functions that can be used commonly by any microphysical scheme
// convention: begin function name with "microphysics"
// Scheme-specific functions should be place in a scheme specific .h file, function name should indicate which scheme

double microphysics_mean_mass(double n, double q, double min_mass, double max_mass){
    // n = number concentration of species x, 1/kg
    // q = specific mass of species x kg/kg
    // min_mass, max_mass = limits of allowable masses, kg
    // return: mass = mean particle mass in kg
    double mass = fmin(fmax(q/fmax(n, MICRO_EPS),min_mass),max_mass); // MAX/MIN: when l_=0, x_=xmin
    return mass;
}

double microphysics_diameter_from_mass(double mass, double prefactor, double exponent){
    // find particle diameter from scaling rule of form
    // Dm = prefactor * mass ** exponent
    double diameter = prefactor * pow(mass, exponent);
    return diameter;
}

// Seifert & Beheng 2006: Equ 23.
double microphysics_g(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), double temperature){
    double lam = lam_fp(temperature);
    double L = L_fp(temperature,lam);
    double pv_sat = lookup(LT, temperature);
    double g_therm = 1.0/(Rv*temperature/DVAPOR/pv_sat + L/KT/temperature * (L/Rv/temperature - 1.0));
    return g_therm;
}

double microphysics_g_vi(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), double temperature){
    double lam = lam_fp(temperature);
    double L = L_fp(temperature,lam);
    double pv_sat = lookup(LT, temperature);
    double G_iv = 1.0/(Rv*temperature/DVAPOR/pv_sat + L_IV/KT/temperature * (L_IV/Rv/temperature - 1.0));
    // double g_therm = 1.0/(Rv*temperature/DVAPOR/pv_sat + L/KT/temperature * (L/Rv/temperature - 1.0));
    return G_iv;
}

double microphysics_saturation_ratio(struct LookupStruct *LT,  double temperature, double  p0, double qt){
    double pv_sat = lookup(LT, temperature);
    double qv_sat = qv_star_c(p0, qt, pv_sat);
    double saturation_ratio = qt/qv_sat - 1.0;
    return saturation_ratio;
}

double microphysics_homogenous_freezing_rate(double temperature){
    double T_celsius = temperature - 273.15;
    double liquid_density_cc = 1e-6; // 1 kg/m³ = 1e6 kg/cm³
    
    double exp_var_tmp;
    // following section is based on Cotton & Feild, Equ 12
    if (T_celsius < -65.0){
        exp_var_tmp = 25.63;
    }
    else if(T_celsius <= -30.0){
        exp_var_tmp = -243.4 - 14.75*T_celsius - 0.307*T_celsius*T_celsius - 0.00287*pow(T_celsius, 3.0) - 1.02e-5*pow(T_celsius, 4.0);
    }
    else{
        exp_var_tmp = -7.63 - 2.996*(T_celsius + 30.0);
    }

    // J_i: cm ^-3 /s; is defined in Cotton & Feild, Equ 12
    double J_i = exp(exp_var_tmp); 
    
    // return J_hom with the unit: kg^-1/s
    return liquid_density_cc * J_i;
}

// Seifert & Beheng 2006: Equ 44
double microphysics_heterogenous_freezing_rate(double temperature){
    // A_het and B_het are adopted from Pruppacher1997
    double A_het = 0.2; // kg^-1 s^-1
    double B_het = 0.65; // K^-1
    double t_3 = 273.15; // J
    double var_tmp = B_het*(t_3 - temperature) - 1.0;
    return A_het * exp(var_tmp);
}

double compute_wetbulb(struct LookupStruct *LT,const double p0, const double s, const double qt, const double T){
    double Twet = T;
    double T_1 = T;
    double pv_star_1  = lookup(LT, T_1);
    double qv_star_1 = qv_star_c(p0,qt,pv_star_1);
    ssize_t iter = 0;
    /// If not saturated
    if(qt >= qv_star_1){
        Twet = T_1;
    }
    else{
        qv_star_1 = pv_star_1/(eps_vi * (p0 - pv_star_1) + pv_star_1);
        double pd_1 = p0 - pv_star_1;
        double s_1 = sd_c(pd_1,T_1) * (1.0 - qv_star_1) + sv_c(pv_star_1,T_1) * qv_star_1;
        double f_1 = s - s_1;
        double T_2 = T_1 - 0.5;
        double delta_T  = fabs(T_2 - T_1);

        do{
            double pv_star_2 = lookup(LT, T_2);
            double qv_star_2 = pv_star_2/(eps_vi * (p0 - pv_star_2) + pv_star_2);
            double pd_2 = p0 - pv_star_2;
            double s_2 = sd_c(pd_2,T_2) * (1.0 - qv_star_2) + sv_c(pv_star_2,T_2) * qv_star_2;
            double f_2 = s - s_2;
            double T_n = T_2 - f_2*(T_2 - T_1)/(f_2 - f_1);
            T_1 = T_2;
            T_2 = T_n;
            f_1 = f_2;
            delta_T  = fabs(T_2 - T_1);
            iter += 1;
        } while(iter < 1);    //(delta_T >= 1.0e-3);
        Twet  = T_1;
    }
    return Twet;
}

void microphysics_wetbulb_temperature(struct DimStruct *dims, struct LookupStruct *LT, double* restrict p0, double* restrict s,
                                      double* restrict qt,  double* restrict T,  double* restrict Twet ){
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
                Twet[ijk] = compute_wetbulb(LT, p0[k], s[ijk], qt[ijk],  T[ijk]);

            } // End k loop
        } // End j loop
    } // End i loop
    return;
 }

void microphysics_stokes_sedimentation_velocity(const struct DimStruct *dims, double* restrict density, double ccn,
                                     double* restrict ql, double* restrict qt_velocity){

   const ssize_t istride = dims->nlg[1] * dims->nlg[2];
   const ssize_t jstride = dims->nlg[2];
   const ssize_t imin = 0;
   const ssize_t jmin = 0;
   const ssize_t kmin = 0;
   const ssize_t imax = dims->nlg[0];
   const ssize_t jmax = dims->nlg[1];
   const ssize_t kmax = dims->nlg[2];
   const double distribution_factor = exp(5.0 * log(SIGMA_G) * log(SIGMA_G));
   const double number_factor = C_STOKES_VEL * cbrt((0.75/pi/DENSITY_LIQUID/ccn) * (0.75/pi/DENSITY_LIQUID/ccn));

   for(ssize_t i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
       for(ssize_t j=jmin; j<jmax; j++){
           const ssize_t jshift = j * jstride;
           for(ssize_t k=kmin-1; k<kmax+1; k++){
               const ssize_t ijk = ishift + jshift + k;
               double ql_tmp = fmax(ql[ijk],0.0);

               qt_velocity[ijk] = -number_factor * distribution_factor *  cbrt(density[k]* density[k] *ql_tmp* ql_tmp);

           }
       }
   }

    for(ssize_t i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
       for(ssize_t j=jmin; j<jmax; j++){
           const ssize_t jshift = j * jstride;
           for(ssize_t k=kmin; k<kmax-1 ; k++){
               const ssize_t ijk = ishift + jshift + k;

               qt_velocity[ijk] = interp_2(qt_velocity[ijk], qt_velocity[ijk+1]) ;
           }
       }
   }

   return;

}
//See Ackerman et al 2009 (DYCOMS-RF02 IC paper) Eq. 7


double entropy_src_precipitation_c(const double pv_sat_T, const double p0, const double temperature, 
        const double qt, const double qv, const double L, const double precip_rate){
    double pd = pd_c(p0, qt, qv);
    double sd = sd_c(pd, temperature);
    double pv = pv_c(p0, qt, qv);
    double sv_star_t = sv_c(pv_sat_T, temperature);
    double sc = sc_c(L, temperature);
    double S_P = sd - sv_star_t - sc;

    return S_P * precip_rate;
};

double entropy_src_evaporation_c(const double pv_sat_T, const double pv_sat_Tw, const double p0, 
        const double temperature, double Tw, const double qt, const double qv, const double L_Tw, const double evap_rate){
    double pd = pd_c(p0, qt, qv);
    double pv = pv_c(p0, qt, qv);
    double sd = sd_c(pd, temperature);
    double sv_star_tw = sv_c(pv_sat_Tw, Tw);
    double sc = sc_c(L_Tw, Tw);
    double S_E = sv_star_tw + sc - sd;
    double S_D = -Rv*log(pv/pv_sat_T) + cpv*log(temperature/Tw);
    //
    return (S_E + S_D) * evap_rate;
};
