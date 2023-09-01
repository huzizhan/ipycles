import numpy as np
from collections import OrderedDict
from scipy.special import gamma
from math import sqrt

def main():
    default_arctic() #Generate the default microphysical parameters for the Arctic Mixed-Phase scheme
    return

def default_arctic():

    #########################
    # Users should modify here
    #########################

    parameters = OrderedDict()

    # Microphysical constants

    parameters["SB_EPS"] = 1.0e-13 #small value
    parameters["DENSITY_SB"] = 1.225 # kg/m^3; a reference density used in Seifert & Beheng 2006, DALES
    parameters['MAX_ITER'] = 15 # max interation of micro source terms
    parameters["XSTAR"] = 2.6e-10 # the threshold value separates droplets from raindrop
    parameters["A_VENT_RAIN"] = 0.78
    parameters["B_VENT_RAIN"] = 0.308
    parameters["NSC_3"] = 0.892112 #cbrt(0.71) // Schmidt number to the 1/3 power
    parameters["KIN_VISC_AIR"] = 1.4086e-5 #m^2/s kinematic viscosity of air
    parameters["KT"] = 2.5e-2 # J/m/s/K
    parameters["DVAPOR"] =3.0e-5 # m^2/s
    parameters["DENSITY_LIQUID"] = 1000.0 # density of liquid water, kg/m^3
    parameters['LHF'] = 3.34e5  # constant latent heat of fusion
    
    #====<<< ventilation parameters >>> ========
    parameters["A_VR"] = 0.78 # constant ventilation coefficient for raindrops aᵥᵣ
    parameters["A_VI"] = (0.78+0.86)/2 # constant ventilation coefficient for ice aᵥᵢ
    parameters["B_VR"] = 0.308 # constant ventilation coefficient for raindrops bᵥᵣ
    parameters["B_VI"] = (0.28+0.308)/2 # constant ventilation coefficient for raindrops bᵥᵢ

    # Liquid parameters
    
    parameters["LIQUID_MIN_MASS"] = 4.20e-15 # kg
    parameters["LIQUID_MAX_MASS"] = 2.6e-10 #1.0e-11  // kg
    parameters["C_LIQUID_SED"] =702780.63036 #1.19e8 *(3.0/(4.0*pi*rho_liq))**(2.0/3.0)*np.exp(5.0*np.log(1.34)**2.0)

    # Rain parameters
    
    parameters["RAIN_MAX_MASS"] = 5.2e-7 #kg; DALES: 5.0e-6 kg
    parameters["RAIN_MIN_MASS"] = 2.6e-10 #kg
    parameters["DROPLET_MIN_MASS"] = 4.20e-15 # kg
    parameters["DROPLET_MAX_MASS"] = 2.6e-10 #1.0e-11  // kg
    parameters["RAIN_MAX_MASS"] = 5.2e-7 #kg; DALES: 5.0e-6 kg
    parameters["RAIN_MIN_MASS"] = 2.6e-10 #kg
    parameters["DROPLET_MIN_MASS"] =4.20e-15 # kg
    parameters["DROPLET_MAX_MASS"] = 2.6e-10 #1.0e-11  // kg
    parameters["KCC"] = 10.58e9 # Constant in cloud-cloud kernel, m^3 kg^{-2} s^{-1}: Using Value in DALES; also, 9.44e9 (SB01, SS08), 4.44e9 (SB06)
    parameters["KCR"] = 5.25 # Constant in cloud-rain kernel, m^3 kg^{-1} s^{-1}: Using Value in DALES and SB06;  KCR = kr = 5.78 (SB01, SS08)
    parameters["KRR"] = 7.12 # Constant in rain-rain kernel,  m^3 kg^{-1} s^{-1}: Using Value in DALES and SB06; KRR = kr = 5.78 (SB01, SS08); KRR = 4.33 (S08)
    parameters["KAPRR"] = 60.7 # Raindrop typical mass (4.471*10^{-6} kg to the -1/3 power), kg^{-1/3}; = 0.0 (SB01, SS08)
    parameters["KAPBR"] = 2.3e3 # m^{-1} - Only used in SB06 break-up
    parameters["D_EQ"] = 0.9e-3 # equilibrium raindrop diameter, m, used for SB-breakup
    parameters["D_EQ_MU"] = 1.1e-3 # equilibrium raindrop diameter, m, used for SB-mu, opt=4
    parameters["A_RAIN_SED"] = 9.65 # m s^{-1}
    parameters["B_RAIN_SED"] = 9.796 # 10.3    # m s^{-1}
    parameters["C_RAIN_SED"] = 600.0  # m^{-1}
    parameters["A_NU_SQ"] = sqrt(parameters["A_RAIN_SED"]/parameters["KIN_VISC_AIR"])
    
    # Single-Ice parameters

    parameters["T_ICE"] = 235.0 # set to be the threshold for vapor nucli, deposition and freezing
    parameters["ICE_MAX_MASS"] = 1.0e-7 #kg; SB06 snow
    parameters["ICE_MIN_MASS"] = 1.0e-12 #kg; SB06 cloud ice
    parameters["N_M92"] = 1e3 # m^{-3}
    parameters["A_M92"] = -0.639
    parameters["B_M92"] = 12.96
    parameters["X_ICE_NUC"] = 1e-12 # kg
    parameters["L_MELTING"] = 0.333e6 # J/kg latent heat of melting
    parameters["L_SUBLIMATION"] = 2.834e6 # J/kg latent heat of sublimation
    parameters["D_L0"] = 1.5e-5 # 15μm 
    parameters["D_L1"] = 4e-5 # 40μm 
    parameters["D_I0"] = 1.5e-4 # 150μm 
    parameters["SIGMA_ICE"] = 0.2 # m/s

    #############################
    # Users shouldn't modify below
    #############################

    # Some warning to put in the generated code
    message1 = 'Generated code! Absolutely DO NOT modify this file, ' \
               'microphysical parameters should be modified in generate_parameters_a1m.py \n'
    message2 = 'End generated code'

    # First write the pxi file
    f = './parameters_micro_sb.pxi'
    fh = open(f, 'w')
    fh.write('#' + message1)
    fh.write('\n')
    for param in parameters:
        fh.write(
            'cdef double ' + param + ' = ' + str(parameters[param]) + '\n')
    fh.write('#' + 'End Generated Code')
    fh.close()

    # Now write the C include file
    f = './Csrc/parameters_micro_sb.h'
    fh = open(f, 'w')
    fh.write('//' + message1)
    for param in parameters:
        fh.write('#define ' + param + ' ' + str(parameters[param]) + '\n')
    fh.write('//' + message2)
    fh.close()

    print('Generated ./parameters_micro_sb.pxi and '
          './Csrc/parameters_micro.h with the following values:')
    for param in parameters:
        print('\t' + param + ' = ' + str(parameters[param]))

    return

if __name__ == "__main__":
    main()
