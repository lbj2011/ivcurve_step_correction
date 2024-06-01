import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_voc(v, i):
    """
    Find voc of i-v curve

    v: voltage of i-v curve, numpy array
    i: current of i-v curve, numpy array

    """
    inx = np.where(i == np.nanmin(i[i>=0]))[0][0]
    voc = np.interp(0, i[inx:inx+2][::-1], v[inx:inx+2][::-1])
    return voc

def get_alpha(i, iref, Gref, G, Tref, T):
    """
    Alpha is the coefficient to make the Isc of the adjusted match with the measured one

    i: current of the i-v curve to correct (with step), numpy array
    iref: current of the reference i-v curve (without step), numpy array
    Gref: irradiance of the reference i-v curve (without step)
    G: irradiance of the i-v curve to correct (with step)
    Tref: module temperature of the reference i-v curve (without step)
    T: module temperature of the i-v curve to correct (with step)

    """
    isc_corr = np.nanmax(iref)*G/Gref
    isc = np.nanmax(i)
    alpha = (isc_corr/isc-1)/(T-Tref)
    return alpha

def get_beta(v, i, vref, iref, Gref, G, Tref, T, best_alpha):
    """
    Beta is the coefficient to make the Voc of the adjusted match with the measured one

    v: voltage of the i-v curve to correct (with step), numpy array
    vref: voltage of the reference i-v curve (without step), numpy array
    i: current of the i-v curve to correct (with step), numpy array
    iref: current of the reference i-v curve (without step), numpy array
    Gref: irradiance of the reference i-v curve (without step)
    G: irradiance of the i-v curve to correct (with step)
    Tref: module temperature of the reference i-v curve (without step)
    T: module temperature of the i-v curve to correct (with step)
    best_alpha: the alpha to make the Isc of the adjusted match with the measured one, obtained from get_alpha()

    """
    i_corr = iref*G/Gref/(1 + best_alpha*(T-Tref))
    voc = find_voc(v, i)
    best_err = 100
    for beta_voc_abs in np.arange(-0.1, 0.1, 0.005):
        v_corr = vref + beta_voc_abs*(T-Tref)
        voc_c = find_voc(v_corr, i_corr)
        if np.abs(voc_c-voc)<best_err:
            best_err = np.abs(voc_c-voc)
            best_beta = beta_voc_abs
    return best_beta

def get_k(v, i, vref, iref, Gref, G, Tref, T, best_alpha, best_beta):
    """
    k is the coefficient to make the high-voltage part (v>vmp, i<imp)of the adjusted match with the measured one

    v: voltage of the i-v curve to correct (with step), numpy array
    vref: voltage of the reference i-v curve (without step), numpy array
    i: current of the i-v curve to correct (with step), numpy array
    iref: current of the reference i-v curve (without step), numpy array
    Gref: irradiance of the reference i-v curve (without step)
    G: irradiance of the i-v curve to correct (with step)
    Tref: module temperature of the reference i-v curve (without step)
    T: module temperature of the i-v curve to correct (with step)
    best_alpha: the alpha to make the Isc of the adjusted match with the measured one, obtained from get_alpha()
    best_beta: the beta obtained from get_beta()


    """
    i_corr = iref*G/Gref/(1 + best_alpha*(T-Tref))
    best_err = 100
    for k in np.arange(0, 0.1, 0.01):
        v_corr = vref + best_beta*(T-Tref)- k*i_corr*(T-Tref)

        idpmp = np.nanargmax(v*i)
        imp = i[idpmp]
        i_corr_partial = np.interp(v[(i<imp) & (i>=0)], v_corr, i_corr)
        i_partial = np.interp(v[(i<imp) & (i>=0)], v, i)
        err = np.sum(np.abs(i_partial - i_corr_partial))
        
        if err<best_err:
            best_err = err
            best_k = k
    return best_k

def get_all(v, i, vref, iref, Gref, G, Tref, T):
    """
    get coefficients alpha, beta, k for correction

    v: voltage of the i-v curve to correct (with step), numpy array
    vref: voltage of the reference i-v curve (without step), numpy array
    i: current of the i-v curve to correct (with step), numpy array
    iref: current of the reference i-v curve (without step), numpy array
    Gref: irradiance of the reference i-v curve (without step)
    G: irradiance of the i-v curve to correct (with step)
    Tref: module temperature of the reference i-v curve (without step)
    T: module temperature of the i-v curve to correct (with step)
    
    """
    best_alpha = get_alpha(i, iref, Gref, G, Tref, T)
    best_beta = get_beta(v, i, vref, iref, Gref, G, Tref, T, best_alpha)
    best_k = get_k(v, i, vref, iref, Gref, G, Tref, T, best_alpha, best_beta)
    return best_alpha, best_beta, best_k