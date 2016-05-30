""" This file is part of The Cannon analysis project.
Copyright 2014 Melissa Ness.
# NOTE THE ISSUE MAY BE THE Fe_H SCALE THIS IS DIFFERENT IN THE UNFILTERED VERSIONA
# TO DO - CHECK THE UNFILTERED VERSION WITH THE 5 LABELS HERE GIVES THE SAME RESULTS 

# urls
- http://iopscience.iop.org/1538-3881/146/5/133/suppdata/aj485195t4_mrt.txt for calibration stars 
- http://data.sdss3.org/irSpectrumDetail?locid=4330&commiss=0&apogeeid=2M17411636-2903150&show_aspcap=True object explorer 
- http://data.sdss3.org/basicIRSpectra/searchStarA
- http://data.sdss3.org/sas/dr10/apogee/spectro/redux/r3/s3/a3/ for the data files 

# to-do
- need to add a test that the wavelength range is the same - and if it isn't interpolate to the same range 
- format PEP8-ish (four-space tabs, for example)
- take logg_cut as an input
- extend to perform quadratic fitting
"""

#from astropy.io import fits as pyfits 
import pyfits
import os
import scipy 
import glob 
import pickle
import pylab  as plt
from scipy import interpolate 
from scipy import ndimage 
from scipy import optimize as opt
import numpy as np
from datetime import datetime
import multiprocessing as _mp
from multiprocessing.pool import Pool as _Pool
import signal as _signal
from multiprocessing import TimeoutError
import functools as _fntools
LARGE = 200


# to do item : Fix this 
filteroff = 1 # if 1 filters are off, if 0 filters are on 
filteroff = 0 # if 1 filters are off, if 0 filters are on 
calflag = 1 # if 1 get the normalised calibrationd data and if 0 doing test on something else like redclump
calflag = 0 # if 1 get the normalised calibrationd data and if 0 doing test on something else like redclump
endval = 'binary3'
normed_training_data = 'normed_dr13_nofilt_large_010516_1.pickle'
normed_training_data = 'normed_dr13_binary.pickle'
normed_training_data = 'normed_dr13_'+endval+'.pickle'
coeffs_file = "coeffs_dr13_"+endval+".pickle"
tags_file = "tags_dr13_30eB_"+endval+".pickle"
fn = "training_dr13e2_large_cleaner_H_fix_dr13.list"
#fn = "test.list"
fn_filt = '../APOGEE_DR12/mkn_filters_wider1.txt' # this one now has the real filters in 
fn_filt_1 = 'filters_14.txt' # this one now has the real filters in 
nelem = 22
#nelem = 14
pmap = map

def getscale(in_array): 
  valin = np.percentile(in_array, (2.5, 50, 97.5))
  valscale = (valin[2]-valin[0])/4.
  valoff = (valin[1])
  return valscale, valoff

def rescale(in_array):
  valscale, valoff = getscale(in_array) 
  return (in_array - valoff) / valscale

def unscale(scaled_val, in_array):
  valscale, valoff = getscale(in_array) 
  return  scaled_val*valscale  + valoff 

def unscale_covs(scaled_val, in_array):
  valscale, valoff = getscale(in_array) 
  return  scaled_val*valscale  


# all inputs are below 
# read in all the values 
T_est,g_est,feh_est,alpha_est, T_A, g_A, feh_A,rc_est = np.loadtxt(fn, usecols = (1,2,3,4,1,2,3,4), unpack =1) 
Fe, C, N, O, Na, Mg, Al, Si, P, S, K, Ca, Ti, V, Cr, Mn,Co, Ni, Cu, mass = np.loadtxt(fn, usecols = (3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22), unpack =1)
offsets = np.array([np.mean(rescale(T_est)), np.mean(rescale(g_est)), np.mean(rescale(feh_est)), np.mean(rescale(C)), np.mean(rescale(N)), np.mean(rescale(O)), np.mean(rescale(Na)), np.mean(rescale(Mg)), np.mean(rescale(Al)), np.mean(rescale(Si)), np.mean(rescale(P)), np.mean(rescale(S)), np.mean(rescale(K)), np.mean(rescale(Ca)), np.mean(rescale(Ti)), np.mean(rescale(V)), np.mean(rescale(Cr)), np.mean(rescale(Mn)), np.mean(rescale(Co)), np.mean(rescale(Ni)), np.mean(rescale(Cu)), np.mean(rescale(mass))]) 
offsets = offsets[0:nelem]
inputs = [T_est, g_est, feh_est, C, N, O, Na, Mg, Al, Si,P, S, K, Ca, Ti, V, Cr,Mn, Co, Ni, Cu, mass][0:nelem]
labels = ["teff", "logg", "feh", "C", "N", "O", "Na", "Mg", "Al", "Si", "P", "S", "K", "Ca", "Ti", "V", "Cr", "Mn", "Co", "Ni", "Cu", "mass"][0:nelem]

scales = np.array([np.mean(getscale(T_est)), np.mean(getscale(g_est)), np.mean(getscale(feh_est)), np.mean(getscale(C)), np.mean(getscale(N)), np.mean(getscale(O)), np.mean(getscale(Na)), np.mean(getscale(Mg)), np.mean(getscale(Al)), np.mean(getscale(Si)), np.mean(getscale(P)), np.mean(getscale(S)), np.mean(getscale(K)), np.mean(getscale(Ca)), np.mean(getscale(Ti)), np.mean(getscale(V)), np.mean(getscale(Cr)), np.mean(getscale(Mn)), np.mean(getscale(Co)), np.mean(getscale(Ni)), np.mean(getscale(Cu)), np.mean(getscale(mass))]) [0:nelem] 
T_filt, g_filt, feh_filt, alpha_filt = np.loadtxt(fn_filt_1, usecols = (0,1,2,3), unpack =1)
C_filt, N_filt, Al_filt, Mg_filt, Na_filt, O_filt, S_filt, V_filt, Mn_filt, Ni_filt = np.loadtxt(fn_filt, usecols = (3,4,10,9,8,5,13,20,22,25), unpack =1)
Si_filt,P_filt, K_filt, Ca_filt, Cr_filt, Co_filt, Rb_filt, C_filt, N_filt, O_filt, Na_filt = np.loadtxt(fn_filt, usecols = (11,12,16,17,21,24,29,3,4,5,8), unpack =1)
Ti_filt = np.loadtxt(fn_filt, usecols = (19,), unpack =1) 
Fe_filt = np.loadtxt(fn_filt, usecols = (23,), unpack =1) 
Cu_filt = np.loadtxt(fn_filt, usecols = (26,), unpack =1) 
#BINARY
offsets = hstack((offsets,offsets))
  
T_est = rescale(T_est)
g_est = rescale(g_est)
feh_est = rescale(feh_est)
mass = rescale(mass)
alpha_est = rescale(alpha_est)
C, N, O, Na, Mg, Al = rescale(C), rescale(N), rescale(O), rescale(Na), rescale(Mg), rescale(Al) 
Si, S, K, Ca, Ti, V, Mn, Fe, Ni = rescale(Si), rescale(S), rescale(K), rescale(Ca), rescale(Ti), rescale(V), rescale(Mn), rescale(Fe), rescale(Ni) 
P, Cr, Co  = rescale(P), rescale(Cr), rescale(Co)
Cu = rescale(Cu) 

# this goes into the function 
label_array = [T_est, g_est, feh_est, C, N, O, Na, Mg, Al, Si, P,S, K, Ca, Ti, V,Cr, Mn, Co, Ni, Cu, mass][0:nelem]
filter_array = [T_filt, g_filt, feh_filt, C_filt, N_filt, O_filt, Na_filt, Mg_filt, Al_filt, Si_filt, P_filt, S_filt, K_filt, Ca_filt, Ti_filt, V_filt, Cr_filt, Mn_filt, Co_filt, Ni_filt, Cu_filt, T_filt][0:nelem]
if filteroff == 1: 
  filter_array = [T_filt,g_filt,feh_filt,T_filt,T_filt,T_filt,T_filt,T_filt,T_filt,T_filt,T_filt,T_filt,T_filt,T_filt,T_filt,T_filt,T_filt,T_filt,T_filt,T_filt,T_filt,T_filt][0:nelem]


def weighted_median(values, weights, quantile):
    """weighted_median

    keywords
    --------

    values: ndarray
        input values

    weights: ndarray
        weights to apply to each value in values

    quantile: float
        quantile selection

    returns
    -------
    val: float
        median value
    """
    sindx = np.argsort(values)
    cvalues = 1. * np.cumsum(weights[sindx])
    cvalues = cvalues / cvalues[-1]
    foo = sindx[cvalues > quantile]
    if len(foo) == 0:
        return values[0]
    indx = foo[0]
    return values[indx]

def continuum_normalize_tsch(dataall,maskall, pixlist, delta_lambda=150):
    pixlist = list(pixlist) 
    Nlambda, Nstar, foo = dataall.shape
    continuum = np.zeros((Nlambda, Nstar))
    dataall_flat = np.ones((Nlambda, Nstar, 3))
    for jj in range(Nstar):
        bad_a = np.logical_or(np.isnan(dataall[:, jj, 1]) ,np.isinf(dataall[:,jj, 1]))
        bad_b = np.logical_or(dataall[:, jj, 2] <= 0. , np.isnan(dataall[:, jj, 2]))
        bad = np.logical_or( np.logical_or(bad_a, bad_b) , np.isinf(dataall[:, jj, 2]))
        dataall[bad, jj, 1] = 0.
        dataall[bad, jj, 2] = np.Inf 
        continuum = np.zeros((Nlambda, Nstar))
        var_array = np.Inf + np.zeros(len(dataall)) 
        var_array[pixlist] = 0.000
        ivar = 1. / ((dataall[:, jj, 2] ** 2) + var_array) 
        bad = np.isnan(ivar)
        ivar[bad] =  0
        bad = np.isinf(ivar)
        ivar[bad] =  0
        take1 = np.logical_and(dataall[:,jj,0] > 15150, dataall[:,jj,0] < 15800)
        take2 = np.logical_and(dataall[:,jj,0] > 15890, dataall[:,jj,0] < 16430)
        take3 = np.logical_and(dataall[:,jj,0] > 16490, dataall[:,jj,0] < 16950)
        fit1 = np.polynomial.chebyshev.Chebyshev.fit(x=dataall[take1,jj,0], y=dataall[take1,jj,1], w=ivar[take1],deg=2)# 2 or 3 is good for all, 2 only a few points better in temp 
        fit2 = np.polynomial.chebyshev.Chebyshev.fit(x=dataall[take2,jj,0], y=dataall[take2,jj,1], w=ivar[take2],deg=2)
        fit3 = np.polynomial.chebyshev.Chebyshev.fit(x=dataall[take3,jj,0], y=dataall[take3,jj,1], w=ivar[take3],deg=2)
        continuum[take1,jj] = fit1(dataall[take1,jj,0])
        continuum[take2,jj] = fit2(dataall[take2,jj,0])
        continuum[take3,jj] = fit3(dataall[take3,jj,0])
        dataall_flat[:, jj, 0] = dataall[:,jj,0]
        dataall_flat[take1, jj, 1] = dataall[take1,jj,1]/fit1(dataall[take1,0,0])
        dataall_flat[take2, jj, 1] = dataall[take2,jj,1]/fit2(dataall[take2,0,0]) 
        dataall_flat[take3, jj, 1] = dataall[take3,jj,1]/fit3(dataall[take3,0,0]) 
        dataall_flat[take1, jj, 2] = dataall[take1,jj,2]/fit1(dataall[take1,0,0]) 
        dataall_flat[take2, jj, 2] = dataall[take2,jj,2]/fit2(dataall[take2,0,0]) 
        dataall_flat[take3, jj, 2] = dataall[take3,jj,2]/fit3(dataall[take3,0,0]) 
    for jj in range(Nstar):
        print "continuum_normalize_tcsh working on star", jj
        bad_a = np.logical_not(np.isfinite(dataall_flat[:, jj, 1]))
        bad_a = np.logical_or(bad_a, dataall_flat[:, jj, 2] <= 0.)
        bad_a = np.logical_or(bad_a, np.logical_not(np.isfinite(dataall_flat[:, jj, 2])))
        bad_a = np.logical_or(bad_a, dataall[:, jj, 2] > 1.) # magic 1.
        # grow the mask
        bad = np.logical_or(bad_a, np.insert(bad_a,0,False,0)[0:-1])
        bad = np.logical_or(bad, np.insert(bad_a,len(bad_a),False)[1:])
        LARGE  = 2000. # magic LARGE sigma value
        dataall_flat[bad,jj, 1] = 1.
        dataall_flat[bad,jj, 2] = LARGE
    return dataall_flat, continuum 

def continuum_normalize(dataall, delta_lambda=50):
    """
    ## inputs:
    dataall:       (Nlambda, Nstar, 3) wavelengths, flux densities, errors
    delta_lambda:  half-width of meadian region in angstroms

    ## output:
    continuum:     (Nlambda, Nstar) continuum level

    ## comments:
    * does a lot of stuff *other* than continuum normalization

    ## bugs:
    * for loops!
    """
    Nlambda, Nstar, foo = dataall.shape
    continuum = np.zeros((Nlambda, Nstar))
    # sanitize inputs
    for jj in range(Nstar):
    #    #BROKEN
        bad_a = np.logical_or(np.isnan(dataall[:, jj, 1]) ,np.isinf(dataall[:,jj, 1]))
        bad_b = np.logical_or(dataall[:, jj, 2] <= 0. , np.isnan(dataall[:, jj, 2]))
        bad = np.logical_or( np.logical_or(bad_a, bad_b) , np.isinf(dataall[:, jj, 2]))
        dataall[bad, jj, 1] = 0.
        dataall[bad, jj, 2] = np.Inf #LARGE#np.Inf #100. #np.Inf
        base = scipy.ndimage.gaussian_filter(dataall[:,jj,1], 100)
        continuum[:,jj] = base
    for jj in range(Nstar):
        bad = np.where(continuum[:,jj] <= 0)
        continuum[bad,jj] = 1.
        dataall[:, jj, 1] /= continuum[:,jj]
        dataall[:, jj, 2] /= continuum[:,jj]
        dataall[bad,jj, 1] = 1.
        dataall[bad,jj, 2] = LARGE
        bad = np.where(dataall[:, jj, 2] > LARGE)
        dataall[bad,jj, 1] = 1.
        dataall[bad,jj, 2] = LARGE
    return dataall


def get_bad_pixel_mask(testfile,nlam): 
  name = testfile.split('.txt')[0]
  adir = open(testfile, 'r')
  al2 = adir.readlines()
  bl2 = []
  bl3 = []
  dirname = '/home/ness/new_laptop/Apogee_DR12/data.sdss3.org/sas/dr12/apogee/spectro/redux/r5/stars/l25_6d/v603/'
  for each in al2:
    bl2.append(each.strip()) 
    bl3.append((each.split('/'))[-2] +'/'+ ("apStar-s3-")+each.split('aspcapStar-v304-')[-1].strip())  
  if glob.glob(dirname):
    dirin = [dirname+each for each in bl3] 
    mask  = np.zeros((nlam, len(bl2),1))
    for jj,each in enumerate(dirin):
      a=pyfits.open(each) 
      mask[:,jj,0] = (np.atleast_2d(a[3].data))[0]
  else: 
    mask  = np.zeros((nlam, len(bl2),1))
  return mask 

def get_normalized_test_data_tsch(testfile, pixlist):
  name = testfile.split('.txt')[0]
  a = open(testfile, 'r')
  al2 = a.readlines()
  bl2 = []
  for each in al2:
    bl2.append(each.strip())
  ids = []
  for each in bl2:
    ids.append(each.split('-2M')[-1].split('.fits')[0])

  if glob.glob(name+'_alpha.pickle'):
    file_in2 = open(name+'_alpha.pickle', 'r')
    testdata = pickle.load(file_in2)
    file_in2.close()
    a = open(testfile, 'r')
    al2 = a.readlines()
    bl2 = []
    for each in al2:
      bl2.append(each.strip())
    SNR = np.zeros((len(bl2)))
    return testdata, ids

  SNRall = np.zeros(len(bl2))
  for jj,each in enumerate(bl2):
    a = pyfits.open(each) 
    if np.shape(a[1].data) != (8575,):
      ydata = a[1].data[0] 
      ysigma = a[2].data[0]
      len_data = a[2].data[0]
      #mask = a[3].data[0] # was 3 before for SNRVIS1
      #ydata = a[1].data[3] # SNR test - NOTE THIS IS FOR TEST TO READ IN A SINGLE VISIT - TESTING ONLY - OTHERWISE SHOULD BE 0 TO READ IN THE MEDIAN SPECTRA 
      #ysigma = a[2].data[3]
      #len_data = a[2].data[3]
      if jj == 0:
        nlam = len(a[1].data[0])
        testdata = np.zeros((nlam, len(bl2), 3))
    if np.shape(a[1].data) == (8575,):
      ydata = a[1].data
      ysigma = a[2].data
      len_data = a[2].data
      if jj == 0:
        nlam = len(a[1].data)
        testdata = np.zeros((nlam, len(bl2), 3))
    start_wl =  a[1].header['CRVAL1']
    diff_wl = a[1].header['CDELT1']
    SNR = a[0].header['SNR']
    #SNR = a[0].header['SNRVIS4']
    SNRall[jj] = SNR

    val = diff_wl*(nlam) + start_wl 
    wl_full_log = np.arange(start_wl,val, diff_wl) 
    wl_full = [10**aval for aval in wl_full_log]
    xdata = wl_full
    testdata[:, jj, 0] = xdata
    testdata[:, jj, 1] = ydata
    testdata[:, jj, 2] = ysigma
    #maskdata[:, jj] = mask
  #mask = get_bad_pixel_mask(testfile,nlam) 
  mask = np.zeros((nlam, len(bl),1))
  #for jj,each in enumerate(bl2):
  #  bad = mask[:,jj] != 0 
  #  testdata[bad, jj, 2] = 200.
# do no normalisataion
#  dataall, contall = continuum_normalize_tsch(testdata,mask, pixlist, delta_lambda=50)
  file_in = open(name+'_alpha.pickle', 'w')  
  file_in2 = open(name+'_SNR.pickle', 'w')
  pickle.dump(testdata,  file_in)
  pickle.dump(SNRall,  file_in2)
  file_in.close()
  file_in2.close()
  return testdata , ids # not yet implemented but at some point should probably save ids into the normed pickle file 


def get_normalized_test_data(testfile,noise=0): 
  """
    inputs
    ------
    testfile: str
        the file in with the list of fits files want to test - if normed, move on,
        if not normed, norm it
    if not noisify carry on as normal, otherwise do the noise tests

    returns
    -------
    testdata:
  """
  noise=0
  name = testfile.split('.txt')[0]
  a = open(testfile, 'r')
  al2 = a.readlines()
  bl2 = []
  for each in al2:
    bl2.append(each.strip())
  ids = []
  for each in bl2:
    ids.append(each.split('-2M')[-1].split('.fits')[0])
  
  if noise == 0: 
    if glob.glob(name+'_alpha.pickle'):
      file_in2 = open(name+'_alpha.pickle', 'r') 
      testdata = pickle.load(file_in2)
      file_in2.close()
      a = open(testfile, 'r')
      al2 = a.readlines()
      bl2 = []
      for each in al2:
        bl2.append(each.strip())
      SNR = np.zeros((len(bl2))) 
      for jj,each in enumerate(bl2):
        a = pyfits.open(each) 
        #SNR[jj]  = a[0].header['SNRVIS4']
        #SNR[jj]  = a[0].header['SNRVIS4']
        SNR[jj]  = a[0].header['SNR']
        file_in2 = open(name+'_alpha_SNR.pickle', 'w')  
        pickle.dump(SNR,  file_in2)
        file_in2.close()
      return testdata, ids 
  if noise == 1: 
    if not glob.glob(name+'._SNR.pickle'):
      a = open(testfile, 'r')
      al2 = a.readlines()
      bl2 = []
      for each in al2:
       # bl2.append(testdir+each.strip())
        bl2.append(each.strip())
      SNR = np.zeros((len(bl2))) 
      for jj,each in enumerate(bl2):
        a = pyfits.open(each) 
        SNR[jj]  = a[0].header['SNR']
        #SNR[jj]  = a[0].header['SNRVIS4']
        file_in2 = open(name+'_SNR.pickle', 'w')  
        pickle.dump(SNR,  file_in2)
        file_in2.close()
    if glob.glob(name+'.pickle'):
      if glob.glob(name+'_SNR.pickle'): 
        file_in2 = open(name+'.pickle', 'r') 
        testdata = pickle.load(file_in2)
        file_in2.close()
        file_in3 = open(name+'_SNR.pickle', 'r') 
        SNR = pickle.load(file_in3)
        file_in3.close()
        ydata = testdata[:,:,1]
        ysigma = testdata[:,:,2]
        testdata[:,:,1], testdata[:,:,2] =  add_noise(ydata, ysigma, SNR)
        return testdata, ids

  a = open(testfile, 'r')
  al2 = a.readlines()
  bl2 = []
  for each in al2:
    bl2.append(each.strip())
  ids = []
  for each in bl2:
    ids.append(each.split('-2M')[-1].split('.fits')[0])

  SNRall = np.zeros(len(bl2))
  for jj,each in enumerate(bl2):
    a = pyfits.open(each) 
    if np.shape(a[1].data) != (8575,):
      ydata = a[1].data[0] 
      ysigma = a[2].data[0]
      len_data = a[2].data[0]
      if jj == 0:
        nlam = len(a[1].data[0])
        testdata = np.zeros((nlam, len(bl2), 3))
    if np.shape(a[1].data) == (8575,):
      ydata = a[1].data
      ysigma = a[2].data
      len_data = a[2].data
      if jj == 0:
        nlam = len(a[1].data)
        testdata = np.zeros((nlam, len(bl2), 3))
    start_wl =  a[1].header['CRVAL1']
    diff_wl = a[1].header['CDELT1']
    SNR = a[0].header['SNR']
    #SNR = a[0].header['SNRVIS4']
    SNRall[jj] = SNR

    val = diff_wl*(nlam) + start_wl 
    wl_full_log = np.arange(start_wl,val, diff_wl) 
    wl_full = [10**aval for aval in wl_full_log]
    xdata = wl_full
    testdata[:, jj, 0] = xdata
    testdata[:, jj, 1] = ydata
    testdata[:, jj, 2] = ysigma
  testdata = continuum_normalize(testdata,delta_lambda=50) # testdata
  file_in = open(name+'_alpha.pickle', 'w')  
  file_in2 = open(name+'_SNR.pickle', 'w')
  pickle.dump(testdata,  file_in)
  pickle.dump(SNRall,  file_in2)
  file_in.close()
  file_in2.close()
  return testdata , ids # not yet implemented but at some point should probably save ids into the normed pickle file 

def get_normalized_training_data(pixlist,numtake, label_array, labels, filter_array):
  if glob.glob(normed_training_data): 
        file_in2 = open(normed_training_data, 'r') 
        dataall, filterall, metaall, labels, Ametaall, cluster_name, ids = pickle.load(file_in2)
        file_in2.close()
        return dataall, filterall, metaall, labels, Ametaall, cluster_name, ids
  
  
  a = open(fn, 'r')  
  al = a.readlines() 
  al = np.array(al)
  bl = []
  cluster_name = [] 
  ids = []
  for each in al:
    bl.append(each.split()[0]) 
    cluster_name.append(each.split()[1]) 
    ids.append(each.split()[0].split('-2M')[-1].split('.fits')[0])

  for jj,each in enumerate(bl):
    each = each.strip('\n')
    a = pyfits.open(each) 
    b = pyfits.getheader(each) 
    start_wl =  a[1].header['CRVAL1']
    diff_wl = a[1].header['CDELT1']
    if jj == 0:
      nmeta = len(labels)
      nlam = len(a[1].data)
      #nlam = len(a[1].data[0])
    val = diff_wl*(nlam) + start_wl 
    wl_full_log = np.arange(start_wl,val, diff_wl) 
    ydata = (np.atleast_2d(a[1].data))[0] 
    ydata_err = (np.atleast_2d(a[2].data))[0] 
    ydata_flag = (np.atleast_2d(a[3].data))[0] 
    assert len(ydata) == nlam
    wl_full = [10**aval for aval in wl_full_log]
    xdata= np.array(wl_full)
    ydata = np.array(ydata)
    ydata_err = np.array(ydata_err)
    starname2 = each.split('.fits')[0]+'.txt'
    sigma = (np.atleast_2d(a[2].data))[0]# /y1
    if jj == 0:
      npix = len(xdata) 
      dataall = np.zeros((npix, len(bl), 3))
      filterall = np.ones((npix, nmeta))
      metaall = np.ones((len(bl), nmeta))
      Ametaall = np.ones((len(bl), nmeta))
    if jj > 0:
      assert xdata[0] == dataall[0, 0, 0]

    dataall[:, jj, 0] = xdata
    dataall[:, jj, 1] = ydata
    dataall[:, jj, 2] = sigma

# changed this so check this 
  for k in range(0,len(bl)): 
      # must be synchronised with labels 
      for i in range(0, len(labels)):
        metaall[k,i] = label_array[i][k]
  for i in range(0, len(labels)):
    filterall[:,i] = filter_array[i]
  
  pixlist = list(pixlist) 
  mask = np.zeros((nlam, len(bl),1))
  # use no normalisation
  #dataall, contall = continuum_normalize_tsch(dataall,mask, pixlist, delta_lambda=50)
  # swap just to get gaussian spread 
  dataall = continuum_normalize(dataall,delta_lambda=50)
  file_in = open(normed_training_data, 'w')  
  pickle.dump((dataall, filterall, metaall, labels, Ametaall, cluster_name, ids),  file_in)
  file_in.close()
  return dataall, filterall, metaall, labels , Ametaall, cluster_name, ids


def do_one_regression_at_fixed_scatter(args, **kwargs):
    """
    Parameters
    ----------
    data: ndarray, [nobjs, 3]
        wavelengths, fluxes, invvars

    meta: ndarray, [nobjs, nmeta]
        Teff, Feh, etc, etc

    scatter:


    Returns
    -------
    coeff: ndarray
        coefficients of the fit

    MTCinvM: ndarray
        inverse covariance matrix for fit coefficients

    chi: float
        chi-squared at best fit

    logdet_Cinv: float
        inverse of the log determinant of the cov matrice
        :math:`\sum(\log(Cinv))`
    use the same terminology as in the paper 
    """
    data, filter1, features = args
    scatter = kwargs.get('scatter', 0)
    nmeta = filter1.shape[0]
    nobjs, npars = features.shape
    print nobjs, npars, nmeta
    assert npars == nmeta * (nmeta + 3) / 2 + 1
    #print filter1.shape,nmeta, features.shape
    filter_features = [np.hstack((1, filter1)) ] 
    filter1 = np.array([filter1])
    # the way the filters is done is repeating code that exists in train()
    filter_newfeatures = np.array([np.outer(m, m)[np.triu_indices(nmeta)] for m in filter1])
    filter_features = np.hstack((filter_features, filter_newfeatures))[0]
    filter_features = np.array(filter_features) 
    filter_features_bool = filter_features.astype(bool) 
    assert np.shape(filter_features_bool)[0] == npars 
    D = np.sum(filter1)
    assert np.sum(filter_features_bool) == D * (D + 3) / 2 + 1
    ### make the filter above
    # least square fit
    Cinv = 1. / (data[:, 2] ** 2 + scatter ** 2)  # invvar slice of data
    M = features[:,filter_features_bool]
    MTCinvM = np.dot(M.T, Cinv[:, None] * M) # craziness b/c Cinv isnt a matrix
    x = data[:, 1] # intensity slice of data
    MTCinvx = np.dot(M.T, Cinv * x)
    coeff_full = np.zeros(len(filter_features))
    coeff_ind = np.arange(len(filter_features))[filter_features_bool] 
    try:
        coeff = np.linalg.solve(MTCinvM, MTCinvx)
    except np.linalg.linalg.LinAlgError:
        print MTCinvM, MTCinvx, data[:,0], data[:,1], data[:,2]
        print features
    if not np.all(np.isfinite(coeff)):
        print "coefficients not finite"
        print coeff, median(data[:,2]), data.shape , scatter
        assert False
    for a,b in zip(coeff_ind ,coeff):
      coeff_full[a] = b
    chi = np.sqrt(Cinv) * (x - np.dot(M, coeff)) 
    logdet_Cinv = np.sum(np.log(Cinv)) 
    #print nobjs, nmeta, D, filter_features_bool.shape, coeff_ind, filter1, filter_features,  np.linalg.eigvalsh(MTCinvM) 
    #assert False 
    #print type(coeff_full)
    #print np.shape(coeff_full)
    return (coeff_full, MTCinvM, chi, logdet_Cinv )

def do_one_regression(data, filter1, metadata):
    """
    This is currently deprecated 
    does a regression at a single wavelength to fit calling the fixed scatter routine
    # inputs:
    """
    #print "do_one_regression(): working on wavelength", data[0, 0]
    ln_s_values = np.arange(np.log(0.0001), 0., 0.5)
    chis_eval = np.zeros_like(ln_s_values)
    for ii, ln_s in enumerate(ln_s_values):
        foo, bar, chi, logdet_Cinv = do_one_regression_at_fixed_scatter(data, filter1,metadata, scatter = np.exp(ln_s))
        chis_eval[ii] = np.sum(chi * chi) - logdet_Cinv
    if np.any(np.isnan(chis_eval)):
        s_best = np.exp(ln_s_values[-1])
        return do_one_regression_at_fixed_scatter(data,filter1, metadata, scatter = s_best) + (s_best, )
    lowest = np.argmin(chis_eval)
    #if lowest == 0 or lowest == len(ln_s_values) + 1:
    if lowest == 0 or lowest == len(ln_s_values)-1:
        s_best = np.exp(ln_s_values[lowest])
        return do_one_regression_at_fixed_scatter(data, filter1, metadata, scatter = s_best) + (s_best, )
    #print data
    #print metadata
    #print "LOWEST" , lowest
    ln_s_values_short = ln_s_values[np.array([lowest-1, lowest, lowest+1])]
    chis_eval_short = chis_eval[np.array([lowest-1, lowest, lowest+1])]
    z = np.polyfit(ln_s_values_short, chis_eval_short, 2)
    f = np.poly1d(z)
    fit_pder = np.polyder(z)
    fit_pder2 = pylab.polyder(fit_pder)
    s_best = np.exp(np.roots(fit_pder)[0])
    return do_one_regression_at_fixed_scatter(data,filter1, metadata, scatter = s_best) + (s_best, )

def do_regressions(dataall, filterall, features):
    """
    """
    nlam, nobj, ndata = dataall.shape
    nobj, npred = features.shape
    featuresall = np.zeros((nlam,nobj,npred))
    featuresall[:, :, :] = features[None, :, :]
    return pmap(do_one_regression_at_fixed_scatter, zip(dataall, filterall, featuresall))

def train(dataall_a, filterall_a, metaall_a, order, fn, Ametaall, cluster_name, logg_cut=100., teff_cut=0., leave_out=None):
    """
     # BUGS REMOVE Ametaall, teff_cut , logg_cut 
    - `leave out` must be in the correct form to be an input to `np.delete`
    """
    #dataall = np.zeros((8575, 25000, 3))
    t1,t2 = loadtxt('testinds.txt', usecols = (0,1), unpack =1) 
    t1 = t1.astype(int)
    t2 = t2.astype(int)
    j,k = 6,3
    star_flux1 = dataall_a[:,t1,1][j:]
    star_err1 = dataall_a[:,t1,2][j:]
    test,testerr = [],[]
    for each in t2:
      test.append(list([0]*k)+list(dataall_a[:,each,1]))
      testerr.append(list([100]*k)+list(dataall_a[:,each,2]))
    star_flux2 = array(test).T
    star_err2 = array(testerr).T
    # truncate array
    val1,val2 = 10,8500
    star_flux_both = (star_flux1[val1:val2] + star_flux2[val1:val2])/2
    star_err_both = (star_err1[val1:val2]**2 + star_err2[val1:val2]**2)**0.5
    dataall_a[val1:val2:,0:len(t1),1] = star_flux_both
    dataall_a[val1:val2:,0:len(t1),2] = star_err_both
    # this is if it is bigger
    dataall = dataall_a[val1:val2:,0:len(t1),:]
    dataall[val1:val2:,0:len(t1),1] = star_flux_both
    dataall = dataall_a[val1:val2:,0:len(t1),:]
    filterall = hstack((filterall_a, filterall_a))[val1:val2,:]
    metaall = hstack(( metaall_a[t1] ,metaall_a[t2] ) ) 
   
    nstars, nlabel= metaall.shape
    nwl, nnstars, three= dataall.shape
    assert nstars == nnstars
    assert three == 3
    assert order ==  2 # if we run at any other order do one regression will fail 

    features = np.ones((nstars, 1))
    if order >= 1:
        features = np.hstack((features, metaall - offsets)) 
    if order >= 2:
        newfeatures = np.array([np.outer(m, m)[np.triu_indices(nlabel)] for m in (metaall - offsets)])
        features = np.hstack((features, newfeatures))

    blob = do_regressions(dataall, filterall, features)
    coeffs = np.array([b[0] for b in blob])
    covs = np.array([np.linalg.inv(b[1]) for b in blob])
    chis = np.array([b[2] for b in blob])
    chisqs = np.array([np.dot(b[2],b[2]) - b[3] for b in blob]) # holy crap be careful
    if len(blob[0]) > 4: 
      scatters = np.array([b[4] for b in blob])
    else:
      scatters = 0.0*chisqs

    fd = open(fn, "w")
    # DUMMY covs as too big - when filters are on the covs is the wrong shape - only the number of training spectra long 
    #errval = [] 
    print np.shape(covs) 
    #scovs = np.shape(covs)[1]
    #slen = len(features) 
    #for i in range(0,slen-1):
    #  errval.append(covs[:,i,i]**0.5) 
    #  
    #pickle.dump((dataall, metaall, labels, offsets, coeffs, errval, scatters,chis,chisqs), fd)
    covs = 0 
    pickle.dump((dataall, metaall, labels, offsets, coeffs, covs, scatters,chis,chisqs), fd)
    fd.close()

def get_goodness_fit(fn_pickle, filein, Params_all_scaled, MCM_rotate_all):
    fd = open(fn_pickle,'r')
    dataall, metaall, labels, schmoffsets, coeffs, covs, scatters, chis, chisq = pickle.load(fd) 
    #assert np.all(schmoffsets == offsets) 
    assert (round(offsets[-1],4)) == round(schmoffsets[-1],4) 
    fd.close() 
    print str(filein)
   # if filein == 'cal_file':
    if calflag == 1:
      file_with_star_data = 'cal_file_DR13.pickle'
    else: 
      file_with_star_data = str(filein)+"_alpha.pickle"
    file_normed = normed_training_data.split('.pickle')[0]
    if filein != file_normed: 
      f_flux = open(file_with_star_data, 'r') 
      flux = pickle.load(f_flux) 
    if filein == file_normed: 
      f_flux = open(normed_training_data, 'r') 
      flux, filterall, metaall, labels, Ametaall, cluster_name, ids = pickle.load(f_flux)
    f_flux.close() 
    labels = Params_all_scaled 
    nlabels = np.shape(labels)[1]
    nstars = np.shape(labels)[0]
    features_data = np.ones((nstars, 1))

    features_data = np.hstack((features_data, labels - offsets)) 
    newfeatures_data = np.array([np.outer(m, m)[np.triu_indices(nlabels)] for m in (labels - offsets)])
    features_data = np.hstack((features_data, newfeatures_data))
    chi2_all = np.zeros(nstars) 
    chi_all = np.zeros((len(coeffs),nstars) )
    model_all = []
    for jj in range(nstars):
        model_gen = np.dot(coeffs,features_data.T[:,jj]) 
        model_all.append(model_gen) 
        data_star = flux[10:8500,jj,1] 
        Cinv = 1. / (flux[10:8500,jj, 2] ** 2 + scatters ** 2)  # invvar slice of data
	Cinv[np.where(flux[10:8500,jj,2] > 1.)] = 0. # magic number 1.
        chi =  np.sqrt(Cinv) * (data_star - np.dot(coeffs, features_data.T[:,jj]))  
        chi2 = sum( (Cinv) * (data_star - np.dot(coeffs, features_data.T[:,jj]))**2) 
        #chi2 = (Cinv)*(model_gen - data_star)**2 
        chi2_all[jj] = chi2
        chi_all[:,jj] = chi
        plot_flag = 0
        if plot_flag != 0:
        # below plots to check for goodness of fit  
          fig = plt.figure()
          ax1 = fig.add_subplot(211)
          ax2 = fig.add_subplot(212)
          #ax2.plot(flux[:,jj,0],data_star- model_gen, 'r')
          noises = (flux[:,jj,2]**2 + scatters**2)**0.5
          ydiff_norm = 1./noises*(data_star - model_gen)
          bad = flux[:,jj,2] > 0.1
          ydiff_norm[bad] = None
          data_star[bad] = None
          model_gen[bad] = None
          ax1.plot(flux[10:8500,jj,0], data_star, 'k')
          ax1.plot(flux[10:8500,jj,0], model_gen, 'r')
          ax2.plot(flux[10:8500,jj,0],ydiff_norm , 'r')
          ax1.set_xlim(15200,16000)
          ax1.set_ylim(0.5,1.2)
          ax2.set_xlim(15200,16000)
          ax2.set_ylim(-10.2,10.2)
          prefix = str('check'+str(filein)+"_"+str(jj))
          savefig2(fig, prefix, transparent=False, bbox_inches='tight', pad_inches=0.5)
          close()
    #return chi2_all  
    fm = open("model_binary.pickle", 'w')
    pickle.dump((model_all), fm)
    fm.close()
    return chi_all

def savefig2(fig, prefix, **kwargs):
        suffix = ".png"
        print "writing %s" % (prefix + suffix)
        fig.savefig(prefix + suffix, **kwargs)


def _get_lvec(labels):
    """
    Constructs a label vector for an arbitrary number of labels
    Assumes that our model is quadratic in the labels
    Parameters
    ----------
    labels: np.ndarray
        pivoted label values for one star
    Returns
    -------
    lvec: np.ndarray
        label vector
    """
    nlabels = len(labels)
    # specialized to second-order model
    linear_terms = labels
    quadratic_terms = np.outer(linear_terms, 
                               linear_terms)[np.triu_indices(nlabels)]
    lvec = np.hstack((linear_terms, quadratic_terms))
    return lvec


def _func(coeffs, *labels):
    """ Takes the dot product of coefficients vec & labels vector 
    
    Parameters
    ----------
    coeffs: np.ndarray
        the coefficients on each element of the label vector
    *labels: np.ndarray
        label vector
    Returns
    -------
    dot product of coeffs vec and labels vec
    """
    nlabels = len(labels)
    linear_terms = labels
    quadratic_terms = np.outer(linear_terms, linear_terms)[np.triu_indices(nlabels)]
    lvec = np.hstack((linear_terms, quadratic_terms))
    return np.dot(coeffs[:,1:], lvec)

## non linear stuff below ##
# returns the non linear function 

# thankyou stack overflow for the example below on how to use the optimse function  
def nonlinear_invert(f, sigmas, coeffs, scatters,labels): 
    xdata = np.vstack([coeffs])
    sigmavals = np.sqrt(sigmas ** 2 + scatters ** 2) 
    guessit = [0]*len(labels)
    try: 
        model, cov = opt.curve_fit(_func, xdata, f, sigma = sigmavals, maxfev=18000, p0 = guessit)
    except RuntimeError:
        print "there has been a runtime error"
	model = [999]*len(labels)
	cov = np.ones((len(labels),len(labels) ))
    return model, cov


def infer_labels_nonlinear(fn_pickle,testdata_a, ids, fout_pickle, weak_lower,weak_upper):
#def infer_labels(fn_pickle,testdata, fout_pickle, weak_lower=0.935,weak_upper=0.98):
    """
    """
    file_in = open(fn_pickle, 'r') 
    dataall, metaall, labels, schmoffsets, coeffs, covs, scatters,chis,chisq = pickle.load(file_in)
    assert (round(offsets[-1],4)) == round(schmoffsets[-1],4) 
    file_in.close()
    testdata = testdata_a[10:8500,:,:]
    nstars = (testdata.shape)[1]
    nlabels = len(labels)
    Params_all = np.zeros((nstars, nlabels))
    Params_all_scaled = np.zeros((nstars, nlabels))
    MCM_rotate_all = np.zeros((nstars, np.shape(coeffs)[1]-1, np.shape(coeffs)[1]-1.))
    covs_all = np.zeros((nstars,nlabels, nlabels))
    for jj in range(0,nstars):
      print jj
      #if np.any(testdata[:,jj,0] != dataall[:, 0, 0]):
      if np.any(abs(testdata[:,jj,0] - dataall[:, 0, 0]) > 0.0001): 
          print testdata[range(5),jj,0], dataall[range(5),0,0]
          assert False
      xdata = testdata[:,jj,0]
      ydata = testdata[:,jj,1]
      ysigma = testdata[:,jj,2]
      ydata_norm = ydata  - coeffs[:,0] # subtract the np.mean 
      f = ydata_norm 
      Params,covs = nonlinear_invert(f,ysigma, coeffs, scatters,labels)
      Params= Params+offsets 
      Params_scaled= Params+offsets 
      for i in range(0,len(offsets)):
          Params[i] = unscale(Params[i], inputs[i]) 
          #covs[i] = unscale_covs(covs[i], inputs[i]) 
      #for a,b in zip(Params_unscaled, scales):
         #Params_scaled = unscaled(a,b)
      num_cut = -1*(np.shape(coeffs)[-1] -1) 
      coeffs_slice = coeffs[:,num_cut:]
      Cinv = 1. / (ysigma ** 2 + scatters ** 2)
      MCM_rotate = np.dot(coeffs_slice.T, Cinv[:,None] * coeffs_slice)
      Params_all[jj,:] = Params 
      Params_all_scaled[jj,:] = Params_scaled
      MCM_rotate_all[jj,:,:] = MCM_rotate 
      covs_all[jj,:,:] = covs
    filein = fout_pickle.split('_tags') [0] 
    if filein == 'self_2nd_order': 
      file_in = open(fout_pickle, 'w')  
      file_normed = normed_training_data.split('.pickle')[0]
      chi2 = get_goodness_fit(fn_pickle, file_normed, Params_all_scaled, MCM_rotate_all)
      #chi2 = 1.
      chi2_def = chi2#/len(xdata)*1.
      pickle.dump((Params_all, covs_all,chi2_def,ids),  file_in)
      file_in.close()
    else: 
      chi2 = get_goodness_fit(fn_pickle, filein, Params_all_scaled, MCM_rotate_all)
      #chi2 = 1.
      #chi2_def = chi2/len(xdata)*1.
      chi2_def = chi2
      file_in = open(fout_pickle, 'w')  
      pickle.dump((Params_all, covs_all, chi2_def, ids),  file_in)
      file_in.close()
    return Params_all , MCM_rotate_all


def lookatfits(fn_pickle, pixelvalues,testdataall): 
  #  """"
  #  this is to plot the individual pixel fits  on the 6x6 panel 
  #  """"
    file_in = open(fn_pickle, 'r') 
    testdataall, metaall, labels, schmoffsets, coeffs, covs, scatters,chis,chisqs = pickle.load(file_in)
    assert (round(offsets[-1],4)) == round(schmoffsets[-1],4) 
    #assert np.all(schmoffsets == offsets) 
    file_in.close()
    axis_t, axis_g, axis_feh = metaall[:,0], metaall[:,1], metaall[:,2]
    nstars = (testdataall.shape)[1]
    features = np.ones((nstars, 1))
    features = np.hstack((features, metaall - offsets)) 
    features2 = np.hstack((features, metaall )) 
    for each in pixelvalues:
        flux_val_abs = testdataall[each,:,1]
        flux_val_norm = testdataall[each,:,1] - np.dot(coeffs, features.T)[each,:] 
        coeff = coeffs[each,:] 
        y_feh_abs = coeff[3]*features[:,3] + coeff[0]*features[:,0]
        y_feh_norm = coeff[3]*features[:,3] + coeff[0]*features[:,0]  -(coeff[3]*features2[:,3] + coeff[0]*features2[:,0]) 
        y_g_abs = coeff[2]*features[:,2] + coeff[0]*features[:,0]
        y_g_norm = coeff[2]*features[:,2] + coeff[0]*features[:,0]  - (coeff[2]*features2[:,2] + coeff[0]*features2[:,0]) 
        y_t_abs = coeff[1]*features[:,1] + coeff[0]*features[:,0] 
        y_t_norm = coeff[1]*features[:,1] + coeff[0]*features[:,0] - (coeff[1]*features2[:,1] + coeff[0]*features2[:,0]) 
        for flux_val, y_feh, y_g, y_t, namesave,lab,ylims in zip([flux_val_abs, flux_val_norm], [y_feh_abs,y_feh_norm],[y_g_abs, y_g_norm], [y_t_abs,y_t_norm],['abs','norm'], ['flux','flux - np.mean'],
                [[-0.2,1.2], [-1,1]] ): 
            y_meandiff = coeff[0] - flux_val 
            fig = plt.figure(figsize = [12.0, 12.0])
            #
            ax = plt.subplot(3,2,1)
            pick = testdataall[each,:,2] > 0.1
            ax.plot(metaall[:,2], flux_val, 'o',alpha =0.5,mfc = 'None', mec = 'r') 
            ax.plot(metaall[:,2][pick], flux_val[pick], 'kx',markersize = 10) 
            ax.plot(metaall[:,2], y_feh, 'k') 
            ind1 = argsort(metaall[:,2]) 
            ax.fill_between(sort(metaall[:,2]), np.array(y_feh + std(flux_val))[ind1], np.array(y_feh - std(flux_val))[ind1] , color = 'y', alpha = 0.2)
            ax.set_xlabel("[Fe/H]", fontsize = 14 ) 
            ax.set_ylabel(lab, fontsize = 14 ) 
            ax.set_title(str(np.int((testdataall[each,0,0])))+"  $\AA$")
            ax.set_ylim(ylims[0], ylims[1]) 
            #
            ax = plt.subplot(3,2,2)
            ax.plot(metaall[:,1], flux_val, 'o', alpha =0.5, mfc = 'None', mec = 'b') 
            ax.plot(metaall[:,1][pick], flux_val[pick], 'kx',markersize = 10)  
            ax.plot(metaall[:,1], y_g, 'k') 
            ind1 = argsort(metaall[:,1]) 
            ax.fill_between(sort(metaall[:,1]), np.array(y_g + std(flux_val))[ind1], np.array(y_g - std(flux_val))[ind1] , color = 'y', alpha = 0.2)
            ax.set_xlabel("log g", fontsize = 14 ) 
            ax.set_ylabel(lab, fontsize = 14 ) 
            ax.set_title(str(np.int((testdataall[each,0,0])))+"  $\AA$")
            ax.set_ylim(ylims[0], ylims[1]) 
            #
            ax = plt.subplot(3,2,3)
            ax.plot(metaall[:,0], flux_val, 'o',alpha =0.5, mfc = 'None', mec = 'green') 
            ax.plot(metaall[:,0][pick], flux_val[pick], 'kx', markersize = 10) 
            ax.plot(metaall[:,0], y_t, 'k') 
            ind1 = argsort(metaall[:,0]) 
            ax.fill_between(sort(metaall[:,0]), np.array(y_t + std(flux_val))[ind1], np.array(y_t - std(flux_val))[ind1] , color = 'y', alpha = 0.2)
            ax.set_xlabel("Teff", fontsize = 14 ) 
            ax.set_ylabel(lab, fontsize = 14 ) 
            ax.set_ylim(ylims[0], ylims[1]) 
            #
            ax = plt.subplot(3,2,4)
            diff_flux = coeffs[each,0] - testdataall[each,:,1] 
            xrange1 = np.arange(0,shape(testdataall)[1],1) 
            ind1 = argsort(metaall[:,2]) 
            ind1_pick = argsort(metaall[:,2][pick]) 
            ax.plot(xrange1, (coeffs[each,0] - testdataall[each,:,1])[ind1], 'o',alpha = 0.5, mfc = 'None', mec = 'grey') 
            ax.plot(xrange1[pick], (coeffs[each,0] - testdataall[each,:,1][pick])[ind1_pick], 'kx',markersize = 10) 
            ax.fill_between(xrange1, np.array(mean(diff_flux) + std(diff_flux)), np.array(mean(diff_flux) - std(diff_flux))  , color = 'y', alpha = 0.2)
            ax.set_xlabel("Star Number (increasing [Fe/H])", fontsize = 14 ) 
            ax.set_ylabel("flux star - np.mean flux", fontsize = 14 ) 
            ax.set_ylim(-1.0, 1.0) 
            #
            ax = plt.subplot(3,2,5)
            for indx, color, label in [
                                       ( 1, "g", "Teff"),
                                       ( 2, "b", "logg"),
                                       ( 3, "r", "FeH")]:
              _plot_something(ax, testdataall[:, 0, 0][each-10:each+10], coeffs[:, indx][each-10:each+10], covs[:, indx, indx][each-10:each+10], color, label=label)
            ax.axvline(testdataall[:,0,0][each],color = 'grey') 
            ax.axhline(0,color = 'grey',linestyle = 'dashed') 
            ax.set_xlim(testdataall[:,0,0][each-9], testdataall[:,0,0][each+9]) 
            ax.legend(loc = 4,fontsize  = 10) 
            ax.set_xlabel("Wavelength $\AA$", fontsize = 14 ) 
            ax.set_ylabel("coeffs T,g,FeH", fontsize = 14 ) 
            #
            ax = plt.subplot(3,2,6)
            _plot_something(ax, testdataall[:, 0, 0][each-10:each+10], coeffs[:, 0][each-10:each+10], covs[:, 0, 0][each-10:each+10], 'k', label='mean')
            ax.set_ylim(0.6,1.1) 
            ax.set_xlim(testdataall[:,0,0][each-9], testdataall[:,0,0][each+9]) 
            ax.legend(loc = 4,fontsize  = 10) 
            ax.axvline(testdataall[:,0,0][each],color = 'grey') 
            ax.axhline(0,color = 'grey',linestyle = 'dashed') 
            ax.set_xlabel("Wavelength $\AA$", fontsize = 14 ) 
            ax.set_ylabel("Mean flux", fontsize = 14 ) 

            savefig(fig, str(each)+"_"+str(namesave) , transparent=False, bbox_inches='tight', pad_inches=0.5)
            fig.clf()
       # return 

def _plot_something(ax, wl, val, var, color, lw=2, label=""):
    factor = 1.
    if label == "Teff": factor = 1000. # yes, I feel dirty; MAGIC
    sig = np.sqrt(var)
    ax.plot(wl, factor*(val+sig), color=color, lw=lw, label=label)
    ax.plot(wl, factor*(val-sig), color=color, lw=lw) 
    ax.fill_between(wl, factor*(val+sig), factor*(val-sig), color = color, alpha = 0.2) 
    return None
  
    

def savefig(fig, prefix, **kwargs):
 #   for suffix in (".png"):
    suffix = ".png"
    print "writing %s" % (prefix + suffix)
    fig.savefig(prefix + suffix)#, **kwargs)
    close() 


def leave_one_cluster_out():
# this is the test routine to leave one cluster out 
    dataall, metaall, labels, Ametaall, cluster_name, ids= get_normalized_training_data_tsch()
    nameu = unique(cluster_name) 
    nameu = np.array(nameu) 
    cluster_name = np.array(cluster_name)
    for each in nameu:
      clust_pick = each
      take = np.array(cluster_name) == clust_pick
      inds = np.arange(0,len(cluster_name),1) 
      inds1 = inds[take] 
      cluster_take = each #cluster_name[take][0]
      #return inds1, cluster_name
      train(dataall, metaall,  2,  fpickle2, Ametaall, cluster_name, logg_cut= 40.,teff_cut = 0., leave_out=inds1)
      field = "self_2nd_order_alpha_"
      file_in = open(normed_training_data, 'r') 
      testdataall, metaall, labels, Ametaall, cluster_name, ids = pickle.load(file_in)
      file_in.close() 
      testmetaall, inv_covars = infer_labels_nonlinear("coeffs_2nd_order_14_cal_filt_dr13.pickle", testdataall,ids, field+str(cluster_take)+"_tags_logmass.pickle",-10.950,10.99) 
      #plot_leave_one_out(field, clust_pick) 
    return 

def leave_one_star_out():
# this is the test routine to leave one star out 
    dataall, metaall, labels, Ametaall, cluster_name, ids= get_normalized_training_data()
    #nameu = unique(cluster_name) 
    #nameu = np.array(nameu) 
    cluster_name = np.array(cluster_name)
    ids = np.array(ids)
    idsnew = [] 
    for each in ids: 
      if len(ids) > 20:
        idsnew.append(each.split('2m')[-1]) 
      else: 
        idsnew.append(each.split)
    idsnew = np.array(idsnew) 
    nameu = [a+"_"+b for a,b in zip(cluster_name, idsnew)] 
    nameu = np.array(nameu) 
    for each in nameu:
      name_pick = each
      take = np.array(nameu) == name_pick
      inds = np.arange(0,len(cluster_name),1) 
      inds1 = inds[take] 
      star_take = each #cluster_name[take][0]
      #return inds1, cluster_name
      train(dataall, metaall,  2,  fpickle2, Ametaall, cluster_name, logg_cut= 40.,teff_cut = 0., leave_out=inds1)
      # up to here 
      field = "self_2nd_order_5_"
      file_in = open(normed_training_data, 'r') 
      testdataall, metaall, labels, Ametaall, cluster_name, ids = pickle.load(file_in)
      file_in.close() 
      testmetaall, inv_covars = infer_labels_nonlinear("coeffs_2nd_order_14_cal_filt_dr13.pickle", testdataall[:,take], idsnew[take], field+str(star_take)+"_itags_logmass.pickle",-10.950,10.99) 
      #plot_leave_one_out(field, clust_pick) 
    return 

def plot_leave_one_out(filein,cluster_out): 
    file_in2 = open(filein+"tags_logmass.pickle", 'r') 
    params, covs_params = pickle.load(file_in2)
    sp = np.shape(params) 
    params = np.array(params)
    covs_params = np.array(covs_params)
    file_in2.close()
    # this is the test to 
    filein2 = 'test14.txt' # originally had for test4g_self and for ages_test4g_self that goes with this
    filein2 = 'test18.txt' # originally had for test4g_self and for ages_test4g_self that goes with this
    filein3 = 'ages.txt' # note ages goes with test14 
    plot_markers = ['ko', 'yo', 'ro', 'bo', 'co','k*', 'y*', 'r*', 'b*', 'c*', 'ks', 'rs', 'bs', 'cs', 'rd', 'kd', 'bd', 'rd', 'mo', 'ms' ]
    # M92, M15, M53, N5466, N4147, M13, M2, M3, M5, M107, M71, N2158, N2420, Pleaides, N7789, M67, N6819 , N188, N6791 
    t,g,feh,t_err,feh_err = np.loadtxt(filein2, usecols = (4,6,8,16,17), unpack =1) 
    tA,gA,fehA = np.loadtxt(filein2, usecols = (3,5,7), unpack =1) 
    age = np.loadtxt(filein3, usecols = (0,), unpack =1) 
    g_err, age_err = [0]*len(g) , [0]*len(g) 
    g_err, age_err = np.array(g_err), np.array(age_err) 
    diffT = abs(array(t) - np.array(tA) ) 
    a = open(filein2) 
    al = a.readlines() 
    
    names = []
    for each in al:
      names.append(each.split()[1]) 
    diffT = np.array(diffT) 
    #pick =logical_and(names != cluster_name,  diffT < 600. ) 
    names = np.array(names) 
    #pick =  diffT < 600. # I need to implement this < 6000 K 
    #pick2 =logical_and(names == cluster_out,  diffT < 600. ) 
    pick =  diffT < 6000. # I need to implement this < 6000 K 
    pick2 =logical_and(names == cluster_out,  diffT < 6000. ) 

    t_sel,g_sel,feh_sel,t_err_sel,g_err_sel,feh_err_sel = t[pick2], g[pick2], feh[pick2], t_err[pick2], g_err[pick2], feh_err[pick2] 
    t,g,feh,t_err,g_err,feh_err = t[pick], g[pick], feh[pick], t_err[pick], g_err[pick], feh_err[pick] 
    #
    names = np.array(names) 
    names = names[pick] 
    unames = unique(names) 
    starind = np.arange(0,len(names), 1) 
    name_ind = [] 
    names = np.array(names) 
    for each in unames:
      takeit = each == names 
      name_ind.append(np.int(starind[takeit][-1]+1. ) )
    cluster_ind = [0] + list(sort(name_ind))# + [len(al)]
    #
    params_sel = np.array(params)[pick2]
    covs_params_sel = np.array(covs_params)[pick2]
    params = np.array(params)[pick]
    covs_params = np.array(covs_params)[pick]
    sp2 = np.shape(params) 
    sp3 = len(t) 
    rcParams['figure.figsize'] = 12.0, 10.0
    fig, temp = pyplot.subplots(3,1, sharex=False, sharey=False)
    fig = plt.figure() 
    ax = fig.add_subplot(111, frameon = 0 ) 
    ax.set_ylabel("The Cannon", labelpad = 40, fontsize = 20 ) 
    ax.tick_params(labelcolor= 'w', top = 'off', bottom = 'off', left = 'off', right = 'off' ) 
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    params_labels = [params[:,0], params[:,1], params[:,2] ,  covs_params[:,0,0]**0.5, covs_params[:,1,1]**0.5, covs_params[:,2,2]**0.5 ]
    cval = ['k', 'b', 'r', ] 
    input_ASPCAP = [t, g, feh, t_err, g_err, feh_err ] 
    listit_1 = [0,1,2]
    listit_2 = [1,0,0]
    axs = [ax1,ax2,ax3]
    labels = ['teff', 'logg', 'Fe/H']
    for i in range(0,len(cluster_ind)-1): 
      indc1 = cluster_ind[i]
      indc2 = cluster_ind[i+1]
      for ax, num,num2,label1,x1,y1 in zip(axs, listit_1,listit_2,labels, [4800,3.0,0.3], [3400,1,-1.5]): 
        pick = np.logical_and(g[indc1:indc2] > 0, np.logical_and(t_err[indc1:indc2] < 300, feh[indc1:indc2] > -4.0) ) 
        cind = np.array(input_ASPCAP[1][indc1:indc2][pick]) 
        cind = np.array(input_ASPCAP[num2][indc1:indc2][pick]).flatten() 
        ax.plot(input_ASPCAP[num][indc1:indc2][pick], params_labels[num][indc1:indc2][pick], plot_markers[i]) 
    
    ax1.plot(params_sel[:,0], t_sel, 'y*', label = cluster_out, markersize = 14)
    ax2.plot(params_sel[:,1], g_sel, 'y*', label = cluster_out, markersize = 14)
    ax3.plot(params_sel[:,2], feh_sel, 'y*', label = cluster_out, markersize = 14)
    ax1.legend(loc=2,numpoints=1)
    ax2.legend(loc=2,numpoints=1)
    ax3.legend(loc=2,numpoints=1)
   
    ax1.text(5400,3700,"y-axis, $<\sigma>$ = "+str(round(mean(params_labels[0+3]),2)),fontsize = 14) 
    ax2.text(3.9,1,"y-axis, $<\sigma>$ = "+str(round(mean(params_labels[1+3]),2)),fontsize = 14) 
    ax3.text(-0.3,-2.5,"y-axis, $<\sigma>$ = "+str(round(mean(params_labels[2+3]),2)),fontsize = 14) 
    ax1.plot([0,6000], [0,6000], linewidth = 1.5, color = 'k' ) 
    ax2.plot([0,5], [0,5], linewidth = 1.5, color = 'k' ) 
    ax3.plot([-3,2], [-3,2], linewidth = 1.5, color = 'k' ) 
    ax1.set_xlim(3500, 6000) 
    ax1.set_ylim(1000,6000)
    ax1.set_ylim(3500,6000)
    ax2.set_xlim(0, 5) 
    ax3.set_xlim(-3, 1) 
    ax1.set_xlabel("ASPCAP Teff, [K]", fontsize = 14,labelpad = 5) 
    ax1.set_ylabel("Teff, [K]", fontsize = 14,labelpad = 5) 
    ax2.set_xlabel("ASPCAP logg, [dex]", fontsize = 14,labelpad = 5) 
    ax2.set_ylabel("logg, [dex]", fontsize = 14,labelpad = 5) 
    ax3.set_xlabel("ASPCAP [Fe/H], [dex]", fontsize = 14,labelpad = 5) 
    ax3.set_ylabel("[Fe/H], [dex]", fontsize = 14,labelpad = 5) 
    ax2.set_ylim(0,5)
    ax3.set_ylim(-3,1) 
    fig.subplots_adjust(hspace=0.22)
    prefix = "/Users/ness/Downloads/Apogee_Raw/calibration_apogeecontinuum/documents/plots/"+str(cluster_out)+"_out"
    savefig2(fig, prefix, transparent=False, bbox_inches='tight', pad_inches=0.5)
    close("all")
    print sp, sp2, sp3
    return 

if __name__ == "__main__":
    pixlist = np.loadtxt("pixtest4.txt", usecols = (0,), unpack =1) 
    # if want to read in list of 1640 stars
    num1,num2=0,1640
    numtake = np.arange(0,1639,1)
    print 'main:reading normed data file' 
    dataall_a, filterall_aa, metaall_aa, labels, Ametaall, cluster_name, ids = get_normalized_training_data(pixlist,numtake, label_array, labels, filter_array)
    nl = 9 # number of labels want to fit for 
    metaall_a = metaall_aa[:,0:nl]
    offsets = hstack((offsets[0:nl], offsets[0:nl]))
    labels = hstack((labels[0:nl], labels[0:nl])) 
    inputs = array(inputs)
    inputs = vstack((inputs[0:nl,:], inputs[0:nl,:]))
    filterall_a = filterall_aa[:,0:nl]
    fpickle2 = coeffs_file #"coeffs_dr13_filt.pickle"
    if not glob.glob(fpickle2):
        train(dataall_a, filterall_a, metaall_a, 2,  fpickle2, Ametaall, cluster_name, logg_cut= 40.,teff_cut = 0.)
    self_flag = 0
    #self_flag = 2

    if self_flag < 1:
      startTime = datetime.now()
      a = open('binaries.txt', 'r') 
      al = a.readlines()
      bl = []
      for each in al:
        bl.append(each.strip()) 
      for each in bl: 
        testfile = each
        field = testfile.split('.txt')[0]+'_' #"4332_"
        testdataall_a, ids = get_normalized_test_data(testfile,pixlist)
        # file already exists and is normalised
        #testdataall_a, ids = get_normalized_test_data_tsch(testfile,pixlist)
        testmetaall, inv_covars = infer_labels_nonlinear(coeffs_file, testdataall_a, ids, field+tags_file,0.00,1.40) 
        print(datetime.now()-startTime)
    if self_flag == 2:
      field = "self_2nd_order_"
      file_in = open(normed_training_data, 'r') 
      testdataall, filterall, metaall, labels, Ametaall, cluster_name,ids = pickle.load(file_in)
      file_in.close() 
      testmetaall, inv_covars = infer_labels_nonlinear(coeffs_file, testdataall, ids, field+tags_file,-10.950,10.99) 
