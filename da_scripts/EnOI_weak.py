"""
This script runs the whole cycled DA experiment using EnOI.
"""
import numpy as np
from scipy.linalg import solve
from scipy import interpolate
from scipy import sparse
import h5py
from netCDF4 import Dataset
import subprocess
import timeit
import glob
import os

# Time setup
tic = timeit.default_timer()

# Set parameter values
Ne = 400 # Ensemble size
Nt = 600 # Number of cycles
N = 3*384*95 + 2*384*96 + 1536**2 + 3*1535**2 # Total number of variables.
rInf = 1 # inflation factor for ensemble perturbations

# Set up paths to executables & restart.nc files
ref_dir = "/projects/groomsi/q-gcm/examples/ref/"
mean_dir = mean_dir = os.environ['SLURM_SCRATCH']+'/en_weak/'
analog_file = "/rc_scratch/groomsi/data.h5"

# load observation configuration
from obs_config_weak import *

# Set up hdf5 file for output
with h5py.File(mean_dir+"mean.h5", "w",swmr=True) as mean_out:
    mean_out.create_dataset("forecast/pa",(Nt,3,95,384))
    mean_out.create_dataset("forecast/hmixa",(Nt,96,384))
    mean_out.create_dataset("forecast/ast",(Nt,96,384))
    mean_out.create_dataset("forecast/sst",(Nt,1536,1536))
    mean_out.create_dataset("forecast/po",(Nt,3,1535,1535))
    mean_out.create_dataset("analysis/pa",(Nt,3,95,384))
    mean_out.create_dataset("analysis/hmixa",(Nt,96,384))
    mean_out.create_dataset("analysis/ast",(Nt,96,384))
    mean_out.create_dataset("analysis/sst",(Nt,1536,1536))
    mean_out.create_dataset("analysis/po",(Nt,3,1535,1535))

# Load library/catalog into array A.
A = np.zeros((N,Ne))
po2 = np.zeros((3,257,257))
po = np.zeros((3,1535,1535))
sst = np.zeros((1536,1536))
with h5py.File(analog_file,'r',swmr=True) as f:
    inc = f['at'].shape[0]//Ne
    for i in range(Ne):
        ast = np.array(f['at'][i*inc,:,:,0])
        hmixa = np.log(np.array(f['at'][i*inc,:,:,1]))
        pa = np.moveaxis(np.array(f['at'][i*inc,1:,:,2:]),2,0)
        sst2 = np.array(f['oc'][i*inc,:,:,0])
        po2[:,:-1,:-1] = np.moveaxis(np.array(f['oc'][i*inc,:,:,1:]),2,0)
        po2[:,-1,:] = po2[:,0,:] # Add boundaries back in
        po2[:,:,-1] = po2[:,:,0] # Add boundaries back in
        for j in range(3):
            po[j,:,:] = interpolate.RectBivariateSpline(np.arange(0,1537,6),np.arange(0,1537,6),po2[j,:,:],kx=3,ky=3)(np.arange(1,1536),np.arange(1,1536))
        sst = interpolate.RectBivariateSpline(np.arange(0,1536,6),np.arange(0,1536,6),sst2,kx=3,ky=3)(np.arange(0,1536),np.arange(0,1536))
        # Now put the data into an ensemble matrix. We will flatten each array using row-major ordering
        # with the order pa,hmixa,ast,sst,po
        A[:,i] = np.concatenate((pa[:,:,:].flatten(),hmixa[:,:].flatten(),ast[:,:].flatten(),
                               sst[:,:].flatten(),po[:,:,:].flatten()))

del po2, sst2

# Partial setup time
toc = timeit.default_timer()
print(toc-tic,' A before subtracting mean and scaling ',flush=True)

# The line below *does not* initialize the mean, it merely allocates space.
# The mean is currently initialized as just whatever is in the restart.nc file.
# If we want something else we would have to write it to the restart.nc file.
x_mean = np.mean(A,axis=1)

# Set up A matrix
for i in range(Ne):
    A[:,i] -= x_mean

# rescale so that the mean variance on each block is 1
A[ipa:ipa1,:] *= (1/np.sqrt(np.mean(A[ipa:ipa1,:]**2)))
A[ipa1:ipa2,:] *= (1/np.sqrt(np.mean(A[ipa1:ipa2,:]**2)))
A[ipa2:ihmixa,:] *= (1/np.sqrt(np.mean(A[ipa2:ihmixa,:]**2)))
A[ihmixa:iast,:] *= (1/np.sqrt(np.mean(A[ihmixa:iast,:]**2)))
A[iast:isst,:] *= (1/np.sqrt(np.mean(A[iast:isst,:]**2)))
A[isst:ipo,:] *= (1/np.sqrt(np.mean(A[isst_box,:]**2)))
A[ipo:,:] *= (1/np.sqrt(np.mean(A[ipo:,:]**2)))

# initialize scaling factors for A
r_pa0 = 1
r_pa1 = 1
r_pa2 = 1
r_hmixa = 1
r_ast = 1
r_sst = 1
r_po = 1

# Total setup time
toc = timeit.default_timer()
print(toc-tic,' seconds total setup',flush=True)

# Assimilation loop: Assimilate then forecast
for k in range(Nt):
    tic = timeit.default_timer()

    # Get ref state
    with h5py.File(ref_dir+"ref.h5", "r", swmr=True) as rootgrp_ref:
        pa_ref = np.array(rootgrp_ref["pa"][k,:,:,:])
        hmixa_ref = np.log(np.array(rootgrp_ref["hmixa"][k,:,:]))
        ast_ref = np.array(rootgrp_ref["ast"][k,:,:])
        sst_ref = np.array(rootgrp_ref["sst"][k,:,:])
        po_ref = np.array(rootgrp_ref["po"][k,:,:,:])

    # Get forecast.
    with Dataset(mean_dir+"outdata/restart.nc", "r", format="NETCDF4") as rootgrp_mean:
        pa_mean = np.array(rootgrp_mean["pa"][:,1:-1,:-1])
        hmixa_mean = np.log(np.array(rootgrp_mean["hmixa"][:,:]))
        ast_mean = np.array(rootgrp_mean["ast"][:,:])
        sst_mean = np.array(rootgrp_mean["sst"][:,:])
        po_mean = np.array(rootgrp_mean["po"][:,1:-1,1:-1])

    # Save forecast to disk
    with h5py.File(mean_dir+"mean.h5", "r+", swmr=True) as mean_out:
        mean_out["forecast/pa"][k,:,:,:] = pa_mean
        mean_out["forecast/hmixa"][k,:,:] = np.exp(hmixa_mean)
        mean_out["forecast/ast"][k,:,:] = ast_mean
        mean_out["forecast/sst"][k,:,:] = sst_mean
        mean_out["forecast/po"][k,:,:,:] = po_mean

    # get x_mean
    x_mean = np.concatenate((pa_mean[:,:,:].flatten(),hmixa_mean[:,:].flatten(),ast_mean[:,:].flatten(),
                             sst_mean[:,:].flatten(),po_mean[:,:,:].flatten()))
    # get observations
    if( k%ocn_assim_period == 0 ): # observe atm and ocn
        # Get y
        y = np.concatenate((pa_ref[:,first_obs_atm_y:95:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm].flatten(),
                            hmixa_ref[first_obs_atm_y:96:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm].flatten(),
                            ast_ref[first_obs_ast_y:96:inc_obs_ast,first_obs_ast_x:384:inc_obs_ast].flatten(),
                            sst_ref[first_obs_sst_y:last_obs_sst_y:inc_obs_sst,first_obs_sst_x:last_obs_sst_x:inc_obs_sst].flatten(),
                            po_ref[0,first_obs_po_y:1535:inc_obs_po,first_obs_po_x:1535:inc_obs_po].flatten()))
        y = y + np.sqrt(tot_obs_err_var)*np.random.standard_normal(No_tot)

        # get Hx
        Hx = np.concatenate((pa_mean[:,first_obs_atm_y:95:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm].flatten(),
                             hmixa_mean[first_obs_atm_y:96:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm].flatten(),
                             ast_mean[first_obs_ast_y:96:inc_obs_ast,first_obs_ast_x:384:inc_obs_ast].flatten(),
                             sst_mean[first_obs_sst_y:last_obs_sst_y:inc_obs_sst,first_obs_sst_x:last_obs_sst_x:inc_obs_sst].flatten(),
                             po_mean[0,first_obs_po_y:1535:inc_obs_po,first_obs_po_x:1535:inc_obs_po].flatten()))

    else: # assimilate just atm
        # Get y
        y = np.concatenate((pa_ref[:,first_obs_atm_y:95:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm].flatten(),
                            hmixa_ref[first_obs_atm_y:96:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm].flatten(),
                            ast_ref[first_obs_ast_y:96:inc_obs_ast,first_obs_ast_x:384:inc_obs_ast].flatten()))
        y = y + np.sqrt(atm_obs_err_var)*np.random.standard_normal(No_atm)

        # Get Hx
        Hx = np.concatenate((pa_mean[:,first_obs_atm_y:95:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm].flatten(),
                             hmixa_mean[first_obs_atm_y:96:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm].flatten(),
                             ast_mean[first_obs_ast_y:96:inc_obs_ast,first_obs_ast_x:384:inc_obs_ast].flatten()))

    # Get innovation vector
    innov = y - Hx # innovation vector

    # Undo previous scaling of A for atm fields
    A[ipa:ipa1,:] *= (1/r_pa0)
    A[ipa1:ipa2,:] *= (1/r_pa1)
    A[ipa2:ihmixa,:] *= (1/r_pa2)
    A[ihmixa:iast,:] *= (1/r_hmixa)
    A[iast:isst,:] *= (1/r_ast)

    # Get estimate of forecast spread for all atm fields.
    r_pa0 = rInf*np.sqrt(np.mean(innov[ipa_obs:ipa1_obs]**2) - obs_var_pa)
    if np.isnan(r_pa0):
        r_pa0 = 0.5*np.sqrt(obs_var_pa)
    r_pa1 = rInf*np.sqrt(np.mean(innov[ipa1_obs:ipa2_obs]**2) - obs_var_pa)
    if np.isnan(r_pa1):
        r_pa1 = 0.5*np.sqrt(obs_var_pa)
    r_pa2 = rInf*np.sqrt(np.mean(innov[ipa2_obs:ihmixa_obs]**2) - obs_var_pa)
    if np.isnan(r_pa2):
        r_pa2 = 0.5*np.sqrt(obs_var_pa)
    r_hmixa = rInf*np.sqrt(np.mean(innov[ihmixa_obs:iast_obs]**2) - obs_var_hmixa)
    if np.isnan(r_hmixa):
         r_hmixa = 0.5*np.sqrt(obs_var_hmixa)
    r_ast = rInf*np.sqrt(np.mean(innov[iast_obs:isst_obs]**2) - obs_var_ast)
    if np.isnan(r_ast):
        r_ast = 0.5*np.sqrt(obs_var_ast)

    # rescale atm ensemble perturbations to match innovations
    A[ipa:ipa1,:] *= r_pa0/np.sqrt((Ne-1)*np.mean(A[ipa:ipa1,:]**2))
    A[ipa1:ipa2,:] *= r_pa1/np.sqrt((Ne-1)*np.mean(A[ipa1:ipa2,:]**2))
    A[ipa2:ihmixa,:] *= r_pa2/np.sqrt((Ne-1)*np.mean(A[ipa2:ihmixa,:]**2))
    A[ihmixa:iast,:] *= r_hmixa/np.sqrt((Ne-1)*np.mean(A[ihmixa:iast,:]**2))
    A[iast:isst,:] *= r_ast/np.sqrt((Ne-1)*np.mean(A[iast:isst,:]**2))

    # Update the ocn part of the A matrix
    if( k%ocn_assim_period  == 0 ):
        # Undo previous scaling of A for ocn fields
        A[isst:ipo,:] *= (1/r_sst)
        A[ipo:,:] *= (1/r_po)

        # Estimate forecast spread
        r_sst = rInf*np.sqrt(np.mean(innov[isst_obs:ipo_obs]**2) - obs_var_sst)
        if np.isnan(r_sst):
            r_sst = 0.5*np.sqrt(obs_var_sst)
        r_po = rInf*np.sqrt(np.mean(innov[ipo_obs:]**2) - obs_var_po)
        if np.isnan(r_po):
            r_po = 0.5*np.sqrt(obs_var_po)

        # rescale ocn ensemble perturbations to match innovations
        if is_sst_box:
            A[isst:ipo,:] *= r_sst/np.sqrt((Ne-1)*np.mean(A[isst_box,:]**2))
        else:
            A[isst:ipo,:] *= r_sst/np.sqrt((Ne-1)*np.mean(A[isst:ipo,:]**2))
        A[ipo:,:] *= r_po/np.sqrt((Ne-1)*np.mean(A[ipo:,:]**2))

    # Assimilate
    if( k%ocn_assim_period == 0 ): # assimilate atm and ocn obs
        V = A[ind_obs_tot,:] # Scaled observation ensemble perturbation matrix

        # Compute c = (HBH^T +  R)^{-1}(y - Hx)
        c = solve(loc_HLHT_tot*(V@V.T) + R_tot,innov,sym_pos=True)

        # The following line would assimilate all at once (no localization)
        # x_mean = x_mean + A@(V.T@c)

        # Assimilate with localization
        for i in range(No_tot):
            incr_ind = loc_LHT_tot[:,i].indices
            x_mean[incr_ind] += c[i]*np.squeeze(loc_LHT_tot[incr_ind,i].toarray())*(A[incr_ind,:]@V[i,:])

    else:
        V = A[ind_obs_atm, :]  # Scaled observation ensemble perturbation matrix

        # Compute c = (HBH^T +  R)^{-1}(y - Hx)
        c = solve(loc_HLHT_atm*(V@V.T) + R_atm,innov,sym_pos=True)

        # The following line would assimilate all at once (no localization)
        # x_mean = x_mean + A@(V.T@c)

        # Assimilate with localization
        for i in range(No_atm):
            incr_ind = loc_LHT_atm[:,i].indices
            x_mean[incr_ind] += c[i]*np.squeeze(loc_LHT_atm[incr_ind,i].toarray())*(A[incr_ind,:]@V[i,:])

    # extract components of the analysis mean
    pa = np.zeros((3,97,385))
    pa[:,1:-1,:-1] = np.reshape(x_mean[ipa:ihmixa],(3,95,384))
    pa[:,0,:-1] = np.mean(pa[:,1,:-1],axis=1)[:,np.newaxis] # set southern boundary
    pa[:,-1,:-1] = np.mean(pa[:,-2,:-1],axis=1)[:,np.newaxis] # set northern boundary
    pa[:,:,-1] = pa[:,:,0] # periodic east/west
    hmixa = np.reshape(x_mean[ihmixa:iast],(96,384))
    ast = np.reshape(x_mean[iast:isst],(96,384))
    sst = np.minimum(np.maximum(-15,np.reshape(x_mean[isst:ipo],(1536,1536))),30)
    po = np.zeros((3,1537,1537))
    po[:,1:-1,1:-1] = np.reshape(x_mean[ipo:],(3,1535,1535))
    for level in range(3):
        po_bc = np.mean(np.concatenate((po[level,1,1:-1],po[level,-2,1:-1],po[level,2:-2,1],po[level,2:-2,-2])))
        po[level,0,:] = po_bc # southern boundary
        po[level,-1,:] = po_bc # northern boundary
        po[level,1:-1,0] = po_bc # eastern boundary
        po[level,1:-1,-1] = po_bc # western boundary

    # Save analysis to disk
    with h5py.File(mean_dir+"mean.h5", "r+", swmr=True) as mean_out:
        mean_out["analysis/pa"][k,:,:,:] = pa[:,1:-1,:-1]
        mean_out["analysis/hmixa"][k,:,:] = np.exp(hmixa)
        mean_out["analysis/ast"][k,:,:] = ast
        mean_out["analysis/sst"][k,:,:] = sst
        mean_out["analysis/po"][k,:,:,:] = po[:,1:-1,1:-1]

    # write analysis mean to the restart.nc file
    with Dataset(mean_dir+"outdata/restart.nc", "r+", format="NETCDF4") as rootgrp_mean:
        rootgrp_mean["pa"][:,:,:] = pa
        rootgrp_mean["hmixa"][:,:] = np.exp(hmixa)
        rootgrp_mean["ast"][:,:] = ast
        rootgrp_mean["sst"][:,:] = sst
        rootgrp_mean["po"][:,:,:] = po

    toc = timeit.default_timer()
    print(toc-tic,' seconds assimilation cycle',flush=True)

    # forecast ensemble mean
    tic = timeit.default_timer()
    subprocess.run(mean_dir+"q-gcm",cwd=mean_dir,stdout=subprocess.DEVNULL)
    toc = timeit.default_timer()
    print(toc-tic,' seconds forecast',flush=True)

    # Check if q-gcm crashed
    if not(not glob.glob(mean_dir+'core*')):
        exit()
