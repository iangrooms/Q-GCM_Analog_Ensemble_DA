"""
This script runs the whole cycled DA experiment using cAnEnOI
Analogs are constructed by a combination of localization and interpolation
"""
import numpy as np
from scipy.linalg import solve
from scipy import interpolate
from scipy.signal import resample
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
Ne = 100 # Ensemble size
Nt = 600 # Number of cycles
N = 3*384*95 + 2*384*96 + 1536**2 + 3*1535**2 # Total number of variables.
rInf = 1 # inflation factor for ensemble perturbations

# Set number of sub-domains for interpolation
N_x_atm = 32 # size of local atm subdomains in the x direction
N_subs_atm = 384 // N_x_atm # number of local atm subdomains in the x direction
N_x_ocn = 32 # size of local ocn subdomains in the x direction. This is on the coarsened 256x256 grid.
N_subs_ocn = 256 // N_x_ocn # number of local ocn subdomains in the x direction

# Set up paths to executables & restart.nc files
ref_dir = "/projects/groomsi/q-gcm/examples/ref/"
mean_dir = os.environ['SLURM_SCRATCH']+'/lan_weak/'
analog_file = os.environ['SLURM_SCRATCH']+'/data.h5'

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
    mean_out.create_dataset("analysis/po",(Nt,3,1535,1535,))

# Load library/catalog into array A.
A = np.zeros((N,Ne))
with h5py.File(analog_file,'r',swmr=True) as f:
    catalogA = np.array(f['at'][:,1::2,::2,:]) # Subsample atmosphere to save space
    catalogS = np.array(f['oc'][:,96:160,:,:2]) # Central strip, only sst and po[top]

catalogA[:,:,:,1] = np.log(catalogA[:,:,:,1])
N_lib = catalogA.shape[0]

# The line below *does not* initialize the mean, it merely allocates space.
# The mean is currently initialized as just whatever is in the restart.nc file.
# If we want something else we would have to write it to the restart.nc file.
x_mean = np.mean(A,axis=1)

# Prep for local interpolated analogs
clim_var_sst = np.array([11.9,4.4,2.1,1.5,1.2,1.0,0.9,0.7])
clim_var_po = np.array([23.3,15.4,10.3,7.1,5.5,4.2,2.9,2.1])
d2_atm = np.zeros((N_subs_atm,N_lib))
d2_ocn = np.zeros((N_subs_ocn,N_lib))
analog_indices_atm = np.zeros((N_subs_atm,Ne))
analog_indices_ocn = np.zeros((N_subs_ocn,Ne))

# Set up the interpolation weights - atm
w_interp_atm = np.zeros((N_subs_atm,96,384))
y = np.zeros(N_subs_atm)
y[0] = 1
y_interpft = resample(y,384)
for i in range(N_subs_atm):
    w_interp_atm[i,:,:] = np.roll(y_interpft,N_x_atm//2 + i*N_x_atm)

# Set up the interpolation weights - ocn
x_tmp = np.arange(N_x_ocn//2,256 -N_x_ocn//2 +1,N_x_ocn)
w_interp_ocn = np.zeros((N_subs_ocn,256,256))
y = np.zeros(N_subs_ocn)
x_grid = np.arange(N_x_ocn//2,256 -N_x_ocn//2 +1)
for i in range(N_subs_ocn):
    y[i] = 1
    tck = interpolate.splrep(x_tmp, y)
    w_interp_ocn[i,:,N_x_ocn//2:(256 -N_x_ocn//2 +1)] = interpolate.splev(x_grid,tck)
    w_interp_ocn[i,:,:N_x_ocn//2] = w_interp_ocn[i,:,N_x_ocn//2][:,np.newaxis]
    w_interp_ocn[i,:,(256 -N_x_ocn//2 +1):] = w_interp_ocn[i,:,256 -N_x_ocn//2][:,np.newaxis]
    y[i] = 0
del x_tmp, x_grid, y, tck

# Allocate some space
E_ocn = np.zeros((Ne,256,256,4))
po2 = np.zeros((3,257,257))
po_A = np.zeros((3,1535,1535))

# Setup time
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

    else: # observe just atm
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

    # find indices of Ne nearest neighbors, local, atm-only
    for i in range(N_subs_atm):
        d2_atm[i,:] = np.tanh(np.mean((ast_mean[np.newaxis,1::2,(i*N_x_atm):((i+1)*N_x_atm):2]-catalogA[:,:,(i*N_x_atm//2):((i+1)*N_x_atm//2),0])**2,axis=(1,2))/18) # scaled squared distances
        d2_atm[i,:] += np.tanh(np.mean((hmixa_mean[np.newaxis,1::2,(i*N_x_atm):((i+1)*N_x_atm):2]-catalogA[:,:,(i*N_x_atm//2):((i+1)*N_x_atm//2),1])**2,axis=(1,2))/0.02)
        d2_atm[i,:] += np.tanh(np.mean((pa_mean[np.newaxis,0,::2,(i*N_x_atm):((i+1)*N_x_atm):2]-catalogA[:,:,(i*N_x_atm//2):((i+1)*N_x_atm//2),2])**2,axis=(1,2))/1e5)
        d2_atm[i,:] += np.tanh(np.mean((pa_mean[np.newaxis,1,::2,(i*N_x_atm):((i+1)*N_x_atm):2]-catalogA[:,:,(i*N_x_atm//2):((i+1)*N_x_atm//2),3])**2,axis=(1,2))/5.3e5)
        d2_atm[i,:] += np.tanh(np.mean((pa_mean[np.newaxis,2,::2,(i*N_x_atm):((i+1)*N_x_atm):2]-catalogA[:,:,(i*N_x_atm//2):((i+1)*N_x_atm//2),4])**2,axis=(1,2))/2.75e5)
        analog_indices_atm[i,:] = np.argsort(d2_atm[i,:])[0:Ne]

    # Get atm part of A matrix using local interpolation
    A[:isst,:] = 0
    with h5py.File(analog_file,'r',swmr=True) as f:
        for i in range(N_subs_atm):
            permutation = np.argsort(analog_indices_atm[i,:])
            analog_indices_atm[i,:] = analog_indices_atm[i,permutation]
            A[:ipa1,permutation] += (w_interp_atm[i,1:,:]*np.array(f['at'][analog_indices_atm[i,:],1:,:,2])).reshape((Ne,-1)).T
            A[ipa1:ipa2,permutation] += (w_interp_atm[i,1:,:]*np.array(f['at'][analog_indices_atm[i,:],1:,:,3])).reshape((Ne,-1)).T
            A[ipa2:ihmixa,permutation] += (w_interp_atm[i,1:,:]*np.array(f['at'][analog_indices_atm[i,:],1:,:,4])).reshape((Ne,-1)).T
            A[ihmixa:iast,permutation] += (w_interp_atm[i,:,:]*np.array(f['at'][analog_indices_atm[i,:],:,:,1])).reshape((Ne,-1)).T
            A[iast:isst,permutation] += (w_interp_atm[i,:,:]*np.array(f['at'][analog_indices_atm[i,:],:,:,0])).reshape((Ne,-1)).T
        A[ihmixa:iast,:] = np.log(np.maximum(A[ihmixa:iast,:],100))

    # Get and subtract mean from columns of A, atm part only
    a_mean = np.mean(A[:isst,:],axis=1)
    for i in range(Ne):
        A[:isst,i] -= a_mean

    # rescale atm ensemble perturbations to match innovations
    A[ipa:ipa1,:] *= r_pa0/np.sqrt((Ne-1)*np.mean(A[ipa:ipa1,:]**2))
    A[ipa1:ipa2,:] *= r_pa1/np.sqrt((Ne-1)*np.mean(A[ipa1:ipa2,:]**2))
    A[ipa2:ihmixa,:] *= r_pa2/np.sqrt((Ne-1)*np.mean(A[ipa2:ihmixa,:]**2))
    A[ihmixa:iast,:] *= r_hmixa/np.sqrt((Ne-1)*np.mean(A[ihmixa:iast,:]**2))
    A[iast:isst,:] *= r_ast/np.sqrt((Ne-1)*np.mean(A[iast:isst,:]**2))

    # Update the ocn part of the A matrix
    if( k%ocn_assim_period  == 0 ):
        # Estimate forecast spread
        r_sst = rInf*np.sqrt(np.mean(innov[isst_obs:ipo_obs]**2) - obs_var_sst)
        if np.isnan(r_sst):
            r_sst = 0.5*np.sqrt(obs_var_sst)
        r_po = rInf*np.sqrt(np.mean(innov[ipo_obs:]**2) - obs_var_po)
        if np.isnan(r_po):
            r_po = 0.5*np.sqrt(obs_var_po)

        # Get local distances and analog indices
        for i in range(N_subs_ocn):
            d2_ocn[i,:] = np.tanh(np.mean((sst_mean[np.newaxis,576:960:6,(i*N_x_ocn*6):((i+1)*N_x_ocn*6):6]-catalogS[:,:,(i*N_x_ocn):((i+1)*N_x_ocn),0])**2,axis=(1,2))/clim_var_sst[i])
            d2_ocn[i,:] += np.tanh(np.mean((po_mean[np.newaxis,0,575:959:6,(i*N_x_ocn*6):((i+1)*N_x_ocn*6):6]-catalogS[:,:,(i*N_x_ocn):((i+1)*N_x_ocn),1])**2,axis=(1,2))/clim_var_po[i])
            analog_indices_ocn[i,:] = np.argsort(d2_ocn[i,:])[0:Ne]

        # Next construct using local interpolation, on coarse catalog grid
        with h5py.File(analog_file,'r',swmr=True) as f:
            # Get ocn analogs on coarse grid using local interpolation
            E_ocn[:,:,:,:] = 0
            for i in range(N_subs_ocn):
                permutation = np.argsort(analog_indices_ocn[i,:])
                analog_indices_ocn[i,:] = analog_indices_ocn[i,permutation]
                E_ocn[permutation,:,:,:] += w_interp_ocn[i,:,:][np.newaxis,:,:,np.newaxis]*np.array(f['oc'][analog_indices_ocn[i,:],:,:,:])

        # Next interpolate analogs from coarse grid back to fine grid
        for i in range(Ne):
            sst2 = E_ocn[i,:,:,0]
            po2[:,:-1,:-1] = np.moveaxis(E_ocn[i,:,:,1:],2,0)
            po2[:,-1,:] = po2[:,0,:] # Add boundaries back in
            po2[:,:,-1] = po2[:,:,0] # Add boundaries back in
            for j in range(3):
                po_A[j,:,:] = interpolate.RectBivariateSpline(np.arange(0,1537,6),np.arange(0,1537,6),po2[j,:,:],kx=3,ky=3)(np.arange(1,1536),np.arange(1,1536))
            sst = interpolate.RectBivariateSpline(np.arange(0,1536,6),np.arange(0,1536,6),sst2,kx=3,ky=3)(np.arange(0,1536),np.arange(0,1536))
            A[isst:,i] = np.concatenate((sst[:,:].flatten(),po_A[:,:,:].flatten()))

        # Get and subtract mean from columns of A, ocn part only
        a_mean = np.mean(A[isst:,:],axis=1)
        for i in range(Ne):
            A[isst:,i] -= a_mean

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
