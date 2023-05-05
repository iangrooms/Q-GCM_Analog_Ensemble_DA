"""
This script runs the whole cycled DA experiment using cAnEnOI.
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

current_path=os.getcwd()
# Set parameter values
Ne = 300 # Ensemble size
Nt = 600 # Number of cycles
N = 3*384*95 + 2*384*96 + 1536**2 + 3*1535**2 # Total number of variables.
rInf = 1 # inflation factor for ensemble perturbations

# Set up paths to executables & restart.nc files
ref_dir = "/projects/groomsi/q-gcm/examples/ref/"
mean_dir = os.environ['SLURM_SCRATCH']+'/can_weak/'

# load observation configuration
from obs_config_super_weak import *

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

# Allocate space for A.
A = np.zeros((N,Ne))

# The line below *does not* initialize the mean, it merely allocates space.
# The mean is currently initialized as just whatever is in the restart.nc file.
# If we want something else we would have to write it to the restart.nc file.
x_mean = np.mean(A,axis=1)

# Load autoencoder stuff
from load_autoencoders import ae_ast, ae_hmixa, ae_pa, ae_sst, ae_po, \
                              PS_ast, PS_hmixa, PS_pa, PS_sst, PS_po, \
                              ast_av, hmixa_av, pa_av, sst_av, po_av, \
                              ast_std, hmixa_std, pa_std, sst_std, po_std

# Define empirical scaling for rz, amplitude of noise in latent space
def rz_pa(pa_spread): # Note that this one is based on forecast spread in the middle atm layer
    return pa_spread/644

def rz_hmixa(hmixa_spread): # Note that this one is based on forecast spread for hmixa, not for log(hmixa)
    return hmixa_spread/160

def rz_ast(ast_spread):
    return ast_spread/4.3

def rz_sst(sst_spread):
    #return sst_spread/1.53 # This line is for sst forecast spread in the west-central box
    return sst_spread/0.64 # This line is for sst forecast spread in the full domain

def rz_po(po_spread): # Note that this one is based on forecast spread in the top ocn layer
    return po_spread/1.93

# Allocate some space
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

    # Get forecast & Set up for autoencoder.
    with Dataset(mean_dir+"outdata/restart.nc", "r", format="NETCDF4") as rootgrp_mean:
        pa_mean = np.array(rootgrp_mean["pa"][:,:-1,:-1]) # Load boundary val for ae
        hmixa_mean = np.log(np.array(rootgrp_mean["hmixa"][:,:]))
        ast_mean = np.array(rootgrp_mean["ast"][:,:])
        sst_mean = np.array(rootgrp_mean["sst"][:,:])
        po_mean = np.array(rootgrp_mean["po"][:,:-1,:-1]) # Load boundary vals for ae

    # Save forecast to disk
    with h5py.File(mean_dir+"mean.h5", "r+", swmr=True) as mean_out:
        mean_out["forecast/pa"][k,:,:,:] = pa_mean[:,1:,:]
        mean_out["forecast/hmixa"][k,:,:] = np.exp(hmixa_mean)
        mean_out["forecast/ast"][k,:,:] = ast_mean
        mean_out["forecast/sst"][k,:,:] = sst_mean
        mean_out["forecast/po"][k,:,:,:] = po_mean[:,1:,1:]

    # get x_mean
    x_mean = np.concatenate((pa_mean[:,1:,:].flatten(),hmixa_mean[:,:].flatten(),ast_mean[:,:].flatten(),
                             sst_mean[:,:].flatten(),po_mean[:,1:,1:].flatten()))

    # get observations
    if( k%ocn_assim_period == 0 ): # observe atm and ocn
        # Get y
        y = np.concatenate((pa_ref[:,first_obs_atm_y:95:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm].flatten(),
                            hmixa_ref[first_obs_atm_y:96:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm].flatten(),
                            ast_ref[first_obs_ast_y:96:inc_obs_ast,first_obs_ast_x:384:inc_obs_ast].flatten(),
                            sst_ref[first_obs_sst_y:last_obs_sst_y:inc_obs_sst,first_obs_sst_x:last_obs_sst_x:inc_obs_sst].flatten(),
                            po_ref[0,first_obs_po_y:1535:inc_obs_po,first_obs_po_x:1535:inc_obs_po].flatten()))
        y = y + np.sqrt(tot_obs_err_var)*np.random.standard_normal(No_tot)

        # Get Hx. Note indexing offset in pa and po because boundaries are retained
        Hx = np.concatenate((pa_mean[:,1+first_obs_atm_y:96:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm].flatten(),
                             hmixa_mean[first_obs_atm_y:96:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm].flatten(),
                             ast_mean[first_obs_ast_y:96:inc_obs_ast,first_obs_ast_x:384:inc_obs_ast].flatten(),
                             sst_mean[first_obs_sst_y:last_obs_sst_y:inc_obs_sst,first_obs_sst_x:last_obs_sst_x:inc_obs_sst].flatten(),
                             po_mean[0,1+first_obs_po_y:1536:inc_obs_po,1+first_obs_po_x:1536:inc_obs_po].flatten()))

    else: # observe just atm
        # Get y
        y = np.concatenate((pa_ref[:,first_obs_atm_y:95:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm].flatten(),
                            hmixa_ref[first_obs_atm_y:96:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm].flatten(),
                            ast_ref[first_obs_ast_y:96:inc_obs_ast,first_obs_ast_x:384:inc_obs_ast].flatten()))
        y = y + np.sqrt(atm_obs_err_var)*np.random.standard_normal(No_atm)

        # Get Hx. Note indexing offset in pa because boundaries are retained
        Hx = np.concatenate((pa_mean[:,1+first_obs_atm_y:96:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm].flatten(),
                             hmixa_mean[first_obs_atm_y:96:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm].flatten(),
                             ast_mean[first_obs_ast_y:96:inc_obs_ast,first_obs_ast_x:384:inc_obs_ast].flatten()))

    # Get innovation vector
    innov = y - Hx

    # Get estimate of forecast spread for all atm fields. Needed to set amplitude of noise in latent space.
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

    # Center & scale, encode, add noise, decode, scale & shift for each atm field. Put into A.
    # pa
    atm_cs = np.moveaxis((pa_mean - pa_av[:,np.newaxis,np.newaxis])/pa_std[:,np.newaxis,np.newaxis],0,2)
    atm_latent = ae_pa.encoder.predict(atm_cs[np.newaxis,:,:,:,np.newaxis]).astype(np.float64)
    eps = rz_pa(r_pa1)*(PS_pa@np.random.standard_normal(size=(atm_latent.shape[1],Ne))).T
    E = ae_pa.decoder.predict(atm_latent + eps)
    E = pa_std[np.newaxis,np.newaxis,np.newaxis,:]*E + pa_av[np.newaxis,np.newaxis,np.newaxis,:]
    for i in range(Ne):
        A[ipa:ihmixa,i] = np.moveaxis(E[i,1:,:,:],2,0).flatten()

    # hmixa
    atm_cs = (np.exp(hmixa_mean) - hmixa_av)/hmixa_std
    atm_latent = ae_hmixa.encoder.predict(atm_cs[np.newaxis,:,:,np.newaxis]).astype(np.float64)
    # Have to get an estimate for forecast spread of hmixa, not log(hmixa) as above.
    # This is an overestimate because can't subtract multiplicative obs error,
    # but obs error is small compared to forecast spread so the error in this approximation should be small
    innov_hmixa = np.exp(y[ihmixa_obs:iast_obs]) - np.exp(Hx[ihmixa_obs:iast_obs])
    r_hmixa_exp = rInf*np.sqrt(np.mean(innov_hmixa**2))
    eps = rz_hmixa(r_hmixa_exp)*(PS_hmixa@np.random.standard_normal(size=(atm_latent.shape[1],Ne))).T
    E = ae_hmixa.decoder.predict(atm_latent + eps)
    E = hmixa_std*E + hmixa_av
    for i in range(Ne):
        A[ihmixa:iast,i] = E[i,:,:].flatten()

    # ast
    atm_cs = (ast_mean - ast_av)/ast_std
    atm_latent = ae_ast.encoder.predict(atm_cs[np.newaxis,:,:,np.newaxis]).astype(np.float64)
    eps = rz_ast(r_ast)*(PS_ast@np.random.standard_normal(size=(atm_latent.shape[1],Ne))).T
    E = ae_ast.decoder.predict(atm_latent + eps)
    E = ast_std*E + ast_av
    for i in range(Ne):
        A[iast:isst,i] = E[i,:,:].flatten()

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
    if( k%ocn_assim_period == 0 ):
        # Estimate forecast spread
        r_sst = rInf*np.sqrt(np.mean(innov[isst_obs:ipo_obs]**2) - obs_var_sst)
        if np.isnan(r_sst):
            r_sst = 0.5*np.sqrt(obs_var_sst)
        r_po = rInf*np.sqrt(np.mean(innov[ipo_obs:]**2) - obs_var_po)
        if np.isnan(r_po):
            r_po = 0.5*np.sqrt(obs_var_po)

        # Center & scale, encode, add noise, decode, scale & shift for each ocn field, then interpolate and put into A
        # sst
        ocn_cs = (sst_mean[::6,::6] - sst_av)/sst_std
        ocn_latent = ae_sst.encoder.predict(ocn_cs[np.newaxis,:,:,np.newaxis]).astype(np.float64)
        eps = rz_sst(r_sst)*(PS_sst@np.random.standard_normal(size=(ocn_latent.shape[1],Ne))).T
        E = ae_sst.decoder.predict(ocn_latent + eps)
        E = sst_std*E + sst_av
        for i in range(Ne):
            sst2 = E[i,:,:]
            sst = interpolate.RectBivariateSpline(np.arange(0,1536,6),np.arange(0,1536,6),sst2,kx=3,ky=3)(np.arange(0,1536),np.arange(0,1536))
            A[isst:ipo,i] = sst[:,:].flatten()

        # po
        ocn_cs = np.moveaxis((po_mean[:,::6,::6] - po_av[:,np.newaxis,np.newaxis])/po_std[:,np.newaxis,np.newaxis],0,2)
        ocn_latent = ae_po.encoder.predict(ocn_cs[np.newaxis,:,:,:,np.newaxis]).astype(np.float64)
        eps = rz_po(r_po)*(PS_po@np.random.standard_normal(size=(ocn_latent.shape[1],Ne))).T
        E = ae_po.decoder.predict(ocn_latent + eps)
        E = po_std[np.newaxis,np.newaxis,np.newaxis,:]*E + po_av[np.newaxis,np.newaxis,np.newaxis,:]
        for i in range(Ne):
            po2[:,:-1,:-1] = np.moveaxis(E[i,:,:,:],2,0)
            # Average along south and west boundaries to get a constant. Double-counts the corner
            po2bvals = .5*(np.mean(po2[:,0,:],axis=1)+np.mean(po2[:,:,0],axis=1))
            po2[:,0,:] = [po2bvals[0]*np.ones(257), po2bvals[1]*np.ones(257), po2bvals[2]*np.ones(257)]
            po2[:,:,0] = po2[:,:,0]
            po2[:,-1,:] = po2[:,0,:] # Add boundaries back in
            po2[:,:,-1] = po2[:,:,0] # Add boundaries back in
            for j in range(3):
                po_A[j,:,:] = interpolate.RectBivariateSpline(np.arange(0,1537,6),np.arange(0,1537,6),po2[j,:,:],kx=3,ky=3)(np.arange(1,1536),np.arange(1,1536))
            A[ipo:,i] = po_A[:,:,:].flatten()

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
