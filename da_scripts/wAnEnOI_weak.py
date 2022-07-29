"""
This script runs the whole cycled DA experiment using wAnEnOI.
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
Ne = 300 # Ensemble size
Nt = 600 # Number of cycles
N = 3*384*95 + 2*384*96 + 1536**2 + 3*1535**2 # Total number of variables.
rInf = 1 # inflation factor for ensemble perturbations

# Set up paths to executables & restart.nc files
ref_dir = "/projects/groomsi/q-gcm/examples/ref/"
mean_dir = os.environ['SLURM_SCRATCH']+'/wan_weak/'
analog_file = os.environ['SLURM_SCRATCH']+'/data.h5'

# load observation configuration
from obs_config_weak import *

# Set up the function from distances to weights
def get_weights(dist_squared):
    d2_max = np.max(dist_squared)
    w = (1 - dist_squared/d2_max)**2
    return w/sum(w)

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
with h5py.File(analog_file,'r',swmr=True) as f:
    catalogA = np.array(f['at'][:,1::2,::2,:]) # Subsample atmosphere to save space
    catalogS = np.array(f['oc'][:,96:160,:,:2]) # Central strip, only sst and po[top]

catalogA[:,:,:,1] = np.log(catalogA[:,:,:,1]) # transform hmixa

# The line below *does not* initialize the mean, it merely allocates space.
# The mean is currently initialized as just whatever is in the restart.nc file.
# If we want something else we would have to write it to the restart.nc file.
x_mean = np.mean(A,axis=1)

# Gets the sum of squares along the second index
# numpy's "var" and even "sum" are too memory intensive
# assumes columns are scaled by sqrt{weights}
def get_var(matrix,num_cols):
    tmp = np.zeros(matrix.shape[0])
    for i in range(num_cols):
        tmp += matrix[:,i]**2
    return tmp

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

    # find catalog distances and indices of Ne nearest neighbors, atm only
    d2 = np.tanh(np.mean((ast_mean[np.newaxis,1::2,::2]-catalogA[:,:,:,0])**2,axis=(1,2))/18) # scaled squared distances
    d2 += np.tanh(np.mean((hmixa_mean[np.newaxis,1::2,::2]-catalogA[:,:,:,1])**2,axis=(1,2))/0.02)
    d2 += np.tanh(np.mean((pa_mean[np.newaxis,0,::2,::2]-catalogA[:,:,:,2])**2,axis=(1,2))/1e5)
    d2 += np.tanh(np.mean((pa_mean[np.newaxis,1,::2,::2]-catalogA[:,:,:,3])**2,axis=(1,2))/5.3e5)
    d2 += np.tanh(np.mean((pa_mean[np.newaxis,2,::2,::2]-catalogA[:,:,:,4])**2,axis=(1,2))/2.75e5)
    analog_indices_atm = np.sort(np.argsort(d2)[0:Ne])

    # Get atm ensemble weights
    w_atm = get_weights(d2[analog_indices_atm])

    # Write ESS for atm
    with open(mean_dir+'ESS_atm.txt','a') as fE:
        fE.write(str(1/np.sum(w_atm**2)))
        fE.write('\n')

    # Get atm part of A matrix.
    with h5py.File(analog_file,'r',swmr=True) as f:
        for i in range(Ne):
            atm5 = np.array(f['at'][analog_indices_atm[i],:,:,:])
            ast = atm5[:,:,0]
            hmixa = np.log(atm5[:,:,1])
            pa = np.moveaxis(atm5[1:,:,2:],2,0)
            # Now put the data into an ensemble matrix. We will flatten each array using row-major ordering
            # with the order pa,hmixa,ast,sst,po
            A[:isst,i] = np.concatenate((pa[:,:,:].flatten(),hmixa[:,:].flatten(),ast[:,:].flatten()))

    # Get mean from columns of A, atm part only
    a_mean = np.zeros(N)
    for i in range(Ne):
        a_mean[:isst] += w_atm[i]*A[:isst,i]

    # Subtract mean from columns of A, then scale using weights. Atm part only.
    for i in range(Ne):
        A[:isst,i] -= a_mean[:isst]
        A[:isst,i] *= np.sqrt(w_atm[i])

    # rescale atm ensemble perturbations to match innovations
    A[ipa:ipa1,:] *= r_pa0/np.sqrt(np.mean(get_var(A[ipa:ipa1,:],Ne)))
    A[ipa1:ipa2,:] *= r_pa1/np.sqrt(np.mean(get_var(A[ipa1:ipa2,:],Ne)))
    A[ipa2:ihmixa,:] *= r_pa2/np.sqrt(np.mean(get_var(A[ipa2:ihmixa,:],Ne)))
    A[ihmixa:iast,:] *= r_hmixa/np.sqrt(np.mean(get_var(A[ihmixa:iast,:],Ne)))
    A[iast:isst,:] *= r_ast/np.sqrt(np.mean(get_var(A[iast:isst,:],Ne)))

    # Update the ocn part of the A matrix
    if( k%ocn_assim_period  == 0 ):
        # Estimate forecast spread
        r_sst = rInf*np.sqrt(np.mean(innov[isst_obs:ipo_obs]**2) - obs_var_sst)
        if np.isnan(r_sst):
            r_sst = 0.5*np.sqrt(obs_var_sst)
        r_po = rInf*np.sqrt(np.mean(innov[ipo_obs:]**2) - obs_var_po)
        if np.isnan(r_po):
            r_po = 0.5*np.sqrt(obs_var_po)

        # Get ocn analog indices
        d2 = np.tanh(np.mean((sst_mean[np.newaxis,576:960:6,0::6]-catalogS[:,:,:,0])**2,axis=(1,2))/0.85)
        d2 += np.tanh(np.mean((po_mean[np.newaxis,0,575:959:6,5::6]-catalogS[:,:,1:,1])**2,axis=(1,2))/4.75)
        analog_indices_ocn = np.sort(np.argsort(d2)[0:Ne]) # indices of Ne analogs

        # Get ocn ensemble weights
        w_ocn = get_weights(d2[analog_indices_ocn])

        # Write ESS for ocn
        with open(mean_dir+'ESS_ocn.txt','a') as fE:
            fE.write(str(1/np.sum(w_ocn**2)))
            fE.write('\n')

        # Load ocn analogs into A matrix
        with h5py.File(analog_file,'r',swmr=True) as f:
            for i in range(Ne):
                ocn4 = np.array(f['oc'][analog_indices_ocn[i],:,:,:])
                sst2 = ocn4[:,:,0]
                po2[:,:-1,:-1] = np.moveaxis(ocn4[:,:,1:],2,0)
                po2[:,-1,:] = po2[:,0,:] # Add boundaries back in
                po2[:,:,-1] = po2[:,:,0] # Add boundaries back in
                for j in range(3):
                    po_A[j,:,:] = interpolate.RectBivariateSpline(np.arange(0,1537,6),np.arange(0,1537,6),po2[j,:,:],kx=3,ky=3)(np.arange(1,1536),np.arange(1,1536))
                sst = interpolate.RectBivariateSpline(np.arange(0,1536,6),np.arange(0,1536,6),sst2,kx=3,ky=3)(np.arange(0,1536),np.arange(0,1536))
                # Now put the data into an ensemble matrix. We will flatten each array using row-major ordering
                # with the order pa,hmixa,ast,sst,po
                A[isst:,i] = np.concatenate((sst[:,:].flatten(),po_A[:,:,:].flatten()))

        # Get mean from columns of A, ocn part only
        for i in range(Ne):
            a_mean[isst:] += w_ocn[i]*A[isst:,i]

        # Subtract mean from columns of A, then scale using weights. Ocn part only.
        for i in range(Ne):
            A[isst:,i] -= a_mean[isst:]
            A[isst:,i] *= np.sqrt(w_ocn[i])

        # rescale ocn ensemble perturbations to match innovations
        if is_sst_box:
            A[isst:ipo,:] *= r_sst/np.sqrt(np.mean(get_var(A[isst_box,:],Ne)))
        else:
            A[isst:ipo,:] *= r_sst/np.sqrt(np.mean(get_var(A[isst:ipo,:],Ne)))
        A[ipo:,:] *= r_po/np.sqrt(np.mean(get_var(A[ipo:,:],Ne)))

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
