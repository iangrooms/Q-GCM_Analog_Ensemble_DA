import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import concatenate
from tensorflow.keras import regularizers
import h5py
import numpy as np
import sys
sys.path.append('/projects/care4975/wcAE/modelpathssvds/')
sys.path.append('/projects/care4975/wcAE/architectures/')
# Get mean and std for each field
ds=2
# ast
seed = 130
filename = "/projects/care4975/wcAE/h5files/statsATast_%04d_ds%d_f.h5" % (seed,ds)
with h5py.File(filename,'r',swmr=True) as f:
    ast_std = f['at_std'][()]
    ast_av = f['at_mean'][()]
# hmixa
seed = 211
filename = "/projects/care4975/wcAE/h5files/statsAT_%04d_ds%d_f.h5" % (seed,ds)
with h5py.File(filename,'r',swmr=True) as f:
    hmixa_std = f['at_std'][()]
hmixa_av = 100
# pa
# seed = 1117
seed = 215
filename = "/projects/care4975/wcAE/h5files/statsATpa_%04d_ds%d_f.h5" % (seed,ds)
with h5py.File(filename,'r',swmr=True) as f:
    pa_std = f['at_std'][()]
    pa_av = f['at_mean'][()]
# sst
seed = 214
filename = "/projects/care4975/wcAE/h5files/statsOCsst_%04d_ds%d_f.h5" % (seed,ds)
with h5py.File(filename,'r',swmr=True) as f:
    sst_std = f['oc_std'][()]
    sst_av = f['oc_mean'][()]
# po
seed = 1111
filename = "/projects/care4975/wcAE/h5files/statsOCpo_%04d_ds%d_f.h5" % (seed,ds)
with h5py.File(filename,'r',swmr=True) as f:
    po_std = f['oc_std'][()]
    po_av = f['oc_mean'][()]

# Load autoencoders, same order
kr=regularizers.l1_l2(l1=1e-2, l2=1e-1)
br=regularizers.l1_l2(l1=1e-2, l2=1e-1)
ar=regularizers.l1_l2(l1=1e-2, l2=1e-1)
at_dimx=384
at_dimy=96
at_latent_dim = np.ceil(np.sqrt(at_dimx*at_dimy)).astype(np.int)
at_ld = np.ceil(np.sqrt(at_dimx*at_dimy)).astype(int)
beta1 = 300
arch_params = [kr, br, ar, at_dimy, at_dimx, beta1]
lw = np.array([1,1,1]);

# ast
import atast_AE2 as gAE
ae_ast = gAE.gAE(arch_params, lw)
ae_ast.load_weights('/projects/care4975/wcAE/modelpathssvds/atast_AE2044_0.013');

# hmixa
import athmixa_AE3 as gAE
ae_hmixa = gAE.gAE(arch_params, lw)
ae_hmixa.load_weights('/projects/care4975/wcAE/modelpathssvds/athmixa_AE3ld300100_0.078');

# pa
#beta1 = at_ld
beta1 = 300
arch_params = [kr, br, ar, at_dimy, at_dimx, beta1]
import atpa123_AE2 as gAE
ae_pa = gAE.gAE(arch_params, lw)
#ae_pa.load_weights('/projects/care4975/wcAE/modelpathssvds/atpa_gAE2028_0.016');
ae_pa.load_weights('/projects/care4975/wcAE/modelpathssvds/atpa_gAE2_ld300098_0.018')

# sst
oc_dimx=256
oc_dimy=256
oc_latent_dim = oc_dimx*ds
oc_ld = np.ceil(oc_dimx).astype(int)
beta3 = 400
arch_params = [kr, br, ar, oc_dimy, oc_dimx, beta3]
import ocsst_newdataAE1 as gAE
ae_sst = gAE.gAE(arch_params, lw)
ae_sst.load_weights('/projects/care4975/wcAE/modelpathssvds/ocsst_AE1newdatald400018_0.003');

# po
beta3 = 300
arch_params = [kr, br, ar, oc_dimy, oc_dimx, beta3]
import po123_AE_latentdim as gAE
ae_po = gAE.gAE(arch_params, lw)
ae_po.load_weights('/projects/care4975/wcAE/modelpathssvds/ocpo_newgAEld3000059_0.634')

# Load P and S matrices for all fields. Combine with PS = P@np.diag(S)
# ast
with h5py.File('/projects/care4975/wcAE/modelpathssvds/atast_AE2044_0.013_cov.h5','r',swmr=True) as f:
    P=f['P'][:,:]
    S=f['S'][:]
PS_ast = P@np.diag(S)

# hmixa
with h5py.File('/projects/care4975/wcAE/modelpathssvds/athmixa_AE3ld300100_0.078_cov.h5','r',swmr=True) as f:
    P=f['P'][:,:]
    S=f['S'][:]
PS_hmixa = P@np.diag(S)

#pa
#with h5py.File('/projects/care4975/wcAE/modelpathssvds/atpa_gAE2028_0.016_cov.h5','r',swmr=True) as f:
with h5py.File('/projects/care4975/wcAE/modelpathssvds/atpa_gAE2ld300_098_0.018_cov.h5','r',swmr=True) as f:
    P=f['P'][:,:]
    S=f['S'][:]
PS_pa = P@np.diag(S)

# sst
with h5py.File('/projects/care4975/wcAE/modelpathssvds/ocsst_AE1nd018_0.003_cov.h5','r',swmr=True) as f:
    P=f['P'][:,:]
    S=f['S'][:]
PS_sst = P@np.diag(S)

#po
with h5py.File('/projects/care4975/wcAE/modelpathssvds/ocpo_AEld300059_0.634_cov.h5','r',swmr=True) as f:
    P=f['P'][:,:]
    S=f['S'][:]
PS_po = P@np.diag(S)
