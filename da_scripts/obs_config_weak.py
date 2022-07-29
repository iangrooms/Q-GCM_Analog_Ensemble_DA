from scipy import sparse
import numpy as np

# Are we observing sst just in the west/central box
is_sst_box = True

# The localization below has inc_obs_ast = 4, inc_obs_atm = 8, inc_obs_sst = 12, inc_obs_po = 24. Sst only observed in the west/central box.
loc_file = "/projects/zost7833/q-gcm_multivariate_localization/DA/Saved_Localization_Matrices/MVGC_DenseAstSst_La1280_Lo240/"

# Set up localization - load localization matrices
# super weak means all cross-variable correlations are zeroed out except among po[0:3]
loc_HLHT_atm = sparse.load_npz(loc_file+"atob_atob.npz").toarray()
loc_HLHT_tot = sparse.load_npz(loc_file+"totob_totob_weak.npz").toarray()
loc_LHT_atm = sparse.load_npz(loc_file+"totall_atob_weak.npz") # CSC format
loc_LHT_tot = sparse.load_npz(loc_file+"totall_totob_weak.npz") # CSC format

# Set observation grid in space and time
ocn_assim_period = 2 # Period of ocn assimilation [days]
inc_obs_atm = 8 # spacing between atmospheric observations, except ast
first_obs_atm_x = 2 # First x index of atm obs, except ast
first_obs_atm_y = 6 # First y index of atm obs, except ast
inc_obs_ast = 4 # spacing between ast observations
first_obs_ast_x = 2 # First x index of ast obs
first_obs_ast_y = 2 # First y index of ast obs
inc_obs_sst = 12 # spacing between sst observations
first_obs_sst_x = 8 # First x index of ast obs
first_obs_sst_y = 522 # First y index of ast obs
last_obs_sst_x = 512 # Last x index of ast obs in np.arange(first,last,inc)
last_obs_sst_y = 1024 # Last y index of ast obs in np.arange(first,last,inc)
inc_obs_po = 24 # spacing between po observations
first_obs_po_x = 8 # First x index of po obs
first_obs_po_y = 18 # First y index of po obs

# Set obs error variances
obs_var_pa = 40000 # error variance for obs of pa
obs_var_hmixa = .001 # error variance for obs of ln(hmixa)
obs_var_ast = 2.25 # error variance for obs of ast
obs_var_sst = 0.0625 # error variance for obs of sst
obs_var_po = 0.25 # error variance for obs of po

# Set first linear index in the A matrix for each field
ipa = 0 # first index for pa
ipa1 = 95*384
ipa2 = 2*95*384
ihmixa = 3*95*384 # first index for hmixa
iast = ihmixa + 96*384 # first index for ast
isst = iast + 96*384 # first index for sst
ipo = isst + 1536**2 # first index for po
sst_box_ind = np.mgrid[512:1024,0:512]
isst_box = isst + np.ravel_multi_index(np.array([sst_box_ind[0,:,:].flatten(),sst_box_ind[1,:,:].flatten()]),(1536,1536))
del sst_box_ind

# Set up observations indices. To do this we need to know the map between indices of X and variables on the grid.
pa_obs_ind = np.mgrid[0:3,first_obs_atm_y:95:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm]
ind_obs_pa = ipa + np.ravel_multi_index(np.array([pa_obs_ind[0,:,:,:].flatten(),pa_obs_ind[1,:,:,:].flatten(),pa_obs_ind[2,:,:,:].flatten()]),(3,95,384))
hmixa_obs_ind = np.mgrid[first_obs_atm_y:96:inc_obs_atm,first_obs_atm_x:384:inc_obs_atm]
ind_obs_hmixa = ihmixa + np.ravel_multi_index(np.array([hmixa_obs_ind[0,:,:].flatten(),hmixa_obs_ind[1,:,:].flatten()]),(96,384))
ast_obs_ind = np.mgrid[first_obs_ast_y:96:inc_obs_ast,first_obs_ast_x:384:inc_obs_ast]
ind_obs_ast = iast + np.ravel_multi_index(np.array([ast_obs_ind[0,:,:].flatten(),ast_obs_ind[1,:,:].flatten()]),(96,384))
sst_obs_ind = np.mgrid[first_obs_sst_y:last_obs_sst_y:inc_obs_sst,first_obs_sst_x:last_obs_sst_x:inc_obs_sst]
ind_obs_sst = isst + np.ravel_multi_index(np.array([sst_obs_ind[0,:,:].flatten(),sst_obs_ind[1,:,:].flatten()]),(1536,1536))
po_obs_ind = np.mgrid[0:1,first_obs_po_y:1535:inc_obs_po,first_obs_po_x:1535:inc_obs_po]
ind_obs_po = ipo + np.ravel_multi_index(np.array([po_obs_ind[0,:,:,:].flatten(),po_obs_ind[1,:,:,:].flatten(),po_obs_ind[2,:,:,:].flatten()]),(3,1535,1535))
ind_obs_atm = np.concatenate((ind_obs_pa,ind_obs_hmixa,ind_obs_ast))
ind_obs_tot = np.concatenate((ind_obs_atm,ind_obs_sst,ind_obs_po))
No_atm = ind_obs_atm.shape[0] # number of atm observations in one cycle
No_tot = ind_obs_tot.shape[0] # number of atm+ocn=tot observations in one cycle
ipa_obs = 0 # index of first pa observation in the y and Hx vectors
ipa1_obs = ind_obs_pa.shape[0]//3
ipa2_obs = 2*ind_obs_pa.shape[0]//3
ihmixa_obs = ipa_obs + ind_obs_pa.shape[0] # index of first hmixa observation in the y and Hx vectors
iast_obs = ihmixa_obs + ind_obs_hmixa.shape[0] # index of first ast observation in the y and Hx vectors
isst_obs = iast_obs + ind_obs_ast.shape[0] # index of first sst observation in the y and Hx vectors
ipo_obs = isst_obs + ind_obs_sst.shape[0] # index of first po observation in the y and Hx vectors

# Set up R matrices
atm_obs_err_var = np.concatenate((obs_var_pa*np.ones(ind_obs_pa.shape[0]),
                                  obs_var_hmixa*np.ones(ind_obs_hmixa.shape[0]),
                                  obs_var_ast*np.ones(ind_obs_ast.shape[0])))
R_atm = np.diag(atm_obs_err_var)
tot_obs_err_var = np.concatenate((atm_obs_err_var,
                                  obs_var_sst*np.ones(ind_obs_sst.shape[0]),
                                  obs_var_po*np.ones(ind_obs_po.shape[0])))
R_tot = np.diag(tot_obs_err_var)
del pa_obs_ind, ind_obs_pa, hmixa_obs_ind, ind_obs_hmixa, ast_obs_ind, ind_obs_ast, sst_obs_ind, ind_obs_sst, po_obs_ind, ind_obs_po
