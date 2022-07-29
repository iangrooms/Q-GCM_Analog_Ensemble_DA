# This script creates and saves multivariate Gaspari-Cohn localization matrices
# with La = 1500 km and Lo = 500 km
# Localization matrices are named with the following naming convention:
# at, oc, tot refer to the fields (atmosphere, ocean, both)
# ob, all refer to the points within the field that are included (observations only, all points)
# The first descriptor describes the rows, the second the columns
# e.g. loc_atall_ocob has rows: all atmospheric locations and columns: oceanic observation locations
import numpy as np
from scipy import sparse
from define_grid import *

#############
# Functions #
#############

def calculate_periodic_distance(x1, x2, y1, y2, nx):
    ''' Calculates distance between two points in a periodic (in x) channel
    INPUT:
    x1, x2, y1, y2 - coordinates
    nx - length of channel
    OUTPUT
    distance = sqrt( (x1-x2)^2 + (y2-y1)^2 ) where x1-x2 acccounts for periodicity in x
    '''
    dx = np.minimum(np.abs(x1-x2), nx-np.abs(x1-x2))
    dy = y1-y2
    distance = np.sqrt(dx**2 + dy**2)
    return distance

def create_distance_matrix(grdx1, grdx2, grdy1, grdy2, nx):
    ''' Creates matrix of distances between pairs of points in R^2
    INPUT:
    grdx1, grdy1 - a set of coordinates
    grdx2, grdy2 - another set of coordinates
    nx - length of channel
    OUTPUT
    dist - matrix storign all pairwise distances
    '''
    length1 = grdx1.shape[0]
    length2 = grdx2.shape[0]
    assert grdx1.shape[0] == grdy1.shape[0], "(x,y) coordinates should come in pairs, but length of x1 is not the same as length of y1."
    assert grdx2.shape[0] == grdy2.shape[0], "(x,y) coordinates should come in pairs, but length of x2 is not the same as length of y2."
    dist = np.zeros([length1, length2])
    for ii in range(length1):
        d = calculate_periodic_distance(grdx1[ii], grdx2, grdy1[ii], grdy2, nx)
        dist[ii, :] = d
    return dist

def gaspari_cohn_univariate(distance, localization_half_width):
    """ Fifth-order piecewise rational localization function from Gaspari & Cohn (1999)
    INPUT:
    distance - one or more distances where we will calculate localization weights
    localization_half_width = localization radius / 2
    OUTPUT:
    localization weights
    """
    x = np.abs(distance)/localization_half_width
    localization_weights = np.zeros(x.shape)
    # cases
    leq1 = x<=1
    leq2 = np.logical_and(x>1, x<=2)
    # evaluation of fifth order piecewise rational function
    localization_weights[leq1] = -.25 * x[leq1]**5 + .5 * x[leq1]**4 + .625 * x[leq1]**3 - (5/3) * x[leq1]**2 + 1
    localization_weights[leq2] = (1/12) * x[leq2]**5 - .5 * x[leq2]**4 + .625 * x[leq2]**3 + (5/3) * x[leq2]**2 - 5 * x[leq2] + 4 - 2/(3*x[leq2])
    return localization_weights

def gaspari_cohn_cross(distance, localization_half_width1, localization_half_width2):
    """ Cross-localization function from Stanley, Grooms and Kleiber (2021)
    INPUT:
    distance - distance where we will calculate cross-localization weights
    localization_half_width1 = (1/2) * localization radius for process 1
    localization_half_width2 = (1/2) * localization radius for process 2
    OUTPUT:
    cross-localization weights
    """
    c_min = min(localization_half_width1, localization_half_width2)
    c_max = max(localization_half_width1, localization_half_width2)
    kappa = np.sqrt(c_max/c_min)
    distance = np.abs(distance)
    x = distance/(kappa*c_min)
    localization_weights = np.zeros(x.shape)
    # Two cases: kappa >= sqrt(2) and kappa < sqrt(2)
    # case 1: kappa >= sqrt(2)
    if kappa >= np.sqrt(2):
        # four cases
        leq_min = distance <= c_min
        leq_dif = np.logical_and(c_min < distance, distance <= c_max-c_min)
        leq_max = np.logical_and(c_max - c_min < distance, distance <= c_max )
        leq_sum = np.logical_and(c_max < distance, distance <= c_min + c_max )
        # evaluation of fifth order piecewise rational function
        localization_weights[leq_min] = -(1/6)*x[leq_min]**5 + (1/2)*(1/kappa)*x[leq_min]**4 -(5/3)*(kappa**-3)*x[leq_min]**2 + (5/2)*(kappa**-3) - (3/2)*(kappa**-5)
        localization_weights[leq_dif] = -(5/2)*(kappa**-4)*x[leq_dif] - ((1/3)*(kappa**-6))/x[leq_dif] + (5/2)*(kappa**-3)
        localization_weights[leq_max] = ( -(1/12)*x[leq_max]**5 + (1/4)*(kappa-1/kappa)*x[leq_max]**4 + (5/8)*x[leq_max]**3
            - (5/6)*(kappa**3-kappa**-3)*x[leq_max]**2 +(5/4)*(kappa**4-kappa**2-kappa**-2-kappa**-4)*x[leq_max]
            + (5/12)/x[leq_max] - (3/8)*(kappa**4+kappa**-4)/x[leq_max] + (1/6)*(kappa**6-kappa**-6)/x[leq_max]
            + (5/4)*(kappa**3+kappa**-3) - (3/4)*(kappa**5-kappa**-5) )
        localization_weights[leq_sum] = ( (1/12)*x[leq_sum]**5 - (1/4)*(kappa+1/kappa)*x[leq_sum]**4 + (5/8)*x[leq_sum]**3
            + (5/6)*(kappa**3+kappa**-3)*x[leq_sum]**2 - (5/4)*(kappa**4+kappa**2+kappa**-2+kappa**-4)*x[leq_sum]
            + (5/12)/x[leq_sum] - (3/8)*(kappa**4+kappa**-4)/x[leq_sum] - (1/6)*(kappa**6+kappa**-6)/x[leq_sum]
            + (5/4)*(kappa**3+kappa**-3) + (3/4)*(kappa**5+kappa**-5) )
    return localization_weights

def create_sparse_univariate_localization_matrix(grdx1, grdx2, grdy1, grdy2, nnz, loc_rad):
    ''' Creates sparse matrix of Gaspari-Cohn localization weights between pairs of points in R^2
    INPUT:
    grdx1, grdy1 - a set of coordinates
    grdx2, grdy2 - another set of coordinates
    nnz - number of nonzero entries in the matrix (must be known ahead of time)
    loc_rad - localization radius
    OUTPUT
    loc_matrix - matrix storing localization weights
    '''
    here = 0
    rows = np.empty([nnz])
    cols = np.empty([nnz])
    data = np.empty([nnz])
    inds2 = np.arange(0, grdx2.shape[0])
    # Loop over first grid (corresponding to columns in loc mat)
    for col_ind in range(grdx1.shape[0]):
        # Get (x1, y1) coordinate pair
        x1 = grdx1[col_ind]
        y1 = grdy1[col_ind]
        # In second grid (corresponding to rows in loc mat), which points are close to (x1, y1)?
        which_xs_are_near = np.minimum(np.abs(grdx2 - x1), dxa*nxta - np.abs(grdx2 - x1)) < loc_rad
        which_ys_are_near = np.abs(grdy2 - y1) < loc_rad
        row_inds_near = inds2[np.logical_and(which_xs_are_near, which_ys_are_near)]
        x2 = grdx2[row_inds_near]
        y2 = grdy2[row_inds_near]
        # Calculate distance and reevaluate which points are close enough to (x1, y1)
        d_near = calculate_periodic_distance(x1, x2, y1, y2, dxa*(nxta))
        row_inds_nonzero = row_inds_near[d_near<loc_rad]
        d_nonzero = d_near[d_near<loc_rad]
        # Calculate and save localization weights
        loc = gaspari_cohn_univariate(d_nonzero, loc_rad//2)
        add_length = len(loc)
        rows[here:here+add_length] = row_inds_nonzero
        cols[here:here+add_length] = np.repeat(col_ind, loc.shape[0])
        data[here:here+add_length] = loc
        here = here+add_length
    loc_matrix = sparse.csc_matrix((data, (rows, cols)), shape=(grdx2.shape[0], grdx1.shape[0]))
    return loc_matrix

def create_sparse_cross_localization_matrix(grdx1, grdx2, grdy1, grdy2, nnz, loc_rad1, loc_rad2):
    ''' Creates sparse matrix of Gaspari-Cohn cross-localization weights between pairs of points in R^2
    INPUT:
    grdx1, grdy1 - a set of coordinates
    grdx2, grdy2 - another set of coordinates
    nnz - number of nonzero entries in the matrix (must be known ahead of time)
    loc_rad1 - localization radius associated with grdx1, grdy1
    loc_rad2 - localization radius associated with grdx2, grdy2
    OUTPUT
    loc_matrix - matrix storing cross-localization weights
    '''
    here = 0
    rows = np.empty([nnz])
    cols = np.empty([nnz])
    data = np.empty([nnz])
    inds2 = np.arange(0, grdx2.shape[0])
    # Loop over first grid (corresponding to columns in loc mat)
    for col_ind in range(grdx1.shape[0]):
        # Get (x1, y1) coordinate pair
        x1 = grdx1[col_ind]
        y1 = grdy1[col_ind]
        # In second grid (corresponding to rows in loc mat), which points are close to (x1, y1)?
        which_xs_are_near = np.minimum(np.abs(grdx2 - x1), dxa*nxta - np.abs(grdx2 - x1)) < (loc_rad1+loc_rad2)/2
        which_ys_are_near = np.abs(grdy2 - y1) < (loc_rad1+loc_rad2)/2
        row_inds_near = inds2[np.logical_and(which_xs_are_near, which_ys_are_near)]
        x2 = grdx2[row_inds_near]
        y2 = grdy2[row_inds_near]
        # Calculate distance and reevaluate which points are close enough to (x1, y1)
        d_near = calculate_periodic_distance(x1, x2, y1, y2, dxa*(nxta))
        row_inds_nonzero = row_inds_near[d_near<(loc_rad1+loc_rad2)/2]
        d_nonzero = d_near[d_near<(loc_rad1+loc_rad2)/2]
        # Calculate and save localization weights
        loc = gaspari_cohn_cross(d_nonzero, loc_rad1//2, loc_rad2//2)
        add_length = len(loc)
        cols[here:here+add_length] = np.repeat(col_ind, loc.shape[0])
        rows[here:here+add_length] = row_inds_nonzero
        data[here:here+add_length] = loc
        here = here+add_length
    loc_matrix = sparse.csc_matrix((data, (rows, cols)), shape=(grdx2.shape[0], grdx1.shape[0]))
    return loc_matrix

assert calculate_periodic_distance(0, 9, 0, 0, 10) == 1
assert gaspari_cohn_univariate(0, 10) == 1
assert gaspari_cohn_univariate(15.01, 7.5) == 0
assert gaspari_cohn_cross(30.01, 7.5, 22.5) == 0

#######################################
# Create localization matrices, HLH^T #
#######################################

# Set variables
La = 1500 # atmospheric localization radius (km)
Lo = 500 # oceanic localization radius (km)
savepath = '/projects/zost7833/q-gcm_multivariate_localization/DA/MVGC_La1500_Lo500/'

dist_atob_atob = create_distance_matrix(grdx_atm_obs, grdx_atm_obs, grdy_atm_obs, grdy_atm_obs, dxa*nxta)
dist_atob_ocob = create_distance_matrix(grdx_atm_obs, grdx_ocn_obs, grdy_atm_obs, grdy_ocn_obs, dxa*nxta)
dist_ocob_ocob = create_distance_matrix(grdx_ocn_obs, grdx_ocn_obs, grdy_ocn_obs, grdy_ocn_obs, dxa*nxta)
loc_atob_atob = gaspari_cohn_univariate(dist_atob_atob, La//2)
loc_atob_atob = sparse.coo_matrix(loc_atob_atob)
loc_atob_ocob = gaspari_cohn_cross(dist_atob_ocob, La//2, Lo//2)
loc_atob_ocob = sparse.coo_matrix(loc_atob_ocob)
loc_ocob_ocob = gaspari_cohn_univariate(dist_ocob_ocob, Lo//2)
loc_ocob_ocob = sparse.coo_matrix(loc_ocob_ocob)
loc_totob_totob = sparse.bmat([[loc_atob_atob, loc_atob_ocob], [loc_atob_ocob.transpose(), loc_ocob_ocob]], format='csc')

# Save matrices
sparse.save_npz(savepath+'MV_GC_La_1500_Lo_500_atob_atob', loc_atob_atob)
sparse.save_npz(savepath+'MV_GC_La_1500_Lo_500_atob_ocob', loc_atob_ocob)
sparse.save_npz(savepath+'MV_GC_La_1500_Lo_500_ocob_ocob', loc_ocob_ocob)
sparse.save_npz(savepath+'MV_GC_La_1500_Lo_500_totob_totob', loc_totob_totob)

# Check that these are symmetric matrices
assert (abs(loc_atob_atob - loc_atob_atob.transpose())>1e-10).nnz == 0, 'atm obs localization matrix should be symmetric'
assert (abs(loc_ocob_ocob - loc_ocob_ocob.transpose())>1e-10).nnz == 0, 'ocn obs localization matrix should be symmetric'
assert (abs(loc_totob_totob - loc_totob_totob.transpose())>1e-10).nnz == 0, 'total obs localization matrix should be symmetric'

#######################################
# Create localization matrices, LH^T #
#######################################

# As currently written the code needs to know the number
# of non-zero elements in each of the submatrices ahead of time.
# There is currently no good way to calculate these numbers.
# For La=1500km and Lo=500 km the numbers are listed below:
nnz_atall_atob = 3661296 # atm obs - atm all
nnz_atall_ocob = 21256897 # ocn obs - atm all
nnz_ocall_atob = 86664487 # atm obs - ocn all
nnz_ocall_ocob = 1094631808 # ocn obs - ocn all

# Calculate component blocks
loc_atall_atob = create_sparse_univariate_localization_matrix(grdx_atm_obs, grdx_atm, grdy_atm_obs, grdy_atm, nnz_atall_atob, La)
loc_atall_ocob = create_sparse_cross_localization_matrix(grdx_ocn_obs, grdx_atm, grdy_ocn_obs, grdy_atm, nnz_atall_ocob, La, Lo)
loc_ocall_atob = create_sparse_cross_localization_matrix(grdx_atm_obs, grdx_ocn, grdy_atm_obs, grdy_ocn, nnz_ocall_atob, La, Lo)
loc_ocall_ocob = create_sparse_univariate_localization_matrix(grdx_ocn_obs, grdx_ocn, grdy_ocn_obs, grdy_ocn, nnz_ocall_ocob, Lo)

# Put blocks together to form LH^T
loc_totall_totob = sparse.bmat([[loc_atall_atob, loc_atall_ocob], [loc_ocall_atob, loc_ocall_ocob]], format='csc')
loc_totall_atob = sparse.bmat([[loc_atall_atob], [loc_ocall_atob]], format='csc')

# Save matrices
sparse.save_npz(savepath+'MV_GC_La_1500_Lo_500_atall_atob', loc_atall_atob)
sparse.save_npz(savepath+'MV_GC_La_1500_Lo_500_atall_ocob', loc_atall_ocob)
sparse.save_npz(savepath+'MV_GC_La_1500_Lo_500_ocall_atob', loc_ocall_atob)
sparse.save_npz(savepath+'MV_GC_La_1500_Lo_500_ocall_ocob', loc_ocall_ocob)
sparse.save_npz(savepath+'MV_GC_La_1500_Lo_500_totall_totob', loc_totall_totob)
sparse.save_npz(savepath+'MV_GC_La_1500_Lo_500_totall_atob', loc_totall_atob)

# Check that this is consistent with previous calculations
assert np.all(loc_atall_atob[ind_obs_atm,:].todense() == loc_atob_atob.todense()), 'The two methods of calculating the localization matrices produce inconsistent results.'
assert np.all(loc_atall_ocob[ind_obs_atm,:].todense() == loc_atob_ocob.todense()), 'The two methods of calculating the localization matrices produce inconsistent results.'
assert np.all(loc_ocall_atob[ind_obs_ocn-isst,:].todense() == loc_atob_ocob.todense().transpose()), 'The two methods of calculating the localization matrices produce inconsistent results.'
assert np.all(loc_ocall_ocob[ind_obs_ocn-isst,:].todense() == loc_ocob_ocob.todense()), 'The two methods of calculating the localization matrices produce inconsistent results.'
assert np.all(loc_totall_totob[ind_obs_tot,:].todense() == loc_totob_totob.todense()), 'The two methods of calculating the localization matrices produce inconsistent results.'
