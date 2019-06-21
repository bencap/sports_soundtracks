# Ben Capodanno
# analysis.py
# This file analyzes data objects

import numpy as np
import math
import data
import random
import scipy.cluster.vq as vq
import scipy.stats

# A function that takes in a list of string/ints and converts them to the
# appropriate index for the data matrix
def convert_indices( data, cols ):
    indices = []

    for head in cols:
        if isinstance( head, int ):
            indices.append( head )
            continue
        else:
            try:
                indices.append( data.header2col[ head ] )
            except IndexError:
                print( "No column (" + head + ") in dataset." )

    return indices

# returns list of pairs defining the min/max of a column for all columns in list passed
def data_range( data, cols ):
    rtrnList = []
    indices = convert_indices( data, cols )

    mins = data.data[ :, indices ].min( 0 )
    max = data.data[ :, indices].max( 0 )

    for i in range( len( indices ) ):
        tmp = []
        tmp.append( mins[0,i] )
        tmp.append( max[0,i] )
        rtrnList.append( tmp )

    return rtrnList

def data_iqr( data, cols ):
    rtrnList = []
    indices = convert_indices( data, cols )

    quarter = np.percentile( data.data[ :, indices], axis = 0, q = ( 25) ).tolist()
    threequarter = np.percentile( data.data[ :, indices], axis = 0, q = ( 75 ) ).tolist()

    for i in range( len( indices ) ):
        tmp = []
        tmp.append( quarter[i] )
        tmp.append( threequarter[i] )
        rtrnList.append( tmp )

    return rtrnList

# returns a list of means for the columns passed in
def data_mean( data, cols ):
    indices = convert_indices( data, cols )
    return data.data[ :, indices ].mean( 0 ).tolist()[0]

def data_median( data, cols ):
    indices = convert_indices( data, cols )
    return np.median( data.data[ :, indices ], axis = 0 ).tolist()[0]

# returns a list of stdev(sample) for the columns passed in
def data_stdev( data, cols ):
    indices = convert_indices( data, cols )
    return np.std( data.data[ :, indices], axis = 0, ddof = 1).tolist()[0]

def data_variance( data, cols ):
    indices = convert_indices( data, cols )
    return np.var( data.data[ :, indices], axis = 0 ).tolist()[0]

# returns a matrix with the specified columns normalized
def normalize_columns_separately( data, cols ):
    data.data = np.matrix( data.data )
    indices = convert_indices( data, cols )
    origin = ( data.data[ :, indices[0] ] - data.data[ :, indices[0] ].min( 0 ) ) / ( data.data[ :, indices[0] ].max( 0 ) - data.data[ :, indices[0] ].min( 0 ) )
    for ind in range( 1, len( indices ) ):
        curr = ( data.data[ :, indices[ind] ] - data.data[ :, indices[ind] ].min( 0 ) ) / ( data.data[ :, indices[ind] ].max( 0 ) - data.data[ :, indices[ind] ].min( 0 ) )
        origin = np.hstack( ( origin, curr ) )
    return origin

# returns a matrix fully normalized
def normalize_columns_together( data, cols ):
    indices = convert_indices( data, cols )
    return ( data.data[ :, indices] - data.data.min() ) / ( data.data.max() - data.data.min() )

# returns a linear regression equation
# tuple = ( slope, intercept, r, p, std error, ( x var min, x var max), (y var min, y var max) )
def single_linear_regression( data, ind_var, dep_var ):
    indices = convert_indices( data, [dep_var, ind_var] )
    dep_rg = data_range( data, [indices[0]] )
    dep_min, dep_max = dep_rg[0][0], dep_rg[0][1]
    ind_rg = data_range( data, [indices[1]] )
    ind_min, ind_max = ind_rg[0][0], ind_rg[0][1]

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress( np.hstack( ( data.data[ :, indices[0] ], data.data[ :, indices[1] ] ) ) )
    return ( slope, intercept, r_value, p_value, std_err, ( dep_min, dep_max ) , ( ind_min, ind_max ) )

def linear_regression(data, ind, dep):
    # set up matrices for regression
    idx_y = convert_indices( data, [ind] )
    y = data.data[ :, idx_y ]
    idx_x = convert_indices( data, dep )
    A = data.data[ :, idx_x ]
    A = np.hstack( ( A, np.ones( ( data.get_num_points(), 1 ) ) ) )

    AAinv = np.linalg.inv( np.dot( A.T, A ) )
    x = np.linalg.lstsq( A, y, rcond=None )

    # calculate fit and size statistics
    b = x[0]
    n = np.size( y, 0 )
    c = np.size( b, 0 )
    df_e = n - c
    df_r = c - 1

    # calculate error statistics
    error = y - np.dot( A, b )
    sse = np.dot( error.T, error ) / df_e
    stderr = np.sqrt( np.diagonal( sse[0, 0] * AAinv ) )

    # calculate test statistics
    t = b.T / stderr
    p = 2*(1 - scipy.stats.t.cdf(abs(t), df_e))
    r2 = 1 - error.var() / y.var()

    return ( b, sse, r2, t, p )

# Calculating PCA w/ SVD
def pca(d, headers, normalize=True):
    # assign to A the desired data. Use either normalize_columns_separately
    #   or get_data, depending on the value of the normalize argument.
    if normalize:
        A = normalize_columns_separately( d, headers )
    else:
        A = d.subset( headers )

    # assign to m the mean values of the columns of A
    m = np.mean( A, 0 )

    # assign to D the difference matrix A - m
    D = A - m

    # assign to U, S, V the result of running np.svd on D, with full_matrices=False
    U, S, V = np.linalg.svd( D, full_matrices = False)

    # the eigenvalues of cov(A) are the squares of the singular values (S matrix)
    #   divided by the degrees of freedom (N-1). The values are sorted.
    eigenvalues = np.square( S ) / ( np.size( D, 0 ) - 1 )

    # project the data onto the eigenvectors. Treat V as a transformation
    #   matrix and right-multiply it by D transpose. The eigenvectors of A
    #   are the rows of V. The eigenvectors match the order of the eigenvalues.
    proj_data = V * D.T

    # create and return a PCA data object with the headers, projected data,
    # eigenvectors, eigenvalues, and mean vector.
    return data.PCAData( proj_data.T, V, eigenvalues, m, headers )

def kmeans_numpy( d, headers, K, whiten = True ):
    '''Takes in a Data object, a set of headers, and the number of clusters to create
    Computes and returns the codebook, codes, and representation error.
    '''

    # assign to A the result of getting the data from your Data object
    A = d.data
    # assign to W the result of calling vq.whiten on A
    if whiten: W = vq.whiten( A )
    else: W = A

    # assign to codebook, bookerror the result of calling vq.kmeans with W and K
    codebook, bookerror = vq.kmeans( W, K )
    # assign to codes, error the result of calling vq.vq with W and the codebook
    codes, error = vq.vq( W, codebook )

    return codebook, codes, error

def kmeans_init( A, K ):
    if A.size < K:
        print( "Size less than K, returning int = -9999" )
        return -9999

    indices = random.sample( range( A.size ), K )
    mean_list = np.zeros( ( K, np.size( A, 1 ) ) )
    for idx in range( len( indices ) ):
        mean_list[idx] = A[idx]

    return np.matrix( mean_list )

def row_mean( row ):
    sum = 0
    row = np.matrix( row )
    for i in range( np.size( row, 1 ) ):
        sum += row[0,i]

    return sum / np.size( row, 1 )

# Given a data matrix A and a set of means in the codebook
# Returns a matrix of the id of the closest mean to each point
# Returns a matrix of the sum-squared distance between the closest mean and each point
def kmeans_classify( A, codebook ):
    # calculate differences between points and data means
    diff = []
    for i in range( np.size( A, 0 ) ):
        diff.append( codebook - A[i,:] )

    # calculate min sse in means for each data point
    min_idx = []
    min_val = []

    # calculate square of all points
    ss = np.square( diff )

    # calculate sum of squares for all points
    # ie for each matrix, create a sub list, then loop over each row. Get the
    # sum of each element in a row and append that list of sums to an intermediate list
    intermediate = []
    for mat in ss:
        sub = []
        for i in range( np.size( mat, 0 ) ):
            sub.append( mat[i].sum() )
        intermediate.append( sub )

    # take root of intermediate list to finalize the sse
    diff = np.sqrt( intermediate )

    # find minimums
    for dist in diff:
        val = np.min( dist )
        idx = np.argmin( dist )
        min_idx.append( [idx] )
        min_val.append( [val] )

    return np.matrix( min_idx ), np.matrix( min_val )

# Given a data matrix A and a set of K initial means, compute the optimal
# cluster means for the data and an ID and an error for each data point
def kmeans_algorithm( A, means, MIN_CHANGE = 1e-7, MAX_ITERATIONS = 100 ):
    D = means.shape[1]    # number of dimensions
    K = means.shape[0]    # number of clusters
    N = A.shape[0]        # number of data points

    # iterate no more than MAX_ITERATIONS
    for i in range( MAX_ITERATIONS ):
        # calculate the codes by calling kmeans_classify
        # codes[j,0] is the id of the closest mean to point j
        indices, values = kmeans_classify( A, means )

        # initialize newmeans to a zero matrix identical in size to means
        # Meaning: the new means given the cluster ids for each point
        newmeans = np.zeros_like( means )

        # initialize a K x 1 matrix counts to zeros
        # Meaning: counts will store how many points get assigned to each mean
        counts = np.zeros( K )

        # for the number of data points
        for j in range( N ):
            # add to the closest mean (row codes[j,0] of newmeans) the mean of jth row of A
            idx = indices[j,0]
            newmeans[idx] += A[j]
            # add one to the corresponding count for the closest mean
            counts[idx] += 1

        # finish calculating the means, taking into account possible zero counts
        for j in range( K ):
            # if counts is not zero, divide the mean by its count
            if counts[j] != 0:
                newmeans[j] = newmeans[j] / counts[j]
            # else pick a random data point to be the new cluster mean
            else:
                newmeans[j] = A[random.sample( range( N ), 1 )]

        # test if the change is small enough and exit if it is
        diff = np.sum(np.square(means - newmeans) )
        means = newmeans
        if diff < MIN_CHANGE:
            break

    # call kmeans_classify one more time with the final means
    codes, errors = kmeans_classify( A, means )

    # return the means, codes, and errors
    return ( means, codes.T, errors )

def kmeans_quality( errors, k ):
    errors = np.square( errors )
    sse = np.sum( errors )
    logn = math.log( np.size( errors ), 2 )
    return sse + ( k / 2 ) * logn

def kmeans( d, headers, K, whiten=True ):
    '''Takes in a Data object, a set of headers, and the number of clusters to create
    Computes and returns the codebook, codes and representation errors.
    '''

    # assign to A the result getting the data given the headers
    if headers == []: A = d
    else: A = d.subset( headers )
    # if whiten is True
    # assign to W the result of calling vq.whiten on A
    if whiten: W = vq.whiten( A )
    else: W = A

    # assign to codebook the result of calling kmeans_init with W and K
    codebook = kmeans_init( W, K )

    # assign to codebook, codes, errors, the result of calling kmeans_algorithm with W and codebook
    codebook, codes, errors = kmeans_algorithm( W, codebook )
    quality = kmeans_quality( errors, K )

    # return the codebook, codes, and representation error
    return ( codebook, codes, errors, quality )
