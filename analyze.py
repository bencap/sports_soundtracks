""" analyze.py


This file contains functions
    *
    *

Created by Ben Capodanno on June 19th, 2019. Updated June 19th, 2019.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

def id_to_pd( filepath ):
    """ converts a csv to a pandas dataframe

    :param filepath: The location of the csv file
    :type filepath: String
    :returns: A df representing the data from the CSV file
    :rtype: Pandas DataFrame
    """

    with open( filepath ) as id_file:
        df = pd.read_csv( id_file, sep="," )

    return df

def extract_features( df, labels, standardize=True ):
    """ returns a data frame that has split the features from the labels

    :param df: The original, full dataframe
    :type df: Pandas DataFrame
    :param features: A list of the columns that are the labels for this dataset
    :type features: List of Strings
    :param standardize: Whether or not to standardize the data
    :type standardize: Boolean
    :returns: A tuple containing a feature and label dataframe
    :rtype: Tuple containing Pandas DataFrames
    """

    features = df.drop( columns = labels, axis=1 ).astype( float )
    labels = df[labels]

    if standardize:
        ssc = StandardScaler()
        features = ssc.fit_transform( features )

    return features, labels

def split_data( features, labels, standardize=True, test_size = 0.3 ):
    """ splits feature/label data into training, test, and validation sets

    :param features: A df containing the feature vectors
    :type features: Pandas DataFrame
    :param labels: A df containing data labels
    :type labels: Pandas DataFrame
    :param standardize: Whether or not to standardize the data
    :type standardize: Boolean
    :param test_size: The size of the test set
    :type test_size: Float
    :returns: A tuple containing standardized (if true) and split feature vectors and corresponding label df's
    :rtype: Tuple containing Pandas DataFrames (6 indices )
    """

    X_train, X_test, y_train, y_test = train_test_split( features, labels, test_size=test_size )

    if standardize:
        ssc = StandardScaler()
        X_train = ssc.fit_transform( X_train )
        X_test  = ssc.fit_transform( X_test )

    return X_train, X_test, y_train, y_test

def support_vector_machine( X_train, X_test, y_train, y_test ):
    """ implementation of support vector machine using scikitlearn SVC implementation

    :param X_train: Feature vector training set
    :type X_train: Pandas DataFrame
    :param X_test: Feature vector test set
    :type X_test: Pandas DataFrame
    :param y_train: Test set label accompaniments
    :type y_train: Pandas DataFrame
    :param y_test: Test set label accompaniments
    :type y_train: Pandas DataFrame
    :returns: A set of class predictions for the test data
    :rtype: Pandas DataFrame
    """

    support_vector_classifier = SVC( kernel='rbf', gamma="auto" )
    support_vector_classifier.fit( X_train, y_train.values.ravel() )
    y_pred = support_vector_classifier.predict( X_test )

    return y_pred

def conf_mat( y_test, y_pred, p=True ):
    """ generates and returns a confusion matrix for a set of predictions

    :param y_test: The set of true data classifications
    :type y_test: Pandas DataFrame
    :param y_pred: The model class predictions
    :type y_pred: Pandas DataFrame
    :param print: Whether to print the confusion matrix or not
    :type print: Boolean
    :returns: A confusion matrix for the predictions given
    :rtype: Array like
    """

    cm = confusion_matrix( y_test, y_pred )
    if p: print( cm, end = "\n\n" )
    return cm

def accuracy( cf_mat, p=True ):
    """ returns the accuracy of a given model

    :param cf_mat: A confusion matrix generated for the data
    :type cf_mat: Array like
    :param print: Whether to print the accuracy or not
    :type print: Boolean
    """
    
    # the accuracy is the diagonals of the confusion matrix, thus the incorrect values
    # are represented by all off diagonal entries, ie the matrix sum - matrix trace
    numerator = np.trace( cf_mat )
    denominator = np.sum( cf_mat ) - numerator

    acc = ( numerator/denominator ) * 100
    if p: print( "Accuracy : ",round( acc, 2 ), " %"  )
    
    return acc

def cross_validate( estimator, features, labels, splits=10, p=True ):
    """ validates the model across multiple train/test splits

    :param estimator: The model object used
    :type estimator: ML Object
    :param features: The feature set
    :type features: Pandas DataFrame
    :param labels: The class labels for feature set
    :type labels: Pandas DataFrame
    :param splits: The number of cv splits to perform
    :type splits: Integer
    :param print: Whether to print the cv score
    :type print: Boolean
    :returns: The cross validation scores
    :rtype: List of Floats
    """

    cross_val = cross_val_score( estimator = estimator, X = features, y = np.ravel( labels ), cv = splits, n_jobs = -1 )
    if p: print( "Cross Validation Accuracy : ",  round( cross_val.mean() * 100, 2 ), " %" )

    return cross_val

def exec_svm( df ):
    """ executes the support vector machine workflow

    :param df: A dataframe containing class labels and feature vectors
    :type df: Pandas DataFrame
    :returns: None
    :rtype: None
    """

    # model training
    df = df.drop( columns = ["track_name", "unique_id"] )
    features, labels = extract_features( df, ['game'] )
    X_train, X_test, y_train, y_test = split_data( features, labels, standardize=False )
    y_pred = support_vector_machine( X_train, X_test, y_train, y_test )

    # model results
    cm = conf_mat( y_test, y_pred )
    accuracy( cm )
    cv = cross_validate( SVC( kernel='rbf', gamma="auto" ), features, labels )
    print( cv )
    return

def df_to_pca( df, components = 10 ):
    """ loads a pandas data frame and performs PCA on it

    :param filepath: The DataFrame containing untransformed data
    :type filepath: Pandas DataFrame
    :param components: The number of component vectors to extract
    :type components: Integer
    :returns: The transformed ID file data
    :rtype: Pandas DataFrame
    """
    
    features = ["danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","avg_bar_len","avg_beat_len","avg_tatum_len","song_len"]
    headers  = [ "principal component " + str( i ) for i in range( 1,components+1 )]

    # Separating out the features
    x = df.loc[:, features].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    # generate pca data frame
    pca = PCA( n_components=components )
    principalComponents = pca.fit_transform( x )
    principalDf = pd.DataFrame( data = principalComponents, columns = headers )

    return pd.concat( [df[['track_name', 'unique_id', 'game']], principalDf], axis = 1 )

def vis_pca( finalDF ):
    """ visualizes PCA data in 2D
    
    :param finalDF: post PCA dataframe, see function 'PCA_id_file'
    :type finalDF: Pandas DataFrame
    :returns: None
    :rtype: None
    """

    # set plot size and labels
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = ['FIFA', 'NHL', 'NFL', 'NBA']
    colors = ['r', 'g', 'b', 'k']

    # plot the data
    for target, color in zip(targets,colors):
        indicesToKeep = finalDF['game'] == target
        ax.scatter( finalDF.loc[indicesToKeep, 'principal component 1']
                  , finalDF.loc[indicesToKeep, 'principal component 2']
                  , c = color
                  , s = 50)

    # add legend and grid lines
    ax.legend(targets)
    ax.grid()

    plt.show()
    return

def main( filepath ):
    df = id_to_pd( filepath )
    choice = input( "execute PCA and visualize on first two components? (y/n): " )
    if choice == "y":
        print()
        pca_df = df_to_pca( filepath )
        print( pca_df.head() )
        vis_pca( pca_df )
    choice = input( "execute SVM train/test workflow and display results? (y/n): ")
    if choice == "y":
        print()
        exec_svm( df )

if __name__ == "__main__":
    if len( sys.argv ) < 2:
        print( "USAGE: python analyze.py <filename>" )
        exit()
    main( sys.argv[1] )