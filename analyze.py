""" analyze.py

This module implements common machine learning algorithms through external libraries.
It is designed to be manipulable and mostly able to work on a variety of data sets
so long as they are cleaned and you can change the values for the id_cols, cat_col, 
and feature_cols variables within the main function. The main function is should be 
able to run the pipeline for any of the implemented algorithms, listed below:

Implemented Models: 
    ** Model (library)
    -> Support Vector Classification (sklearn)
    -> Random Forest Classifier (sklearn)
    -> Logistic Regression (sklearn)
    -> XGBoost Classification (xgboost)
    -> Neural Network (TensorFlow)

This file contains functions
    * id_to_pd( filepath ) - converts a data file to a Pandas DataFrame
    * extract_features( df, labels, standardize ) - Splits a Pandas DataFrame into its derivative features and class labels
    * split_data( features, labels, standardize, test_size ) - Splits a Pandas DataFrame into a training and test set
    * sep_categorical( features ) - Separates categorical feature columns from a Pandas DataFrame
    * tf_create_features( df, categorical_cols, continuous_cols ) - Creates tensor objects from feature vectors
    * deep_model( features, input_fun, pred_fun, size, shape, arch, n_classes, steps ) - creates and trains tensor flow neural net
    * train_model( X_train, X_test, y_train, y_test, model_type ) - creates and trains non neural-net model
    * conf_mat( y_test, y_pred, p ) - generates and prints a confusion matrix for the model
    * accuracy( cf_mat, p ) - generates and prints an accuracy measure for the model
    * cross_validate( estimator, features, labels, splits, p ) - cross-validation of model, also prints average accuracy
    * execute_workflow( df, id_cols, cat_col, model_type, pca ) - executes the workflow of training a model from DataFrame subsetting to accuracy statistics
    * execute_tf_workflow( df ) - executes the tensor flow workflow from DataFrame subsetting to accuracy reporting
    * df_to_pca( df, identifiers, features, components ) - executes PCA on a given dataset
    * vis_pca( finalDF ) - Visualizes the primary variation in a PCA DataFrame

Created by Ben Capodanno on June 19th, 2019. Updated June 25th, 2019.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR )

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

    df_features = df.drop( columns = labels, axis=1 ).astype( float )
    labels = df[labels]

    if standardize:
        ssc = StandardScaler()
        features = ssc.fit_transform( df_features )
        features = pd.DataFrame( features, index=df_features.index, columns=df_features.columns)

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

def sep_categorical( features ):
    """ separates categorical and continuos features

    :param features: unseperated feature vectors
    :type features: Pandas DataFrame
    :returns: A tuple containing the separated categorical and continuos features
    :rtype: Tuple of Pandas DataFrames
    """

    categorical_cols = [col for col in features.columns if len(features[col].unique())==2 or features[col].dtype=='O']
    continuous_cols  = [col for col in features.columns if len(features[col].unique())>2 and (features[col].dtype=='int64' or features[col].dtype=='float64')]

    return categorical_cols, continuous_cols

def tf_create_features( df, categorical_cols, continuous_cols ):
    """ creates the feature columns for tf pipeline

    :param df: The full features dataframe
    :type df: Pandas DataFrame
    :param categorical_cols: The columns containing categorical data
    :type categorical_cols: Pandas DataFrame
    :param continuous_cols: The columns containing continuous data
    :type continuous_cols: Pandas DataFrame
    :returns:
    :rtype:
    """

    # if the data type is an object, put it into this feature set in tf format
    categorical_object_feat_cols = [tf.feature_column.embedding_column(                                   
                                    tf.feature_column.categorical_column_with_hash_bucket(
                                    key=col,hash_bucket_size=1000),dimension = len(df[col].unique()))
                                    for col in categorical_cols if df[col].dtype=='O']

    # if the data type is numeric, put it into this feature set in tf format
    categorical_numeric_feat_cols = [tf.feature_column.embedding_column(                 
                                     tf.feature_column.categorical_column_with_identity(
                                     key=col,num_buckets=2),dimension = len(df[col].unique()))
                                     for col in categorical_cols if df[col].dtype=='int64' or df[col].dtype=='float64']

    # all continuos cols can be converted more easily with less manipulation
    continuous_feat_cols = [tf.feature_column.numeric_column(key=col) for col in continuous_cols]

    feat_cols = categorical_object_feat_cols + categorical_numeric_feat_cols + continuous_feat_cols

    return feat_cols

def deep_model( features, input_fun, pred_fun, size=10, shape=10, arch=10, n_classes=2, steps=5000 ):
    """
    """
    DNN_model = tf.estimator.DNNClassifier( hidden_units=[size,shape,arch], feature_columns=features, n_classes=n_classes )
    DNN_model.train(input_fn=input_fun, steps=steps)
    return DNN_model.predict (pred_fun )

def train_model( X_train, X_test, y_train, y_test, model_type ):
    """ implementation of support vector machine using scikitlearn SVC implementation

    :param X_train: Feature vector training set
    :type X_train: Pandas DataFrame
    :param X_test: Feature vector test set
    :type X_test: Pandas DataFrame
    :param y_train: Test set label accompaniments
    :type y_train: Pandas DataFrame
    :param y_test: Test set label accompaniments
    :type y_train: Pandas DataFrame
    :param model_type: The model to use. Can be "svc" or "rfc" or "xgb" or 'logit"
    :type model_type: String
    :returns: The classifier and a set of class predictions for the test data
    :rtype: Tuple containing the svc classifier object and a Pandas DataFrame
    """

    if model_type == "svc": model = SVC( kernel='rbf', gamma="auto" )
    if model_type == "rfc": model = RandomForestClassifier( n_estimators = 100 )
    if model_type == "xgb": model = XGBClassifier()
    if model_type == "logit": model = LogisticRegression( solver="lbfgs", multi_class="auto")
    if model_type == "knn": model = KNeighborsClassifier( n_neighbors = 4, weights = 'distance' )
    model.fit( X_train, y_train.values.ravel() )
    y_pred = model.predict( X_test )

    return model, y_pred

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

def accuracy( cf_mat, labels, p=True ):
    """ returns the accuracy of a given model

    :param cf_mat: A confusion matrix generated for the data
    :type cf_mat: Array like
    :param labels: The model class labels
    :type labels: Pandas DataFrame
    :param print: Whether to print the accuracy or not
    :type print: Boolean
    """
    
    # the accuracy is the diagonals of the confusion matrix
    diagonals = np.diagonal( cf_mat )
    correct_class = np.trace( cf_mat )
    total_sum = np.sum( cf_mat )
    row_sum = np.sum( cf_mat, axis=1 )
    col_sum = np.sum( cf_mat, axis=0 )

    acc = ( correct_class/total_sum ) * 100
    if p: 
        print( "Overall Prediction Accuracy : ",round( acc, 2 ), " %"  )
        print()
        for i in enumerate( row_sum ):
            print( "Class ", labels['game'].iloc[i[0]]," Precision: ", round( diagonals[i[0]]/i[1] * 100, 2 ), " %" )
        print()
        for i in enumerate( col_sum ):  
            print( "Class ", labels['game'].iloc[i[0]]," Recall: ", round( diagonals[i[0]]/i[1] * 100, 2 ), " %" )
        print()
    return acc

def cross_validate( estimator, features, labels, splits=4, p=True ):
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

def execute_workflow( df, id_cols, cat_col, model_type, pca=False ):
    """ executes the model training and validation workflow

    :param df: A dataframe containing class labels and feature vectors
    :type df: Pandas DataFrame
    :param id_cols: Columns used just for data identification
    :type id_cols: List of Strings
    :param cat_col: Column used for data classification
    :type cat_col: List of Strings
    :param model_type: The model to use. Can be "svc" or "rfc" or "xgb" or "logit"
    :type model_type: String
    :param pca: Whether to perform PCA on the data prior to model construction
    :type pca: Boolean
    :returns: None
    :rtype: None
    """

    # model training
    if pca: 
        non_feature = id_cols + cat_col
        # if a col name is not in the non_features list, put it in feature col list
        feature_cols = [col for col in df.columns if col not in non_feature]
        df = df_to_pca( df, non_feature, feature_cols )

    df = df.drop( columns = id_cols )
    features, labels = extract_features( df, cat_col )
    X_train, X_test, y_train, y_test = split_data( features, labels, standardize=False )
    model, y_pred = train_model( X_train, X_test, y_train, y_test, model_type )

    # model results
    cm = conf_mat( y_test, y_pred )
    accuracy( cm, labels.drop_duplicates() )
    cross_validate( model, features, labels )
    print()

    return

def execute_tf_workflow( df, id_cols, cat_col, pca=False ):
    """ executes the tensor flow workflow

    :param df: A dataframe containing class labels and feature vectors
    :type df: Pandas DataFrame
    :param id_cols: columns used just for data identification
    :type id_cols: List of Strings
    :param cat_col: Column used for data classification
    :type cat_col: List of Strings
    :param pca: Whether to perform PCA on the data prior to model construction
    :type pca: Boolean
    :returns: None
    :rtype: None
    """

    if pca: 
        non_feature = id_cols + cat_col
        # if a col name is not in the non_features list, put it in feature col list
        feature_cols = [col for col in df.columns if col not in non_feature]
        df = df_to_pca( df, non_feature, feature_cols )

    df = df.drop( columns = id_cols )
    feat, label = extract_features( df, cat_col )
    factored = pd.factorize( label[cat_col[0]] )
    label = pd.Series( factored[0], index=label.index )

    X_T, X_t, y_T, y_t = split_data( feat, label, standardize=False, test_size=0.3 )

    cat_cols, cont_cols = sep_categorical( feat )
    feat = tf_create_features( feat, cat_cols, cont_cols )

    # tf expects categorical coding in integer type
    for cat in cat_cols:
        X_T[cat] = df[cat].astype( int )
        X_t[cat] = df[cat].astype( int )
    
    input_fun = tf.compat.v1.estimator.inputs.pandas_input_fn( X_T, y_T, batch_size=50, num_epochs=1000, shuffle=True )
    pred_fun = tf.compat.v1.estimator.inputs.pandas_input_fn( X_t, batch_size=50, shuffle=False )

    predictions = deep_model( feat, input_fun, pred_fun, n_classes=4 )
    res_pred = list(predictions)

    y_pred = []
    for i in range(len(res_pred)):
        y_pred.append(res_pred[i]["class_ids"][0])

    rep = classification_report(y_t,y_pred)

    for game in enumerate( factored[1] ):
        print( "Category %d -> %s" % ( game[0], game[1] ) )

    print()
    print( rep )

def df_to_pca( df, identifiers, features, components = 10 ):
    """ loads a pandas data frame and performs PCA on it

    :param filepath: The DataFrame containing untransformed data
    :type filepath: Pandas DataFrame
    :param identifiers: List of column names that are used for data identification
    :type identifiers: List of Strings
    :param features: List of column names representing feature data
    :type features: List of Strings
    :param components: The number of component vectors to extract
    :type components: Integer
    :returns: The transformed ID file data
    :rtype: Pandas DataFrame
    """
    
    headers  = [ "pc" + str( i ) for i in range( 1,components+1 )]

    # Separating out the features
    x = df.loc[:, features].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    # generate pca data frame
    pca = PCA( n_components=components )
    principalComponents = pca.fit_transform( x )
    principalDf = pd.DataFrame( data = principalComponents, columns = headers )

    return pd.concat( [df[identifiers], principalDf], axis = 1 )

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
    id_cols = ["track_name", "unique_id"]
    cat_col = ["game"]
    feature_cols = ["danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness",
                    "liveness","valence","tempo","avg_bar_len","avg_beat_len","avg_tatum_len","song_len"]

    choice = input( "execute PCA and visualize on first two components? (y/n): " )
    if choice == "y":
        print()
        pca_df = df_to_pca( df, id_cols + cat_col, feature_cols )
        print( pca_df.head() )
        vis_pca( pca_df )
        print()

    print( "Use command 'yp' to execute the given model with pca data instead of untransformed data" )
    choice = input( "execute Support Vector Machine train/test workflow and display results? (y/yp/n): ")
    if choice == "y":
        print()
        execute_workflow( df, id_cols, cat_col, "svc" )
    if choice == "yp":
        print()
        execute_workflow( df, id_cols, cat_col, "svc", pca=True )

    choice = input( "execute Random Forest Classifier train/test workflow and display results? (y/yp/n): ")
    if choice == "y":
        print()
        execute_workflow( df, id_cols, cat_col, "rfc" )
    if choice == "yp":
        print()
        execute_workflow( df, id_cols, cat_col, "rfc", pca=True )

    choice = input( "execute XGBoost train/test workflow and display results? (y/yp/n): ")
    if choice == "y":
        print()
        execute_workflow( df, id_cols, cat_col, "xgb" )
    if choice == "yp":
        print()
        execute_workflow( df, id_cols, cat_col, "xgb", pca=True )

    choice = input( "execute Logistic Regression train/test workflow and display results? (y/yp/n): ")
    if choice == "y":
        print()
        execute_workflow( df, id_cols, cat_col, "logit" )
    if choice == "yp":
        print()
        execute_workflow( df, id_cols, cat_col, "logit", pca=True )

    choice = input( "execute K Nearest Neighbor train/test workflow and display results? (y/yp/n): ")
    if choice == "y":
        print()
        execute_workflow( df, id_cols, cat_col, "knn" )
    if choice == "yp":
        print()
        execute_workflow( df, id_cols, cat_col, "knn", pca=True )

    choice = input( "execute TensorFlow train/test workflow and display results? (y/yp/n): ")
    if choice == "y":
        print()
        execute_tf_workflow( df, id_cols, cat_col )
    if choice == "yp":
        print()
        execute_tf_workflow( df, id_cols, cat_col, pca=True )

if __name__ == "__main__":
    if len( sys.argv ) < 2:
        print( "USAGE: python analyze.py <filename>" )
        exit()
    main( sys.argv[1] )