#*******************************************************************************************************************************************************
## Code for checking and installing the required libraries/packages
#*******************************************************************************************************************************************************

import sys
import subprocess
import pkg_resources

required = {'pandas','pycaret','pywebio'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = list(required - installed)

if(missing):
	for lib in missing:
		python = sys.executable
		subprocess.check_call([python, '-m', 'pip', 'install', lib])


#*******************************************************************************************************************************************************
## Loading Libraries
#*******************************************************************************************************************************************************

from sklearn.feature_extraction.text import TfidfVectorizer
#from pycaret.classification import *

from pywebio import start_server
from pywebio.input import *
from pywebio.output import *

from functools import reduce
import pandas as pd


#*******************************************************************************************************************************************************
## Data Preparation for the prediction
#*******************************************************************************************************************************************************

## Function to calculate the document similarity

def textSimilarity(x1,x2):
    
    try:
        corpus = [x1,x2]

        vectorizer = TfidfVectorizer()
        sparse_matrix = vectorizer.fit_transform(corpus)

        doc_term_matrix = sparse_matrix.todense()
        df = pd.DataFrame(doc_term_matrix, 
                          columns=vectorizer.get_feature_names(), 
                          index=['doc_1', 'doc_2'])

        from sklearn.metrics.pairwise import cosine_similarity
        return round(cosine_similarity(df, df)[0][1], 6)
    
    except:
        return -1


#*******************************************************************************************************************************************************
## Function for Topic Modeling (For extarcting the topics from document)

def getTopicModelling(df, columnName):

    ## When setup is executed, following pre-processing steps are applied automatically
        # Removing Numeric Characters
        # Removing Special Characters
        # Word Tokenization
        # Stopword Removal
        # Bigram Extraction
        # Trigram Extraction
        # Lemmatizing

    from pycaret.nlp import setup, create_model, assign_model
    nlp = setup(data=df, target= columnName, session_id=200)

    model = create_model('lda', num_topics=4, multi_core = True)
    model_results = assign_model(model)

    df_topic = pd.DataFrame()
    df_topic['Topic_0_' + columnName] = model_results['Topic_0']
    df_topic['Topic_1_' + columnName] = model_results['Topic_1']
    df_topic['Topic_2_' + columnName] = model_results['Topic_2']
    df_topic['Topic_3_' + columnName] = model_results['Topic_3']
    
    df_topic.dropna(inplace=True)
    df_topic['id'] = model_results['id'].fillna(-999).astype('int')
    
    return df_topic, model


#*******************************************************************************************************************************************************
## Function for text length measures ( like word count, character count and averahe word length)

def textLengthMeasures(df, columnList=None):

    df_length_metrics = pd.DataFrame()

    for column in columnList:

        # word count: counts the number of tokens in the text (separated by a comma)
        df_length_metrics[column + '_word_count'] = df[column].apply(lambda x: len(str(x).split(",")))

        # character count: sum the number of characters of each token
        df_length_metrics[column + '_char_count'] = df[column].apply(lambda x: sum(len(word) for word in str(x).split(",")))

        # average word length: sum of words length divided by the number of words (character count/word count)
        df_length_metrics[column + '_avg_word_len'] = df_length_metrics[column + '_char_count'] / df_length_metrics[column + '_word_count']

    return df_length_metrics


#*******************************************************************************************************************************************************
## Function for cleaning data

def data_cleaning(df):
    
    ## Creating list of columns with object data type
    object_column_list = list(df.columns[df.dtypes == 'object'])   
    
    for column in object_column_list:
        df[column] = df[column].str.replace('[','').str.replace(']','').str.strip()
        
    return df


#*******************************************************************************************************************************************************
## Function for calculating features - Feature Engineering

def feature_enginnering(df):

    # Calculate column the document similarity(i.e. column similarity)
    df_features = pd.DataFrame()
    df_features['major_cosine_sim'] = df.apply(lambda x: textSimilarity(x['mentee_major'], x['mentor_major']), axis=1)
    df_features['help_topics_cosine_sim'] = df.apply(lambda x: textSimilarity(x['mentee_help_topics'], x['mentor_help_topics']), axis=1)
    df_features['expertise_cosine_sim'] = df.apply(lambda x: textSimilarity(x['mentee_experitse'], x['mentor_experitse']), axis=1)
    
    # Text Length Metrics
    object_column_list = list(df.columns[df.dtypes == 'object'])
    df_length_metrics = textLengthMeasures(df, columnList=object_column_list)
    
    df_final = pd.concat([df['id'], df_features, df_length_metrics], axis=1)
    
    return df_final


#*******************************************************************************************************************************************************
## Main Function
#*******************************************************************************************************************************************************

def main():

    ## Code for taking various inputs (Resume, input/output paths/directories etc.) through GUI

    prediction_inputs = input_group("Mentor and Mentee Match Prediction", [
        input('Enter the Test file directory path', required=True, name='test_file_dir_path'),

        file_upload(label='\nUpload Test File For Prediction', accept=['.csv'], multiple=False, required=True, name='test_file'),

        input('Enter the Build Model directory path', required=True, name='clf_model_dir_path'),

        file_upload(label='\nUpload Build ML Model File For Prediction', accept=['.pkl'], multiple=False, required=True, name='clf_model'),

        input('Enter the Prediction Output Directory Path', required=True, name='pred_output_dir_path')

    ])

    put_markdown("### Please wait for the Predictions to be generated ...")
    put_markdown("---")


    #*******************************************************************************************************************************************************
    ## Variables Initialization based on the inputs from GUI

    ## Directory and File path details
    test_file_dir_path = prediction_inputs['test_file_dir_path']
    test_file = prediction_inputs['test_file']
    clf_model_dir_path = prediction_inputs['clf_model_dir_path']
    clf_model = prediction_inputs['clf_model']
    pred_output_dir_path = prediction_inputs['pred_output_dir_path']


    ## Checking the correctness of directory paths
    if(test_file_dir_path[-1] != '/'):
        test_file_dir_path = test_file_dir_path + '/'

    if(clf_model_dir_path[-1] != '/'):
        clf_model_dir_path = clf_model_dir_path + '/'

    if(pred_output_dir_path[-1] != '/'):
        pred_output_dir_path = pred_output_dir_path + '/'


    #*******************************************************************************************************************************************************
    ## Reading the Test File and saving it as a dataframe
    df = pd.read_csv(test_file_dir_path + test_file['filename'])

    ## Adding 'id' which holds the row index
    indexes = list(df.index)
    df['id'] = indexes

    ## Calling 'data_cleaning' to clean the train dataframe
    df_test_cleaned = data_cleaning(df)

    ## Calling 'feature_enginnering' function for generating features
    df_test_features = feature_enginnering(df_test_cleaned)

    ## Calling Topic Modeling function to get the topics for each text column
    df_test_mentee_major_topics, test_cleanedmodel_mentee_major = getTopicModelling(df_test_cleaned, 'mentee_major')
    df_test_mentee_help_topics_topics, test_cleanedmodel_mentee_help_topics = getTopicModelling(df_test_cleaned, 'mentee_help_topics')
    df_test_mentee_experitse_topics, test_model_mentee_experitse = getTopicModelling(df_test_cleaned, 'mentee_experitse')
    df_test_mentor_major_topics, test_model_mentor_major = getTopicModelling(df_test_cleaned, 'mentor_major')
    df_test_mentor_help_topics_topics, test_model_mentor_help_topics = getTopicModelling(df_test_cleaned, 'mentor_help_topics')
    df_test_mentor_experitse_topics, test_model_mentor_experitse = getTopicModelling(df_test_cleaned, 'mentor_experitse')

    ## Merging dataframes

    # List of dataframes we want to merge
    test_data_frames = [df_test_cleaned[['id']], df_test_features, df_test_mentee_major_topics,df_test_mentee_help_topics_topics,
                        df_test_mentee_experitse_topics,df_test_mentor_major_topics,df_test_mentor_help_topics_topics,df_test_mentor_experitse_topics]

    df_test_merged = reduce(lambda  left,right: pd.merge(left,right,on=['id'],
                                                how='left'), test_data_frames)

    ## Missing Values Imputation
    df_test_merged.fillna(0, inplace=True)

    ## Dropping id column
    df_test_merged.drop(['id'], axis=1, inplace=True)


    #*******************************************************************************************************************************************************
    ## Loading model
    from pycaret.classification import load_model,predict_model
    saved_model = load_model(clf_model_dir_path + clf_model['filename'].replace('.pkl', ''))

    ## Making Predictions
    predictions = predict_model(saved_model, data=df_test_merged)

    ## Saving the predictions
    df = pd.concat([df, predictions[['Score','Label']]], axis=1)
    df.to_csv(pred_output_dir_path + 'mentor_mentee_similarity_predictions.csv', index=False)

    ## Putting message
    put_markdown("### Predictions are generated. Check --> mentor_mentee_similarity_predictions.csv")
    put_markdown("> **_Note:_** Threshold is taken as 0.5 for calculating Label ")


if __name__ == '__main__':
    
    start_server(main, auto_open_webbrowser=True)