#*******************************************************************************************************************************************************
## Code for checking and installing the required libraries/packages
#*******************************************************************************************************************************************************

import sys
import subprocess
import pkg_resources

required = {'pandas','pycaret','fastapi','uvicorn'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = list(required - installed)

if(missing):
	for lib in missing:
		python = sys.executable
		subprocess.check_call([python, '-m', 'pip', 'install', lib])


#*******************************************************************************************************************************************************     
## Importing Libraries
from MentorMenteeDetails import MentorMenteeDetails
from fastapi import FastAPI
import uvicorn

from sklearn.feature_extraction.text import TfidfVectorizer
from functools import reduce
import numpy as np
import pandas as pd
import pickle


#*******************************************************************************************************************************************************
## Creating an app object
app = FastAPI()

# ## Loading model object
# model = open("final_lightgbm_model_06Aug2021.pkl","rb")
# classifier=pickle.load(model)


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
    df_topic['id'] = model['id'].fillna(-999).astype('int')
    
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
## Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Welcome to Mentor-Mentee matching API'}


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_match(data:MentorMenteeDetails):

    # dictionary to dataframe pandas
    df = pd.DataFrame([data.dict()])

    ## Adding 'id' which holds the row index
    indexes = list(df.index)
    df['id'] = indexes

    ## Calling 'data_cleaning' to clean the train dataframe
    df_test_cleaned = data_cleaning(df)

    ## Calling 'feature_enginnering' function for generating features
    df_test_features = feature_enginnering(df_test_cleaned)

    # ## Calling Topic Modeling function to get the topics for each text column
    data_topics = np.random.randint(5, size=(1,24))

    columns_topics = ['Topic_2_mentee_major', 'Topic_3_mentor_major', 'Topic_2_mentee_help_topics', 'Topic_1_mentee_major', 'Topic_0_mentor_help_topics', 
    'Topic_0_mentee_major', 'Topic_1_mentor_help_topics', 'Topic_2_mentee_experitse', 'Topic_0_mentor_experitse', 'Topic_0_mentee_help_topics', 
    'Topic_2_mentor_help_topics', 'Topic_1_mentee_experitse', 'Topic_0_mentee_experitse', 'Topic_2_mentor_experitse', 'Topic_0_mentor_major', 
    'Topic_1_mentor_experitse', 'Topic_3_mentee_major', 'Topic_3_mentor_help_topics', 'Topic_3_mentee_help_topics', 'Topic_1_mentee_help_topics', 
    'Topic_2_mentor_major', 'Topic_1_mentor_major', 'Topic_3_mentee_experitse', 'Topic_3_mentor_experitse']

    df_topics = pd.DataFrame(data=data_topics,columns=columns_topics)

    # df_test_mentee_major_topics, test_cleanedmodel_mentee_major = getTopicModelling(df_test_cleaned, 'mentee_major')
    # df_test_mentee_help_topics_topics, test_cleanedmodel_mentee_help_topics = getTopicModelling(df_test_cleaned, 'mentee_help_topics')
    # df_test_mentee_experitse_topics, test_model_mentee_experitse = getTopicModelling(df_test_cleaned, 'mentee_experitse')
    # df_test_mentor_major_topics, test_model_mentor_major = getTopicModelling(df_test_cleaned, 'mentor_major')
    # df_test_mentor_help_topics_topics, test_model_mentor_help_topics = getTopicModelling(df_test_cleaned, 'mentor_help_topics')
    # df_test_mentor_experitse_topics, test_model_mentor_experitse = getTopicModelling(df_test_cleaned, 'mentor_experitse')

    ## Merging dataframes

    # List of dataframes we want to merge
    # test_data_frames = [df_test_cleaned[['id']], df_test_features, df_test_mentee_major_topics,df_test_mentee_help_topics_topics,
    #                     df_test_mentee_experitse_topics,df_test_mentor_major_topics,df_test_mentor_help_topics_topics,df_test_mentor_experitse_topics]

    test_data_frames = [df_test_cleaned[['id']], df_test_features]

    df_test_merged = reduce(lambda  left,right: pd.merge(left,right,on=['id'],
                                                how='left'), test_data_frames)

    df_test_merged = pd.concat([df_topics,df_test_merged], axis=1)

    ## Missing Values Imputation
    df_test_merged.fillna(0, inplace=True)

    ## Dropping id column
    df_test_merged.drop(['id'], axis=1, inplace=True)

    ## Loading model
    from pycaret.classification import load_model,predict_model
    saved_model = load_model('final_lightgbm_model_06Aug2021')

    ## Making Predictions
    predictions = predict_model(saved_model, data=df_test_merged)
    predictions.head()
    score = predictions['Score'][0]

    return {'prediction': score}


## Run the API with uvicorn (Will run on http://127.0.0.1:8000)
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload
