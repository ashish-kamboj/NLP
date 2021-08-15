## Problem Statement
Objective is to build a model to predict the probability of match from mentor and mentee data (as described below) which be helpful in assigning a relevant mentor (based on the skills and expertise in specific fields) to mentee (which seeks for the helps in the same/similar topics/domains which mentor is having).

## Data Description

| Columns                      | Description                                                      |
|:-----------------------------|:-----------------------------------------------------------------|
| mentee_major                 | Mentee’s major selected in college                               |
| mentee_help_topics           | The topics in which mentee is seeking help                       |
| mentee_experitse             | The domain in which mentee is looking to build his expertise     |
| mentor_major                 | Mentor’s major selected in college                               |
| mentor_help_topics           | The topic in which a mentor is willing to help mentees           |
| mentor_experitse             | Mentor’s domain of expertise                                     |         
| final_match                  | 1 if the mentee and mentor had actually matched                  |
|                              | 0 if the mentee skipped the mentor (was not a good match)        |

## Data Cleaning (Using PyCaret)
 - Removed numeric characters
 - Removed special characters
 - Word Tokenization
 - Stopwords removal
 - Lemmatization

## Feature Engineering
 - Topic Modeling (for feature generation) using Latent Dirichlet Allocation (LDA)
 - Bi-gram extraction
 - Tri-gram extraction
 - Text Length metrics
   - Word count
   - Character count
   - Average word length
 - Calculated document similarity (Cosine similarity) using Tfidf vectorization

## Model Building (Using PyCaret)
 - Tried out the different classification models and choose the best performing after comparing the metrics (like Accuracy, ROC, F1-score etc.)
 - Tuned the best performing model (Hyperparameter tuning)
 - Model Evaluation
 - Model Interpretation
 - Calculated Feature importance

## Model Deployment (Using PyWebIO)
![PyWebIO Application](https://github.com/ashish-kamboj/NLP/blob/master/mentor-mentee-match-prediction/images/pywebio_application.gif)

## Creating a REST API using FastAPI
You can now run your app via

```
$ uvicorn app:app --reload
```
