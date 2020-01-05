#!/usr/bin/env python
# coding: utf-8

# In[79]:


# Import libraries
import sys, sqlite3, re, pickle, nltk, warnings

import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD

nltk.download('punkt', 'stopwords')

warnings.simplefilter('ignore')


# In[94]:


def load_data(database_name):
    """Load and merge messages and categories datasets
    
    Args:
    database_filename: string. Filename for SQLite database containing cleaned message data.
       
    Returns:
    X: dataframe. Dataframe containing features dataset.
    Y: dataframe. Dataframe containing labels dataset.
    category_names: list of strings. List containing category names.
    """
    # Load data from database
    #conn = sqlite3.connect("Data/cleanTwitterDB.db")
    conn = sqlite3.connect(database_name)
    df = pd.read_sql_query("SELECT * FROM messages", conn)

    # Create X and Y datasets
    X = df["message"]
    Y = df.drop(['message', 'genre', 'id', 'original'], axis = 1)
    
    # Create list containing all category names
    category_names = list(Y.columns.values)
    
    return X, Y, category_names


# In[81]:


def tokenize(text):
    """Normalize, tokenize and stem text string
    
    Args:
    text: string. String containing message for processing
       
    Returns:
    stemmed: list of strings. List containing normalized and stemmed word tokens
    """
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)

    # lemmatizer words
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


# In[82]:


def performance_metric(model, X_test, y_test):
    '''
    Function to generate classification report on the model
    Input: Model, test set ie X_test & y_test
    Output: Prints the Classification report
    '''
    y_pred = model.predict(X_test)
    for i, col in enumerate(y_test):
        return classification_report(y_test[col], y_pred[:, i])


# In[83]:


def build_model(X_train, y_train):
    """Build a machine learning pipeline
    
    Args:
    None
       
    Returns:
    cv: gridsearchcv object. Gridsearchcv object that transforms the data, creates the 
    model object and finds the optimal model parameters.
    """
    # Create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('best', TruncatedSVD()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Create parameters dictionary, Param tunning 
    parameters = { #'vect__ngram_range': ((1, 1), (1, 2)), 
              #'vect__max_df': (0.5, 1.0), 
              #'vect__max_features': (None, 5000), 
              'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [50, 100],
              'clf__estimator__learning_rate': [1,2] }
    
    # Create scorer
    scorer = make_scorer(performance_metric)
    
    # Create grid search object
    #cv = GridSearchCV(pipeline, param_grid = parameters, scoring = scorer, verbose = 10)
    cv = GridSearchCV(pipeline, param_grid=parameters)
    cv.fit(X_train, y_train)
    return cv


# In[84]:


def get_eval_metrics(actual, predicted, col_names):
    """Calculate evaluation metrics for ML model
    
    Args:
    actual: array. Array containing actual labels.
    predicted: array. Array containing predicted labels.
    col_names: list of strings. List containing names for each of the predicted fields.
       
    Returns:
    metrics_df: dataframe. Dataframe containing the accuracy, precision, recall 
    and f1 score for a given set of actual and predicted labels.
    """
    metrics = []
    
    # Calculate evaluation metrics for each set of labels
    for i in range(len(col_names)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i], average='micro')
        recall = recall_score(actual[:, i], predicted[:, i], average='micro')
        f1 = f1_score(actual[:, i], predicted[:, i], average='micro')
        
        metrics.append([accuracy, precision, recall, f1])
    
    # Create dataframe containing metrics
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, index = col_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return metrics_df


# In[85]:


def evaluate_model(model, X_test, Y_test, category_names):
    """Returns test accuracy, precision, recall and F1 score for fitted model
    
    Args:
    model: model object. Fitted model object.
    X_test: dataframe. Dataframe containing test features dataset.
    Y_test: dataframe. Dataframe containing test labels dataset.
    category_names: list of strings. List containing category names.
    
    Returns:
    None
    """
    # Predict labels for test dataset
    Y_pred = model.predict(X_test)
    
    # Calculate and print evaluation metrics
    eval_metrics = get_eval_metrics(np.array(Y_test), Y_pred, category_names)
    print(eval_metrics)


# In[86]:


def save_model(model, model_filepath):
    """Pickle fitted model
    
    Args:
    model: model object. Fitted model object.
    model_filepath: string. Filepath for where fitted model should be saved
    
    Returns:
    None
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


# In[92]:


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print(model_filepath)
        print('Building model...')
        #model = build_model()
        model = build_model(X_train, Y_train)
        
        print('Training model...')
        #model.fit(model, X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '              'as the first argument and the filepath of the pickle file to '              'save the model to as the second argument. \n\nExample: python '              'train_classifier.py Data/cleanTwitterDB.db classifier.pkl')


# In[93]:


if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




