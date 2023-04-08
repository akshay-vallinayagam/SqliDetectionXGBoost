#Course: Project 1 - INCS 870
#School: New York , Vancouver Campus
#Contributors: Akshay Vallinayagam(avallina@nyit.edu), Veera Venkata Sai Krishna Sunkara(vsunkara@nyit.edu), Jasneet Singh Parmar(jparmar@nyit.edu)

import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import re
import sklearn
import pickle
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from scipy.sparse import hstack

#Read the training dataset from csv file
X_train = pd.read_csv("train.csv")
X_train.shape
Y_train = X_train.drop('label',axis = 1)
X_train.to_csv("train_copy.csv")
X_train  = pd.read_csv("train_copy.csv")

# Preprocessing data
# Conversion to lower case
def convert_To_Lower_case(x):
    return x.lower()
    
#Removal of duplicates
def preprocess(data):
    data['Query'] = data['Query'].apply(convert_To_Lower_case)
    boolean = data.duplicated(subset = ['Query'])
    data.drop_duplicates(subset = ['Query'],inplace = True)
    #dropping the few queries which has both the label to avoid ambiguity
    #data.drop_duplicates(subset = ['Query'],keep = False,inplace = True)
    return data

X_train = preprocess(X_train)

# Creating all the features
def number_of_single_quotations(x):
    list_single_quotes = re.findall("'",x)
    return len(list_single_quotes)

def number_of_double_quotations(x):
    list_double_quotes = re.findall("\"",x)
    return len(list_double_quotes)

def number_of_hash(x):
    list_hash = re.findall("#",x)
    return len(list_hash)

def number_of_dollar(x):
    list_dollar = re.findall("\$",x)
    return len(list_dollar)

def number_of_paranthesis(x):
    list_paranthesis = re.findall("\(|\)",x)
    return len(list_paranthesis)

def number_of_square_brackets(x):
    list_square_brackets = re.findall("\[|\]",x)
    return len(list_square_brackets)

def number_of_at_symbol(x):
    list_at_symbol = re.findall("@",x)
    return len(list_at_symbol)

def number_of_colon(x):
    list_colon = re.findall(":",x)
    return len(list_colon)

def number_of_semicolon(x):
    list_semicolon = re.findall(";",x)
    return len(list_semicolon)

def number_of_equals(x):
    list_equals = re.findall("=",x)
    return len(list_equals)

def number_of_angular_brackets(x):
    list_angular_brackets = re.findall("<|>",x)
    return len(list_angular_brackets)

def number_of_question_mark(x):
    list_question_mark = re.findall("\?",x)
    return len(list_question_mark)

def number_of_under_score(x):
    list_under_score = re.findall("\_",x)
    return len(list_under_score)

def number_of_arithematic(x):
    list_arithematic = re.findall("\+|-|[^\/]\*|\/[^\*]",x)
    return len(list_arithematic)

def number_of_comma(x):
    list_comma = re.findall(",",x)
    return len(list_comma)

def number_of_dot(x):
    list_dot = re.findall("\.",x)
    return len(list_dot)

def number_of_single_comment(x):
    list_single_comment = re.findall("(--)",x)
    return len(list_single_comment)

def number_of_white_space(x):
    list_white_space = re.findall("\s+",x)
    return len(list_white_space)

def number_of_percentage(x):
    list_percentage = re.findall("%",x)
    return len(list_percentage)

def number_of_logical_operators(x):
    list_logical_operators = re.findall("\snot\s|\sand\s|\sor\s|\sxor\s|&&|\|\||!",x)
    return len(list_logical_operators)

def number_of_punctuation(x):
    list_punctuation = re.findall("[!\"#$%&\'()*+,-.\/:;<=>?@[\\]^_`{|}~]",x)
    return len(list_punctuation)

def number_of_hexadecimal(x):
    list_hexadecimal = re.findall("0[xX][0-9a-fA-F]+\s",x)
    return len(list_hexadecimal)

def number_of_null(x):
    list_null = re.findall("null",x)
    return len(list_null)

def number_of_digits(x):
    list_digits = re.findall("[0-9]",x)
    return len(list_digits)

def number_of_alphabets(x):
    list_alphabets = re.findall("[a-zA-Z]",x)
    return len(list_alphabets)
    
def number_of_keywords(x):
    keywords = ['select', 'insert', 'update', 'delete', 'from', 'where', 'join', 'on', 'group by', 'order by', 'having']
    return sum([1 for kw in keywords if kw in x])
    
def number_of_tables(x):
    tables = re.findall(r'\bfrom\b\s+(\w+)', x)
    return len(tables)
    
def number_of_wheres(x):
    wheres = re.findall(r'\bwhere\b', x)
    return len(wheres)
    
def number_of_ors(x):
    ors = re.findall(r'\bor\b', x)
    return len(ors)
    
def number_of_unions(x):
    unions = re.findall(r'\bunion\b', x)
    return len(unions)
    
def number_of_subqueries(x):
    subqueries = re.findall(r'\bselect\b.*?\bfrom\b', x, re.DOTALL)
    return len(subqueries)
    
def number_of_joins(x):
    joins = re.findall(r'\bjoin\b', x)
    return len(joins)
    

def create_features(data):
    data['number_single_quotes'] = data.Query.apply(number_of_single_quotations)
    data['number_double_quotes'] = data.Query.apply(number_of_double_quotations) 
    data['number_hash'] = data.Query.apply(number_of_hash) 
    data['number_dollar'] = data.Query.apply(number_of_dollar)
    data['number_paranthesis'] = data.Query.apply(number_of_paranthesis)
    data['number_square_brackets'] = data.Query.apply(number_of_square_brackets)
    data['number_at_symbol'] = data.Query.apply(number_of_at_symbol)
    data['number_colon'] = data.Query.apply(number_of_colon)
    data['number_semicolon'] = data.Query.apply(number_of_semicolon)
    data['number_equals'] = data.Query.apply(number_of_equals)
    data['number_angular_brackets'] = data.Query.apply(number_of_angular_brackets)
    data['number_question_mark'] = data.Query.apply(number_of_question_mark)
    data['number_under_score'] = data.Query.apply(number_of_under_score)
    data['number_arithematic'] = data.Query.apply(number_of_arithematic)
    data['number_comma'] = data.Query.apply(number_of_comma)
    data['number_dot'] = data.Query.apply(number_of_dot)
    data['number_single_comment'] = data.Query.apply(number_of_single_comment)
    data['number_white_space'] = data.Query.apply(number_of_white_space) 
    data['number_percentage'] = data.Query.apply(number_of_percentage) 
    data['number_logical_operators'] = data.Query.apply(number_of_logical_operators) 
    data['number_punctuation'] = data.Query.apply(number_of_punctuation)
    data['number_hexadecimal'] = data.Query.apply(number_of_hexadecimal)
    data['number_null'] = data.Query.apply(number_of_null)
    data['number_digits'] = data.Query.apply(number_of_digits)
    data['number_alphabets'] = data.Query.apply(number_of_alphabets)
    data['number_keywords'] = data.Query.apply(number_of_keywords)
    data['number_tables'] = data.Query.apply(number_of_tables)
    data['number_wheres'] = data.Query.apply(number_of_wheres)
    data['number_ors'] = data.Query.apply(number_of_ors)
    data['number_unions'] = data.Query.apply(number_of_unions)
    data['number_subqueries'] = data.Query.apply(number_of_subqueries)
    data['number_joins'] = data.Query.apply(number_of_joins)
    return data

#Feature extraction for training dataset
X_train = create_features(X_train)

# Saving the file with new features
X_train.to_csv("Preprocessed_X_train_copy.csv",index = False)

# Load the saved file
X_train = pd.read_csv("Preprocessed_X_train_copy.csv")

Y_train = X_train['label']
X_train = X_train.drop(['label'],axis = 1)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1,2),max_features = 50000)
X = vectorizer.fit(X_train['Query'])
pickle.dump(X, open('train_comment_features.pkl', 'wb'))
X_train_query = X.transform(X_train['Query'])

# Creating BoW for query column
X_train_query = X.transform(X_train['Query'])
X_train = X_train.drop(['Query','Unnamed: 0'],axis =1 )

from scipy.sparse import hstack
X_train = hstack((X_train,X_train_query)).tocsr()

#Finding the best 
def find_best_GBDT_classifier(X_train,Y_train,params):
    classifier = xgb.XGBClassifier()
    classifierGridSearch = RandomizedSearchCV(classifier,params,n_jobs=-1,cv =3,scoring = 'roc_auc',return_train_score=True)
    result_RS = classifierGridSearch.fit(X_train,Y_train)
    result = pd.DataFrame.from_dict(result_RS.cv_results_)
    return result,classifierGridSearch.best_params_

params = {'n_estimators':[5,10,50,75,100],'learning_rate':[0.0001,0.001,0.01,0.1,0.2,0.3]}
_cv_results_set_1,best_params_set_1 = find_best_GBDT_classifier(X_train,Y_train,params)

# Training the model with the best n_estimators and learning_rate

classifier = xgb.XGBClassifier(n_estimators=best_params_set_1.get('n_estimators'),learning_rate=best_params_set_1.get('learning_rate'))
prediction_result = classifier.fit(X_train,Y_train)

#Exporting the trained model as pickle file
pickle.dump(classifier, open('model.pkl','wb'))


#Remove the temporary files
os.remove("train_copy.csv")
os.remove("Preprocessed_X_train_copy.csv")