# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 18:50:06 2022

@author: Alex
"""

#%%
import re,string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import seaborn as sns
import pickle
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
from tqdm import tqdm
from sklearn import utils
tqdm.pandas(desc="progress-bar")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor

from sklearn.manifold import MDS

from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold

import pandas as pd
import os
import optuna
from functools import partial

from gensim.models import Word2Vec,LdaMulticore, TfidfModel
from gensim import corpora


from gensim.models.doc2vec import Doc2Vec, TaggedDocument


import numpy as np

# machine learning
import tensorflow as tf
from tensorflow import keras

#%%
# Load file
currdir = os.getcwd()
os.chdir(r"C:\Users\Alex\Desktop\Northwestern MSDS\MSDS 453\Final Project")
# read data

df = pd.read_csv("winemag-data_first150k.csv", index_col=(0))

df.isna().sum()

#%% Cleaning Data

df.dropna(inplace = True, subset=("price", "region_1", "variety")) #dropping rows that does not have a price or region
df.drop_duplicates(subset=("description"), inplace= True)
df.reset_index(drop = True, inplace= True)

# Look at wine varieties
variety_df = df.groupby('variety').filter(lambda x: len(x) > 500) # drop
varieties = variety_df['variety'].value_counts().index.tolist()
fig, ax = plt.subplots(figsize = (25, 10))
sns.countplot(x = variety_df['variety'], order = varieties, ax = ax).set(title='Types of Wine by Frequency in Data')

# drop wine varieties with not enough reviews
df_variety = df[df['variety'].isin(varieties)]

col = df.columns

#%% clean doc

def clean_doc(doc): 
    #split document into individual words
    tokens=doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 4]
    #lowercase all words
    tokens = [word.lower() for word in tokens]
    # word stemming    
    #ps=PorterStemmer()
    #tokens=[ps.stem(word) for word in tokens]
    # filter out stop words
    stop_words = stopwords.words('english')
    stop_words.extend(["flavor","flavors", "drink", "fruit", "finish", "tanning"]) #extending stop_words after reviewing WordClouds
    tokens = [w for w in tokens if not w in stop_words]         
    
    return tokens

df_variety["tokens"] = df_variety["description"].apply(lambda x: clean_doc(x)) #create tokens
df_variety["cleaned"] = df_variety["tokens"].apply(lambda x: str(" ".join(x))) #stitching

#%% TFIDF

doc = df_variety["cleaned"].tolist()

#TF-IDF
Tfidf=TfidfVectorizer(ngram_range=(1,3), max_features=300)

TFIDF_matrix=Tfidf.fit_transform(doc)

df_tfidf=pd.DataFrame(TFIDF_matrix.toarray(), columns=Tfidf.get_feature_names())


weight = TFIDF_matrix.toarray()
X = weight

#%% Setup for Machine Learning Methods

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# encode categorical classes
target_var = df_variety['variety']
label_encoder = LabelEncoder()
target = np.array(label_encoder.fit_transform(target_var))

# split into train test split
X_train, X_test, y_train, y_test = train_test_split(df_tfidf, target, test_size=0.2, random_state=42)

#%% Random Forest Classifier

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier()

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_test, y_pred))

# cross validation
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
# n_scores = cross_val_score(rf, X_train, y_train, scoring='accuracy', cv=cv, error_score='raise')
# print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

#%% Gradient Boosting Classifier

from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()

gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_test, y_pred))

# cross validation
# cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=42)
# n_scores = cross_val_score(gb, X_train, y_train, scoring='accuracy', cv=cv, error_score='raise')
# print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


#%% XGBoost

from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_test, y_pred))

# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
# n_scores = cross_val_score(xgb, X_train, y_train, scoring='accuracy', cv=cv, error_score='raise')
# print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

#%% Neural Network Model

def plot_confusion_matrix(y_true,y_pred):
    """If you prefer color and a colorbar"""
    fig = plt.figure(figsize=(5,5))
    matrix = confusion_matrix(y_true, y_pred)
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    #plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.xlabel("Predicted Classes")
    plt.ylabel("Actual Classes")
    # plt.savefig("confusion_matrix_plot_mnist", tight_layout=False)
    plt.show()


model1 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[300]),
    keras.layers.Dense(150, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(28, activation="softmax")])

model1.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

model1.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

evaluation1 = model1.evaluate(X_test, y_test)

y_pred = model1.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
plot_confusion_matrix(y_test, y_pred)