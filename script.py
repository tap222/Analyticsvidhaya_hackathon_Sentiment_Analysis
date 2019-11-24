# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:38:33 2019

@author: tamohant
"""

import os
os.chdir('C:\\Users\\tamohant\\Desktop\\my_experiment\\Analytics_Vdaya')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
subm = pd.read_csv('sample_submission.csv')

###Handling target values
train['sentiment']= train['sentiment'].astype('category')
train = pd.concat([train,pd.get_dummies(train['sentiment'], prefix='sentiment')],axis=1)
train.drop(['sentiment'],axis=1, inplace=True)

train.to_csv('train_o.csv',index=False)

train = pd.concat([train,pd.get_dummies(train['drug'])],axis=1)
train.drop(['drug'],axis=1, inplace=True)

label_cols = ['sentiment_0', 'sentiment_1','sentiment_2']
#train['none'] = 1-train[label_cols].max(axis=1)
#train.describe()

##Filling the Missing vaues

COMMENT = 'text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])

trn_term_doc, test_term_doc

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

x = trn_term_doc
test_x = test_term_doc

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True,class_weight="balanced")
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


preds = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]

submid = pd.DataFrame({'unique_hash': subm["unique_hash"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission['sentiment'] = submission[['sentiment_0', 'sentiment_1','sentiment_2']].max(axis=1)
t = submission[['sentiment_0', 'sentiment_1','sentiment_2']].idxmax(axis=1)
submission = pd.concat([submission,t],axis=1)
submission.to_csv('submission_script_1.csv', index=False)