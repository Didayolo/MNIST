# -*- coding: utf-8 -*-
 
# Math libs
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
import seaborn as sns
import sklearn

# Feature extraction
from ast import literal_eval
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
import string
import unicodedata

# Feature selection
#from genetic_selection import GeneticSelectionCV
from sklearn import feature_selection
#from skfeature.function.similarity_based import fisher_score

# Classifiers
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
import xgboost
from sklearn.neural_network import MLPClassifier
#import mord

# Misc
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
import pafy
from multiprocessing.pool import ThreadPool

# Map additions
def lmap(f,l): return list(map(f,l))
def amap(f,l): return np.array(lmap(f,l))

def strip_accents_unicode(s):
    try:
        s = unicode(s, 'utf-8')
    except NameError:  # unicode is a default on python 3
        pass
    s = unicodedata.normalize('NFD', s)
    s = s.encode('ascii', 'ignore')
    s = s.decode("utf-8")
    return str(s)

def clean_str(sentence, stem=True):
    sentence = strip_accents_unicode(sentence)
    english_stopwords = set(nltk.corpus.stopwords.words('english'))
    punctuation = set(string.punctuation)
    punctuation.update(["``", "`", "..."])
    words = list(filter(lambda t:t.isalpha(), nltk.word_tokenize(sentence)))
    tags = lmap(lambda x:x[1], nltk.pos_tag(words))
    if stem :
        stemmer = nltk.stem.SnowballStemmer('english').stem
        words = lmap(stemmer, words)
    return words, tags

class FeatureExtractor(TransformerMixin):
    def __init__(self, stem_tfidf=True):
        self.stem_tfidf = stem_tfidf
        self.dummies_extractors = {}
        
    def process_dummies(self, col, parser=None, fit=False, add=False): # called with either fit or add to True
        if fit:
            (self.dummies_extractors)[col] = DictVectorizer(sparse=False, dtype=np.int64)
        to_dummy = self.data[col]
        if parser is not None: # list mode
            to_dummy = to_dummy.apply(parser).apply(pd.Series).stack()
        dummies = pd.get_dummies(to_dummy, prefix=col, prefix_sep='_').sum(level=0)
        if add:
            dict_ = dummies.to_dict('records')
        else:
            dict_ = dummies[:1].to_dict('records')
        if fit and add:
            dummies_wdrop = (self.dummies_extractors[col]).fit_transform(dict_)
        elif fit: # only
            (self.dummies_extractors)[col].fit(dict_)
        else: # add only
            dummies_wdrop = (self.dummies_extractors[col]).transform(dict_)
        if add:
            self.out = pd.concat([self.out, pd.DataFrame(dummies_wdrop).add_prefix(col+'_')], axis=1, ignore_index=True)
    
    def process_tfidf(self, col, stem=None, fit=False, add=False, tags=True):
        if stem is None: stem = self.stem_tfidf
        if fit:
            (self.dummies_extractors)['col'] = TfidfVectorizer(analyzer='word')
        words_tags = map(lambda x: clean_str(x, stem=stem), self.data[col])
        statement_preprocess = lmap(lambda wt: ' '.join(wt[1] if tags else wt[0]), words_tags)
        if fit and add:
            transformed = (self.dummies_extractors)['col'].fit_transform(statement_preprocess)
        elif fit: # only
            (self.dummies_extractors)['col'].fit(statement_preprocess)
        else: # add only
            transformed = (self.dummies_extractors)['col'].transform(statement_preprocess)
        if add:
            tfidfdf = pd.DataFrame(transformed.todense()).add_prefix(col+('_t' if tags else '_w'))
            self.out = pd.concat([self.out, tfidfdf], axis=1, ignore_index=True)
    
    def _fit_transform(self, X_df, y=None, fit=True, transform=True):
        self.data = X_df.reset_index(drop=True, inplace=False) # dataframe, reindex because split does shit! Hours of debuging this!
        
        if transform:
            self.out = self.data[[]].copy()
            self.out['aff'] = 1 # Make any linear regression affine
            #self.out['rl'] = self.data['truth'] # cheat feature
        
        self.process_tfidf('description', fit=fit, add=transform)
        #self.process_tfidf('description', tags=True, fit=fit, add=transform)
        self.process_tfidf('title', fit=fit, add=transform)
        #self.process_tfidf('title', tags=True, fit=fit, add=transform)
        self.process_dummies('category', fit=fit, add=transform)
        #self.process_dummies('keywords', parser=lambda x:x, fit=fit, add=transform)
        
        if transform:
            #self.out = pd.concat([self.out, self.data.rating], axis=1, ignore_index=True)
            #self.out = pd.concat([self.out, self.data.viewcount], axis=1, ignore_index=True)
            self.out = pd.concat([self.out, self.data.length], axis=1, ignore_index=True)
            self.out.fillna(0, inplace=True)
    
    def fit(self, X_df, y=None):
        self._fit_transform(X_df, y, fit=True, transform=False)
        return self

    def fit_transform(self, X_df, y=None):
        self._fit_transform(X_df, y, fit=True, transform=True)
        return self.out.values

    def transform(self, X_df):
        self._fit_transform(X_df, y=None, fit=False, transform=True)
        return self.out.values