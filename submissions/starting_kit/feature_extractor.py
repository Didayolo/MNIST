import nltk
import numpy as np
import pandas as pd
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer


def lmap(f, l):
    return list(map(f, l))


def amap(f, l):
    return np.array(lmap(f, l))


class FeatureExtractor:

    def strip_accents_unicode(self, s):
        try:
            s = unicode(s, 'utf-8')
        except NameError:  # unicode is a default on python 3
            pass
        s = unicodedata.normalize('NFD', s)
        s = s.encode('ascii', 'ignore')
        s = s.decode("utf-8")
        return str(s)

    def clean_str(self, sentence, stem=True):
        sentence = self.strip_accents_unicode(sentence)
        words = list(filter(lambda t: t.isalpha(),
                            nltk.word_tokenize(sentence)))
        if stem:
            stemmer = nltk.stem.SnowballStemmer('english').stem
            words = lmap(stemmer, words)
        return words

    def __init__(self, stem_tfidf=True):
        self.stem_tfidf = stem_tfidf
        self.dummies_extractors = {}

    # called with either fit or add to True
    def process_dummies(self, col, parser=None, fit=False, add=False):
        if fit:
            (self.dummies_extractors)[col] = DictVectorizer(
                sparse=False, dtype=np.int64)
        to_dummy = self.data[col]
        if parser is not None:  # list mode
            to_dummy = to_dummy.apply(parser).apply(pd.Series).stack()

        dummies = pd.get_dummies(to_dummy, prefix=col,
                                 prefix_sep='_').sum(level=0)

        if fit and add:
            dict_ = dummies.to_dict('records')
            dummies_wdrop = (self.dummies_extractors[col]).fit_transform(dict_)
            self.out = pd.concat([self.out, pd.DataFrame(
                dummies_wdrop).add_prefix(col + '_')],
                axis=1, ignore_index=True)

        elif fit:  # only
            dict_ = dummies[:1].to_dict('records')
            (self.dummies_extractors)[col].fit(dict_)

        else:  # add only
            dict_ = dummies.to_dict('records')
            dummies_wdrop = (self.dummies_extractors[col]).transform(dict_)
            self.out = pd.concat([self.out, pd.DataFrame(
                dummies_wdrop).add_prefix(col + '_')],
                axis=1, ignore_index=True)

    def process_tfidf(self, col, stem=None, fit=False, add=False):
        if stem is None:
            stem = self.stem_tfidf

        words = map(lambda x: self.clean_str(x, stem=stem), self.data[col])
        statement_preprocess = lmap(lambda w: ' '.join(w), words)

        if fit and add:
            (self.dummies_extractors)[col] = TfidfVectorizer(analyzer='word')
            transformed = (self.dummies_extractors)[
                col].fit_transform(statement_preprocess)
            tfidfdf = pd.DataFrame(transformed.todense()).add_prefix(col)
            self.out = pd.concat([self.out, tfidfdf],
                                 axis=1, ignore_index=True)

        elif fit:  # only
            (self.dummies_extractors)[col] = TfidfVectorizer(analyzer='word')
            (self.dummies_extractors)[col].fit(statement_preprocess)

        else:  # add only
            transformed = (self.dummies_extractors)[
                col].transform(statement_preprocess)
            tfidfdf = pd.DataFrame(transformed.todense()).add_prefix(col)
            self.out = pd.concat([self.out, tfidfdf],
                                 axis=1, ignore_index=True)

    def _fit(self, y=None, fit=True, transform=True):
        self.process_tfidf('description', fit=fit, add=transform)
        self.process_tfidf('title', fit=fit, add=transform)
        self.process_dummies('category', fit=fit, add=transform)

    def _fit_transform(self, X_df, y=None, fit=True, transform=True):

        self.out = self.data[[]].copy()
        self.out['aff'] = 1  # Make any linear regression affine

        self._fit(y, fit, transform)

        self.out = pd.concat([self.out, self.data.length],
                             axis=1, ignore_index=True)
        self.out.fillna(0, inplace=True)

    def fit(self, X_df, y=None):
        self.data = X_df.reset_index(drop=True, inplace=False)
        self._fit(y, fit=True, transform=False)
        return self

    def fit_transform(self, X_df, y=None):
        self.data = X_df.reset_index(drop=True, inplace=False)
        self._fit_transform(X_df, y, fit=True, transform=True)
        return self.out.values

    def transform(self, X_df):
        self.data = X_df.reset_index(drop=True, inplace=False)
        self._fit_transform(X_df, y=None, fit=False, transform=True)
        return self.out.values
