#!/Users/fedorsimonov/anaconda/bin/python
# -*- coding: utf-8 -*-

import random
import numpy as np
from russe.w2v.utils import load_vectors, load_word2vec_format
from collections import Counter, OrderedDict
import pandas as pd
import re
from sys import stdout, stderr
import sys
import gzip
import pymorphy2
import pymorphy2.tokenizers
import re
from sklearn.base import BaseEstimator, ClassifierMixin

class MaxProbClf(BaseEstimator, ClassifierMixin):
    def __init__(self, sense_df, vw, vc):
        self.sense_df = sense_df
        self.vw = vw
        self.vc = vc
           
    def logprob(self, ctx, vsense):
        vc = self.vc
        vctx = vc.syn0norm[vc.vocab[ctx].index]
        return np.log(1.0 / (1.0 + np.exp(-np.dot(vctx,vsense))))
        
    def fit(self, X, y):
        pass
    
    
    def get_senses(self, homo, cache):
        if homo not in cache:
            sense_df = self.sense_df
            senses, svecs = [], []
            contexts = []
            homos = homo.split('|')
            for i, row in sense_df[sense_df.word.isin(homos)].iterrows():
                svec = build_svec(row.word, row.context, self.vw)
                sense = row.id #'%s_%d' % (row.word,row.id) if '|' not in homo else row.word
                context = row.context
                if len(svec):
                    senses.append(sense)
                    svecs.append(svec)
                    contexts.append(context)
            cache[homo] = (senses, svecs, contexts)
        return cache[homo]
    
    
    def predict(self, cdf, gamma):
        y = []
        cache = dict()
        for i in xrange(len(cdf)):
            l, homo, r, id = takeI(cdf, i)
            senses, svecs, contexts = self.get_senses(homo,cache)
            cntSenses = len(senses)
            s = [0.0 for n in xrange(cntSenses)]
            words = [word for word in l+r if word in self.vc and word in self.vw]
            isRelevantCtx = False
            #print ' '.join(l + list(homo) + r)
            maxLp = 0
            for word in words:
                lp = [self.logprob(word, svecs[j]) for j in xrange(cntSenses)]
                
                for i in xrange(cntSenses):
                    for j in xrange(i + 1, cntSenses):
                        diff = lp[i] - lp[j]
                        if np.abs(diff) > maxLp:
                            maxLp = np.abs(diff)
                            
            for word in words:
                lp = [self.logprob(word, svecs[j]) for j in xrange(cntSenses)]
                
                vc = self.vc
                vctx = vc.syn0norm[vc.vocab[word].index]
                dots = [np.dot(vctx, svecs[j]) for j in xrange(cntSenses)]
                ans = []
                for j in range(len(lp)):
                    ans.append('P(%s|%s_%d)=%.4f' % (word, homo, j, lp[j]))
                #print ' '.join(ans)
                for i in xrange(cntSenses):
                    for j in xrange(i + 1, cntSenses):
                        diff = lp[i] - lp[j]
                        if np.abs(diff) > (maxLp * gamma):
                            isRelevantCtx = True
                if isRelevantCtx:
                    for j in xrange(cntSenses):
                        s[j] += lp[j]
            #print s
            ind = np.argsort(s)
            if len(senses) != 0:
                sense = senses[ind[-1]]
                #print 'Predicted sense - %d\n--------------' % sense
                context = contexts[ind[-1]]
                y.append((sense, id))
                cdf.loc[id, 'predict_sense_ids'] = int(sense)
                cdf.loc[id, 'predict_related'] = context
                if id % 1000 == 0:
                    print "%.2f%% completed" % (float(id) / float(len(cdf)))
        return y

# Implement a function that normalizes each row of a matrix to have unit length
def normalizeRows(x):
    mark = False
    if len(x.shape) == 1:
        mark = True
        x = x.reshape((1, x.shape[0]))
    r, c = x.shape
    t = np.sqrt(np.sum(x**2, axis = 1))
    x /= t.reshape(t.shape[0], 1)
    if (mark):
        x = x.reshape((x.shape[1],))
    return x

def takeI(cdf, i):
    context = cdf['context'][i]
    borders = cdf['target_position'][i].split(',')
    borders = [int(x) for x in borders]
    L = context[:borders[0]].split()
    word = cdf['target'][i]
    R = context[borders[1]:].split()
    return L, word, R, i

_morph = pymorphy2.MorphAnalyzer()
rr = re.compile(u'[a-zа-яё]')

def tokenize(sent):
    u"""
    Simple tokenizer - lowercases everything and tokenizes using pymorphy2, removes all punctuation tokens.
    >>> ' * '.join( tokenize(u'Для') ) == u'для'
    True
    >>> ' * '.join(tokenize(u'Для стрельбы-стрельбы, из Арбалета? использовались болты — особые Арбалетные стрелы!')) == u'для * стрельбы-стрельбы * из * арбалета * использовались * болты * особые * арбалетные * стрелы'
    True
    """

    sent = sent.lower()
    words = [x for x in pymorphy2.tokenizers.simple_word_tokenize(sent) if rr.search(x) is not None]
    return words


def has_normal_form(word, normal):
    u"""
    Returns whether the word has specified normal form.
    >>> has_normal_form(u'лейки',u'лейка')
    True
    >>> has_normal_form(u'лейки',u'лейки')
    False
    >>> has_normal_form(u'берег',u'беречь')
    True
    >>> has_normal_form(u'берег',u'берег')
    True
    """
    for p in _morph.parse(word):
        if p.normal_form==normal:
            return True
    return False

def parse_neigh(line, K=None, return_dists=False):
    neigh = [(x.split(':')[0],float(x.split(':')[1])) for x in (line.replace(" ", "")).split(',')]
    neigh = sorted(neigh, key=lambda x:x[1],reverse=True)
    neigh_part = neigh[:K] if K is not None else neigh
    if return_dists:
        return neigh_part
    else:
        return [x[0] for x in neigh_part]

def build_svec(word, neigh, vw):
    neigh = parse_neigh(neigh, return_dists=False)
    nlog = []
    selected_neigh = []
    for x in neigh:
        if has_normal_form(x,word):
            nlog.append('*%s' % x)
            continue
        if x not in vw:
            nlog.append('-%s' % x)
            continue
        sense_cnt = len(idf[idf.word==x])
        if (sense_cnt>1):
            nlog.append('?%ds?%s' % (sense_cnt, x) )
            continue

        nlog.append('+%s' % x)
        selected_neigh.append(x.strip())
    #print >> stderr, '%s (%s)' % (word, ' '.join(nlog))
    if len(selected_neigh) != 0:
        svec = np.average(vw[selected_neigh],axis=0)
        #print >> stderr, ' '.join(['%s %.2lf' % p for p in vw.most_similar(positive=[svec])])
    else:
        svec = np.zeros(vw.syn0[0].shape)
    return svec

def delete_substring(name, str):
	i = name.rfind(str)
	if i != -1:
		return name[0:i]
	else:
		return name

def name_of_file(name):
	i = name.rfind('/')
	if i != -1:
		return name[i + 1 :]
	else:
		return name

if len (sys.argv) < 4:
	print ("Error. Too few arguments.")
	print ("Format: python wsd.py test_set.csv.gz inventory.csv.gz word_vectors.bin [context_vectors.w2v]")
	sys.exit (1)

if len (sys.argv) > 5:
	print ("Error. Too many arguments.")
	print ("Format: python wsd.py test_set.csv.gz inventory.csv.gz word_vectors.bin [context_vectors.w2v]")
	sys.exit (1)

FSET = "{}".format(sys.argv[1])
df = pd.read_csv(FSET, sep = '\t', encoding='utf8')
df.loc[:, 'predict_sense_ids'] = np.array([0] * len(df),dtype='int32')
print ("Test set loaded")
FINV = "{}".format(sys.argv[2])
idf = pd.read_csv(FINV, sep = '\t', encoding='utf8', names=['word', 'id', 'context'])
if (idf.loc[0, 'word'] == 'word'):
	idf = idf.loc[1:]
print ("Inventory loaded")

word_vectors = load_vectors("{}".format(sys.argv[3]))
print ("Word vectors loaded")
if len (sys.argv) == 5:
	context_vectors = load_vectors("{}".format(sys.argv[4]))
	print ("Context vectors loaded")
else:
	context_vectors = word_vectors

clf = MaxProbClf(idf, word_vectors, context_vectors)
clf.predict(df, 2
    )
print "%.2f%% completed" % float(100)
print ("Senses predicted")

FRES = "{}{}{}{}{}{}".format(delete_substring(sys.argv[1], '.csv'), '_', name_of_file(delete_substring(sys.argv[2], '.csv')), '_', name_of_file(delete_substring(delete_substring(sys.argv[3], '.bin'), '.w2v')), "_predicted.csv")
df.to_csv(FRES, sep = '\t', encoding='utf8')
print ("Results are in {}".format(FRES))



