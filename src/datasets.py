'''

@author: Chen Jun

'''

import json
from datetime import datetime
from sys import stderr
import numpy as np
import pandas as pd
import commons as com
from sklearn.preprocessing.label import LabelEncoder
from sklearn.preprocessing.data import MinMaxScaler


TIME_FORMAT = '%Y-%m-%dT%H:%M:%SZ'
TIME_FORMAT_TMALL = '%m%d'
TIME_FORMAT_TMALL_MOBILE = '%Y-%m-%d %H'
REF_DATE_TIME = datetime(year=2000,month=1,day=1)



def preprocess(inpath, name, min_events=com.WINSIZE, save2fig=True):
    ''' 
        name only supports 'go' or 'lfm'
        available for Gowalla, and Lastfm datasets 
        transform original datasets into format that can be handled by our algorithm
    '''
    print '*' * 30
    print '\tdataset:%s' % name
    print '*' * 30
    df = pd.read_csv(inpath, sep='\t', header=None, usecols=[0,1,4], error_bad_lines=False)
    n_samples_before_remove = df.shape[0]
    print '%d samples in all' % n_samples_before_remove
    df.dropna(how='any', inplace=True) # drop lines with missing values in {UserId, ItemId, Timestamp}
    n_samples_removed = n_samples_before_remove - df.shape[0]
    print >> stderr, '%d samples are removed, removal rate:%0.5f' % (n_samples_removed, n_samples_removed * 1.0 / n_samples_before_remove)
    data = df.values
    print 'Loading data finished.'
    
    behaviors = {}
    for i in xrange(data.shape[0]):
        dt = (datetime.strptime(data[i,1], TIME_FORMAT) - REF_DATE_TIME).total_seconds()
        if data[i,0] in behaviors: behaviors[data[i,0]].append([data[i,2], dt])
        else: behaviors[data[i,0]] = [[data[i,2], dt]]
    
    if name == 'lfm': _preprocess_base(behaviors, name, min_events, min_inter_gap=com.THIRTY_SECS_DUR, min_intra_gap=com.SIX_HOUR_DUR)
    elif name == 'go': _preprocess_base(behaviors, name, min_events, min_intra_gap=com.SIX_HOUR_DUR)
    else: raise Exception('wrong dataset name')


def _preprocess_base(behaviors, name, min_events, min_intra_gap=0, min_inter_gap=0):
    """
        @param min_events: minimum number of consumption behaviors a user has
        @param min_intra_gap: minimum gap between the adjacent consumption on a same item
        @param min_inter_gap: minimum gap between the adjacent consumption on any item
    """
    item_set = set()
    for u in behaviors.keys():
        v = behaviors[u]
        if len(v) <= min_events: 
            del behaviors[u]
        else: 
            v.sort(key=lambda r: r[1])  # sort history according to timestamp, early record in smaller index
            
            # remove noisy records
            i = len(v) - 1
            while i > 0:
                if v[i][1] < v[i-1][1] or (min_intra_gap > 0 and v[i][0] == v[i-1][0] and v[i][1] - v[i-1][1] < min_intra_gap): 
                    del v[i]
                    if i >= len(v): i = len(v) - 1
                elif min_inter_gap > 0 and v[i][1] - v[i-1][1] < min_inter_gap:
                    del v[i-1]
                    if i >= len(v): i = len(v) - 1
                else: i -= 1
                
            if len(v) <= min_events: 
                del behaviors[u]
            else:
                _verify_sequence(v, min_intra_gap, min_inter_gap)
                v_new = [r[0] for r in v]
                item_set |= set(v_new)
                del behaviors[u]
                behaviors[u] = v_new
    
    print '%d users left after removal.' % len(behaviors.keys())

    user_id_old = behaviors.keys()
    item_id_old = list(item_set)
    user_id_new = LabelEncoder().fit_transform(user_id_old)
    item_id_new = LabelEncoder().fit_transform(item_id_old)
    user_id_map = {v:user_id_new[i] for i, v in enumerate(user_id_old)}
    item_id_map = {v:item_id_new[i] for i, v in enumerate(item_id_old)}
    
    behaviors_new = [0] * len(user_id_new)
    for u, l in behaviors.iteritems():
        assert(len(l) > min_events)
        for i in xrange(len(l)): l[i] = item_id_map[l[i]]
        behaviors_new[user_id_map[u]] = l
             
    behaviors_new = np.array(behaviors_new)
    behaviors_new.dump('..\\data\\behaviors_%s.array' % name)
    
    with open('..\\data\\user_id_%s.map' % name, 'wb') as outfile:
        json.dump(user_id_map, outfile)
    with open('..\\data\\item_id_%s.map' % name, 'wb') as outfile:
        json.dump(item_id_map, outfile)
    
    print 'Dumping behavior data finished.'


def _verify_sequence(v, min_intra_gap=0, min_inter_gap=0):
    flag = 0
    for i in xrange(1, len(v)):
        if v[i][1] < v[i-1][1]: # records must occur in time ascending order
            flag = 1
            break
        if min_inter_gap > 0 and v[i][1] - v[i-1][1] < min_inter_gap:
            flag = 2
            break
        if min_intra_gap > 0 and v[i][0] == v[i-1][0] and v[i][1] - v[i-1][1] < min_intra_gap:
            flag = 3
            break
    if flag > 0:
        print >> stderr, 'Error flag: %d' % flag
        raise Exception('Error occured in verification.')


class Dataset(object):
    
    
    def __init__(self, name, winsize, training_ratio=com.TRAINING_RATIO, loadraw=True, loadfeature=False):
        self.name = name
        print 'Loading dataset: %s' % name
        print 'Winsize:%d, Training ratio:%0.3f' % (winsize, training_ratio)
        if loadraw:
            self._load_rawdata()
            self._train_test_split(winsize, training_ratio)
        if loadfeature:
            self._load_featuredata()
    
    
    def _load_rawdata(self):
        print 'Loading raw data ...'
        self.behaviors = np.load('..\\data\\behaviors_%s.array' % self.name)
        with open('..\\data\\user_id_%s.map' % self.name, 'rb') as infile:
            self.user_id_map = json.load(infile)
            self.U = len(self.user_id_map)
        with open('..\\data\\item_id_%s.map' % self.name, 'rb') as infile:
            self.item_id_map = json.load(infile)
            self.I = len(self.item_id_map)
        print 'Loading raw data finished.'
    
    
    def _load_featuredata(self):
        self.item_popularity = np.load('..\\data\\feature\\feature_item_popularity_%s.array' % self.name)
        self.item_popularity = MinMaxScaler().fit_transform(np.log(1.0+self.item_popularity))
        self.item_reconsume_ratio = np.load('..\\data\\feature\\feature_item_reconsume_ratio_%s.array' % self.name)
        self.user_reconsume_ratio = np.load('..\\data\\feature\\feature_user_reconsume_ratio_%s.array' % self.name)
    
    
    def _train_test_split(self, winsize, training_ratio):
        print 'Splitting training and test sets ...'
        self.train_set = [[] for _ in xrange(self.U)]
        self.test_set  = [[] for _ in xrange(self.U)]
        for u, seq in enumerate(self.behaviors):
            n_test_samples = int((len(seq)-winsize)*(1.0-training_ratio))
            assert(n_test_samples >= 0 and len(seq) > winsize)
            self.train_set[u] = seq[:-n_test_samples] if n_test_samples > 0 else seq
            self.test_set[u]  = seq[-winsize-n_test_samples:]
            assert(len(self.train_set[u]) >= winsize and len(self.test_set[u]) >= winsize)
        print 'Splitting training and test sets finished.'
            
            
    def print_dataset_statistics(self):
        print '+' * 30
        print '\tuser count: %d' % self.U
        print '\titem count: %d' % self.I
        count = 0
        for seq in self.behaviors:
            count += len(seq)
        print '\trecord count: %d' % count
        print '\tr/u: %0.2f' % (count*1.0/self.U)
        print '\tr/i: %0.2f' % (count*1.0/self.I)
        print '+' * 30
    
    
    def extract_features(self, train_only=True, winsize=100):
        item_popularity = np.zeros(self.I)
        item_reconsume_ratio = np.zeros(self.I)
        user_reconsume_ratio = np.zeros(self.U)
        ds = self.train_set if train_only else self.behaviors
        
        for seq in ds:
            for i in seq:
                item_popularity[i] += 1
        item_popularity.dump('..\\data\\feature\\feature_item_popularity_%s.array' % self.name)
        
        print 'Extracting item reconsume ratio with winsize: %d' % winsize
        item_observations = np.zeros(self.I,dtype=int)
        user_observations = np.zeros(self.U,dtype=int)
        for u, seq in enumerate(ds):
            window = seq[:winsize]
            for i in seq[winsize:]:
                if i in window: 
                    item_reconsume_ratio[i] += 1
                    user_reconsume_ratio[u] += 1
                item_observations[i] += 1
                user_observations[u] += 1
                del window[0]
                window.append(i)
        item_reconsume_ratio /= (item_observations + 1.0e-10)
        item_reconsume_ratio.dump('..\\data\\feature\\feature_item_reconsume_ratio_%s.array' % self.name)
        user_reconsume_ratio /= (user_observations + 1.0e-10)
        user_reconsume_ratio.dump('..\\data\\feature\\feature_user_reconsume_ratio_%s.array' % self.name)
        

if __name__ == '__main__':
    
    name = 'go'
    preprocess('#Your path to the Gowalla file#/Gowalla_totalCheckins.txt', name)
    dataset = Dataset(name, com.WINSIZE, training_ratio=com.TRAINING_RATIO)
    dataset.print_dataset_statistics()
    dataset.extract_features(train_only=True, winsize=com.WINSIZE)
    
    name = 'lfm'
    preprocess('#Your path to the Lastfm-1K file#/userid-timestamp-artid-artname-traid-traname.tsv', name)
    dataset = Dataset(name, com.WINSIZE, training_ratio=com.TRAINING_RATIO)
    dataset.print_dataset_statistics()
    dataset.extract_features(train_only=True, winsize=com.WINSIZE)
    
    
    