'''

@author: Chen Jun

'''

import numpy as np
from collections import Counter
import time


class BaseModel(object):
    
    
    def __init__(self, dataset, winsize, min_rep_gap):
        super(BaseModel, self).__init__()
        self.W = winsize
        self.dataset = dataset
        self.min_rep_gap = min_rep_gap


    def evaluate(self, topks=[1]):
        num_ticks = len(topks)
        correct = np.zeros(num_ticks, dtype=int)
        trials = 0
        accumulate_prec = np.zeros(num_ticks, dtype=float)
        accumulate_user = 0
        topk = max(topks)
        total_time = 0.
        if self.W > 0:
            for u, seq in enumerate(self.dataset.test_set):
                if u % 1000 == 0: print 'u:%d/%d' % (u, self.dataset.U)
                window = Counter(seq[:self.W])
                lastpos = {v:i for i, v in enumerate(seq[:self.W])}
                correct_user = np.zeros(num_ticks, dtype=int)
                trials_user = 0
                for i in xrange(self.W, len(seq)):
                    if seq[i] in window and len(window) > 1 and i - lastpos[seq[i]] > self.min_rep_gap:
                        start_time = time.time()
                        rankings = self._recommend(u, topk, win=window, ws=self.W, seq=seq, current=i, lastpos=lastpos)
                        total_time += time.time() - start_time
                        try: index = rankings.index(seq[i])
                        except: index = topk + 1
                        for o, r in enumerate(topks): 
                            if index < r: correct_user[o] += 1
                        trials_user += 1
                    window[seq[i-self.W]] -= 1                        
                    window[seq[i]] += 1
                    if window[seq[i-self.W]] == 0: del window[seq[i-self.W]]
                    lastpos[seq[i]] = i
                correct += correct_user
                trials += trials_user
                if trials_user > 0:
                    accumulate_prec += (correct_user * 1.0 / trials_user)
                    accumulate_user += 1
                del window, lastpos
        else:
            raise ValueError('unsupported winsize:%d' % self.W)
            
        for o, r in enumerate(topks):
            print 'Stat:\ntop-%d,\t%d/%d=%f,\t%d/%d=%f,\t%f' % (r, correct_user[o], trials_user, correct_user[o] * 1.0 / trials_user, 
                                                                         correct[o], trials, correct[o] * 1.0 / trials,
                                                                         accumulate_prec[o] / accumulate_user)
        print 'Average recommendation time: %0.5f' % (total_time / trials)
        print '\n'


    def _recommend(self, user, topk, win=None, ws=100, seq=None, current=0, lastpos=None):
        """
            override this method in the child class
        """
        raise NotImplementedError('method not implemented in base class')
