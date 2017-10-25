'''

@author: Chen Jun

'''
import numpy as np
import random
import json
import os
from numpy.random.mtrand import shuffle
from collections import Counter
from datasets import Dataset
import commons as com
from src.base import BaseModel


#familarity function
familiarity_func = lambda x: 2.0 / (1.0 + np.exp(-0.1*x)) - 1.0


#recency function
recency_func = lambda x: 1.0 / (x + 1.0)


class RepeaterTSPPR(BaseModel):
    
    
    def __init__(self, dataset, winsize, min_rep_gap=com.MIN_REP_GAP, negative_samples=com.NEG_SAMPLE):
        super(RepeaterTSPPR, self).__init__(dataset, winsize, min_rep_gap)
        self.W = winsize
        self.dataset= dataset
        self.neg_sample = negative_samples
        self.min_rep_gap = min_rep_gap
        feature_path = '../data/feature/feature_repeater_%s_ws%d_ns%d_gap%d.json' % (dataset.name, self.W, self.neg_sample, self.min_rep_gap)
        if os.path.isfile(feature_path): self._load_training_feature(feature_path)
        else: self._extract_training_feature(feature_path, self.neg_sample)
    
    
    def _extract_training_feature(self, feature_path, negative_samples):
        """
            sample positive&negative pairs, extract behavioral features.
        """
        assert(negative_samples > 0 and negative_samples <= self.W and self.W > 0)
        print 'Feature extraction started ...'

        self.training_samples = [0] * self.dataset.U
        feature_file = open(feature_path, 'wb')
        for u, seq in enumerate(self.dataset.train_set):
            if u % 1000 == 0: print 'Extracting u:%d/%d' % (u, self.dataset.U)
            samples_u = []
            window = Counter(seq[:self.W])
            lastpos = {v:i for i, v in enumerate(seq[:self.W])}
            for i in xrange(self.W, len(seq)):
                iid = seq[i]
                if iid in window and len(window) > 1 and i - lastpos[iid] > self.min_rep_gap:
                    #candidates
                    elems = window.keys()
                    elems.remove(iid)
                    if len(elems) > 0:
                        samples = []
                        # 1 positive sample
                        fp = [self.dataset.item_popularity[iid],           # item popularity
                              self.dataset.item_reconsume_ratio[iid],      # item reconsume ratio
                              recency_func(i-lastpos[iid]-1),              # recency
                              window[iid] * 1.0 / self.W]                  # dynamic familiarity
                        samples.append([fp, iid])
                        
                        # some negative samples, at most neg_sample
                        shuffle(elems)
                        for _ in xrange(negative_samples):
                            if len(elems) == 0: break
                            else:
                                niid = random.choice(elems)
                                elems.remove(niid)
                                fn = [self.dataset.item_popularity[niid],            # item popularity
                                      self.dataset.item_reconsume_ratio[niid],       # item reconsume ratio
                                      recency_func(i-lastpos[niid]-1),               # recency
                                      window[niid] * 1.0 / self.W]                   # dynamic familiarity
                                samples.append([fn, niid])
                        samples_u.append(samples)
                window[seq[i-self.W]] -= 1                        
                window[iid] += 1
                if window[seq[i-self.W]] == 0: del window[seq[i-self.W]]
                #assert(sum(window.values()) == self.W)
                lastpos[iid] = i
            del window, lastpos
            json.dump([u, samples_u], feature_file)
            feature_file.write('\n')
            feature_file.flush()
            self.training_samples[u] = samples_u
        feature_file.close()
        
        print 'Feature extraction finished.'
        
        
    def _load_training_feature(self, feature_path):
        print 'loading training features ... '
        assert(self.dataset.U > 0)
        self.training_samples = [0] * self.dataset.U
        with open(feature_path, 'rb') as infile:
            for line in infile:
                samples = json.loads(line)
                self.training_samples[samples[0]] = samples[1]
        print 'loading feature finished.'


    def train(self, K=5, L=4, reg_lambda=0.1, reg_gamma=0.1, max_iters=int(1e6), alpha=0.01, tol=1e-4, ratio=0.01, small_batch=int(1e5)):
        """
            @param K: dimensions of latent features
            @param L: dimensions of behavioral features
            @param reg_lambda: regularization parameter for matrix A
            @param reg_gamma: regularization parameter for user and item latent features
            @param max_iters: maximum number of epochs
            @param alpha: learning rate
            @param tol: tolerance of objective function value difference between adjacent check points
            @param small_batch: small batch size for check point
        """
        self.K, self.L = K, L
        #initialize
        print 'Model training started ... '
        print 'Training parameters:'
        print '\tK=%d,\tL=%d,\tlambda=%f,\tgamma=%f,\tmax_iters=%d,\talpha=%f.' % (K, L, reg_lambda, reg_gamma, max_iters, alpha)
        self.umat = np.random.normal(0, reg_gamma, (self.dataset.U, self.K))
        self.amat = np.random.normal(0, reg_lambda, (self.dataset.U, self.K, self.L))
        self.vmat = np.random.normal(0, reg_gamma, (self.dataset.I, self.K))
        print 'parameter initialization finished.'
         
        user_keys = range(self.dataset.U)
        triple_count = self._get_training_triples_count(ratio=ratio)
        print 'Computing triple count: %d' % triple_count
        prev_avg_diff = -1e10
        for it in xrange(max_iters):
            # randomly select user u
            while True:
                u = random.choice(user_keys)
                if len(self.training_samples[u]) > 0: break
            # randomly select a positive sample for u
            samples = random.choice(self.training_samples[u])
            pos_sample = samples[0]
            # randomly select a negative sample corresponding to the former positive one
            neg_sample = samples[random.randint(1, len(samples) - 1)]
            
            delt_f = np.array(pos_sample[0]) - np.array(neg_sample[0]) # f_i - f_j
            
            i, j = pos_sample[1], neg_sample[1]
            val = self.vmat[i] - self.vmat[j] + self.amat[u].dot(delt_f) # v_i-v_j + A^T(f_i-f_j)
            rest_prob = 1. - 1. / (1. + np.exp(-self.umat[u].dot(val)))
            u_vec_tmp  = (1 - alpha * reg_gamma)  * self.umat[u] + alpha * rest_prob * val
            vi_vec_tmp = (1 - alpha * reg_gamma)  * self.vmat[i] + alpha * rest_prob * self.umat[u]
            vj_vec_tmp = (1 - alpha * reg_gamma)  * self.vmat[j] - alpha * rest_prob * self.umat[u]
            amat_tmp   = (1 - alpha * reg_lambda) * self.amat[u] + alpha * rest_prob * np.outer(self.umat[u], delt_f)
             
            if it % small_batch == 0:
                log_LL = self._get_LogLL(reg_lambda, reg_gamma,ratio=ratio)
                avg_diff = -np.log(1.0/np.exp(log_LL/triple_count)-1.0)
                print 'iter:%d,\tLL:%0.5f,\tAvgLL:%0.5f,\tAvgDiff:%0.5f' % (it, log_LL, log_LL/triple_count, avg_diff)
                if avg_diff - prev_avg_diff < tol: break
                else: prev_avg_diff = avg_diff
                print '\t', rest_prob
                print '\t', self.amat[10,0,:]
                print '\t', self.umat[10,:5]
                print '\t', self.vmat[10,:5]
                 
            self.umat[u,:] = u_vec_tmp
            self.amat[u], amat_tmp = amat_tmp, self.amat[u]
            self.vmat[i,:] = vi_vec_tmp
            self.vmat[j,:] = vj_vec_tmp
 
        print 'Model training finished.'
    
    
    def _get_training_triples_count(self, ratio=1.0):
        ''' get the number of training triples '''
        count = 0
        for u_samples in self.training_samples:
            uplimit = (int)(len(u_samples) * ratio)
            for samples in u_samples[:uplimit]:
                count += len(samples)
            count -= uplimit
        return count
    
    
    def _get_LogLL(self, reg_lambda=0.1, reg_gamma=0.1, ratio=1.0):
        log_LL = 0.0
        for u, u_samples in enumerate(self.training_samples):
            u_vec = self.umat[u]
            uplimit = (int)(len(u_samples) * ratio)
            for samples in u_samples[:uplimit]:
                vi_vec = self.vmat[samples[0][1]] #latent vector of item i
                fi_vec = np.array(samples[0][0])  #temporal feature vector of item i
                for j in xrange(1,len(samples)):
                    vj_vec = self.vmat[samples[j][1]] #latent vector of item j
                    fj_vec = np.array(samples[j][0])  #temporal feature vector of item j
                    log_LL -= np.log(1.0 + np.exp(- u_vec.dot((vi_vec - vj_vec) + self.amat[u].dot(fi_vec - fj_vec))))
        return log_LL


    def _recommend(self, user, topk, win=None, ws=100, seq=None, current=0, lastpos=None):
        ks = [k for k in win if current - lastpos[k] > self.min_rep_gap]
        probs = np.zeros(len(ks),dtype=float)
        for i, iid in enumerate(ks):
            f = [self.dataset.item_popularity[iid],
                 self.dataset.item_reconsume_ratio[iid],
                 recency_func(current-lastpos[iid]-1),
                 win[iid] * 1.0 / self.W]
            probs[i] = self.umat[user].dot((self.vmat[iid] + self.amat[user].dot(f)))
        return [ks[i] for i in np.argsort(-probs)[:topk]]

    
if __name__ == '__main__':
    
    topks=[1,5,10]
    
#     name = 'go'
#     dataset = Dataset(name, com.WINSIZE, training_ratio=com.TRAINING_RATIO, loadraw=True, loadfeature=True)
#     model = RepeaterTSPPR(dataset, com.WINSIZE, min_rep_gap=com.MIN_REP_GAP, negative_samples=com.NEG_SAMPLE)
#     model.train(K=40, reg_lambda=0.01, reg_gamma=0.05, max_iters=int(1e8), alpha=0.02, tol=1e-4, ratio=0.1, small_batch=int(2e5))
#     model.evaluate(topks)
    
    name = 'lfm'
    dataset = Dataset(name, com.WINSIZE, training_ratio=com.TRAINING_RATIO, loadraw=True, loadfeature=True)
    model = RepeaterTSPPR(dataset, com.WINSIZE, min_rep_gap=com.MIN_REP_GAP, negative_samples=com.NEG_SAMPLE)
    model.train(K=40, reg_lambda=0.001, reg_gamma=0.1, max_iters=int(1e8), alpha=0.02, tol=1e-4, ratio=0.1, small_batch=int(2e6))
    model.evaluate(topks)
    
    print 'Done'
    
    