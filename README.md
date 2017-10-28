# TS-PPR #
Codes for the TS-PPR algorithm: Recommendation for repeat consumption from user implicit feedback. [TKDE-2016](http://ieeexplore.ieee.org/document/7518642/?arnumber=7518642) [ICDE-2017 Extended Abstract](http://ieeexplore.ieee.org/document/7929912/)  
TS-PPR is a temporal recommendation algorithm which recommends the items that the target user has consumed before but prefers at this moment. There are wide application scenarios of this kind of algorithm, e.g. music or point-of-interest recommendation.

If you are using the codes in this repo, please cite the above two papers.  
<pre>
<code>
@article{chen2016tsppr,
  title={Recommendation for Repeat Consumption from User Implicit Feedback},
  author={Chen, Jun and Wang, Chaokun and Wang, Jianmin and Philip, S Yu},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={28},
  number={11},
  pages={3083--3097},
  year={2016},
  publisher={IEEE}
}

@inproceedings{chen2017tsppr,
  title={Recommendation for Repeat Consumption from User Implicit Feedback (Extended Abstract)},
  author={Chen, Jun and Wang, Chaokun and Wang, Jianmin and Philip, S Yu},
  booktitle={IEEE International Conference on Data Engineering},
  year={2017},
  publisher={IEEE}
}
</code>
</pre>  

# Code Usage #  
There are four source files in this repo.

### commons.py ###  
Settings of hyper-parameters.
<pre>
<code>
TRAINING_RATIO = 0.7 # training /testing splitting ratio
WINSIZE = 100        # size of sliding window (a.k.a. number of steps)
MIN_REP_GAP = 10     # the minimum length of gap between repetition (parameter \Omega in the paper)
NEG_SAMPLE = 10      # number of negative samples w.r.t. a single positive repeat 
                     # consumption (parameter S in the paper)
</code>
</pre>

### datasets.py ###
The codes to preprocess the datasets of Gowalla and Last.fm from their raw files.  
+ [Gowalla](https://snap.stanford.edu/data/loc-gowalla.html): Gowalla_totalCheckins.txt  
+ [Lastfm](http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html): userid-timestamp-artid-artname-traid-traname.tsv  

In *datasets.py*, you should preprocess the raw data like this:
<pre>
<code>
import commons as com

name = 'go' # denoted for Gowalla, or name whatever you like
preprocess('*Your path to the Gowalla file*/Gowalla_totalCheckins.txt', name)
dataset = Dataset(name, com.WINSIZE, training_ratio=com.TRAINING_RATIO)
dataset.print_dataset_statistics()
dataset.extract_features(train_only=True, winsize=com.WINSIZE)
</code>
</pre>  

This will creates files *behaviors_go.array*, *user_id_go.map* and *item_id_go.map*. 
The statistics of the created dataset will also be printed.

## base.py  ##  
The base class *BaseModel* of all (proposed or baseline) models. 
Basically, there is nothing you need to change in this file. 
This class has an interface *_recommend* which should be override in your model extending this base class.
This class also provides a *evaluate* method as the common evaluation for all models.

## tsppr.py ##
The codes for the main recommendation algorithm TS-PPR. 
The main class *RepeaterTSPPR* extends *BaseModel*. 
In *tsppr.py*, you can run the codes like:
<pre>
<code>
import commons as com
from datasets import Dataset

name = 'go'
dataset = Dataset(name, com.WINSIZE, training_ratio=com.TRAINING_RATIO, loadraw=True, loadfeature=True)
model = RepeaterTSPPR(dataset, com.WINSIZE, min_rep_gap=com.MIN_REP_GAP, negative_samples=com.NEG_SAMPLE)
model.train(K=40,                # user/item embedding length 
            reg_lambda=0.01,     # regularization for parameter lambda
            reg_gamma=0.05,      # regularization for parameter gamma
            max_iters=int(1e8),  # maximum number of iterations (each iteration only scans one sample)
            alpha=0.02,          # learning rate
            tol=1e-4,            # tolerance for early stopping
            ratio=0.1,           # fraction of all samples to compute the loss
            small_batch=int(2e5) # size of *small* batch
            )
model.evaluate(topks=[1,5,10])
</code>
</pre>
