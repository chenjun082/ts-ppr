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
  journal={IEEE International Conference on Data Engineering},
  year={2017},
  publisher={IEEE}
}
</code>
</pre>  

There are four source files in this repo.

## datasets.py ##
The codes to preprocess the datasets of Gowalla and Last.fm from their raw files.  
+ Gowalla_totalCheckins.txt  
+ userid-timestamp-artid-artname-traid-traname.tsv

## commons.py ##

Settings of hyper-parameters.

## base.py  ##

The base class of all comparative methods. 
If you are going to compare with TS-PPR, please extend the *base* class, 
rewrite the *_recommend* method, and call the *evaluate* method for evaluation.

## tsppr.py ##

The codes for the main recommendation algorithm TS-PPR.
