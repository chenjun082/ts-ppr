# TS-PPR
Codes for the TS-PPR algorithm: Recommendation for repeat consumption from user implicit feedback.

TKDE-2016: http://ieeexplore.ieee.org/document/7518642/?arnumber=7518642

ICDE-2017 Extended Abstract: http://ieeexplore.ieee.org/document/7929912/

If you are using the codes in this repo, please cite the above two papers.

This repo contains the main codes for the TS-PPR algorithm in the above research paper.
There are four source files in this repo.

# 1. datasets.py
The codes to preprocess the datasets of Gowalla and Last.fm from their raw files.

Gowalla_totalCheckins.txt

userid-timestamp-artid-artname-traid-traname.tsv

# 2. commons.py

Settings of hyper-parameters.

# 3. base.py

The base class of all comparative methods. 
If you are going to compare with TS-PPR, please extend the base class, 
rewrite the '_recommend' method, and call the 'evaluate' method for evaluation.

# 4. tsppr.py

The codes for the main recommendation algorithm TS-PPR.
