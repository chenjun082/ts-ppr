'''

@author: Chen Jun

'''

from datetime import timedelta

SIX_HOUR_DUR = timedelta(hours=6).total_seconds()
HALF_DAY_DUR = timedelta(hours=12).total_seconds()
ONE_DAY_DUR  = timedelta(days=1).total_seconds()
THIRTY_SECS_DUR = timedelta(seconds=30).total_seconds()

TRAINING_RATIO = 0.7

WINSIZE = 100

MIN_REP_GAP = 10

NEG_SAMPLE = 10