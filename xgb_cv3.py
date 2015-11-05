# WITH CATEGORICAL VARIABLES CONVERTED TO CON_PROB VARIABLES
# import libraries
import pandas as pd
import xgboost as xgb
import re
# %matplotlib inline

# import logger.py
from logger import logger


logger.info('Start modelling.py')

# load xgb matrix from binary
train = xgb.DMatrix('train_proc_full3.buffer')
logger.info('xgb_matrix loaded')

# Run xgb
# initialize xgb params
param = {'eta': 0.0375,
         'gamma': 0.75,
         'max_depth': 14,
         'min_child_weight': 15,
         'sub_sample': 0.85,
         'colsample_bytree': 0.75,
         'alpha': 3,
         'objective': 'binary:logistic',
         'eval_metric': 'auc',
         'seed': 0}
num_round = 2000

auc_hist = xgb.cv(params = param, dtrain = train, num_boost_round = num_round, 
                  nfold = 5, seed = 668, show_stdv = False)

### get auc from train and test
# extract auc from string values
auc_test = {}
auc_train = {}
for row, auc in enumerate(auc_hist):
    auc_test[row] = re.search(r'cv-test-auc:(.*)\s', auc).group(1)
    auc_train[row] = re.search(r'cv-train-auc:(.*)', auc).group(1)

# create auc dataframe
auc_test_df = pd.DataFrame(auc_test.items(), columns = ['rounds', 'auc_test'])
auc_train_df = pd.DataFrame(auc_train.items(), columns = ['rounds', 'auc_train'])
auc_df = auc_train_df.merge(auc_test_df, on = 'rounds')
auc_df = auc_df.astype('float')

auc_df.to_csv('auc_cv3.csv', index = False)
