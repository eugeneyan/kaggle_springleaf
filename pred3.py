# import libraries
import xgboost as xgb
import pandas as pd
import gc

# import logger.py
from logger import logger


# set iteration
iteration = '3'


logger.info('Start pred' + iteration + '.py')

# load xgb matrix from binary
train = xgb.DMatrix('train_proc_full' + iteration + '.buffer')
logger.info('train xgb_matrix loaded')


# RUN XGB
# initialize xgb params
# param = {'eta': 0.0375,
#          'gamma': 0.75,
#          'max_depth': 14,
#          'min_child_weight': 15,
#          'sub_sample': 0.85,
#          'colsample_bytree': 0.75,
#          'alpha': 3,
#          'objective': 'binary:logistic',
#          'eval_metric': 'auc',
#          'seed': 0}
# num_round = 3
# num_round = 1688

# set number of xgb models to bag
bag_size = 30
pred_bag = pd.read_csv('sample_submission.csv')['target']
# pred_bag = pd.read_csv('sample_sub.csv')['target']

test = xgb.DMatrix('test_proc_full' + iteration + '.buffer')
logger.info('test xgb_matrix loaded')


for bag in range(bag_size):
    logger.info('Starting on xgb_model: %d' % bag)

    param = {'eta': 0.0375,
             'gamma': 0.75,
             'max_depth': 14,
             'min_child_weight': 15,
             'sub_sample': 0.85,
             'colsample_bytree': 0.75,
             'alpha': 3,
             'objective': 'binary:logistic',
             'eval_metric': 'auc',
             'seed': 100+bag}
    # num_round = 10
    num_round = 1688

    xgb_mod = xgb.train(params=param, dtrain=train, num_boost_round=num_round)
    logger.info('xgb_model %d created' % bag)

    gc.collect()

    # CREATE PREDICTIONS
    test_idx = pd.read_csv('sample_submission.csv')['ID']
    # test_idx = pd.read_csv('sample_sub.csv')['ID']

    pred = xgb_mod.predict(test)
    pred_bag += pred
    pred_df = pd.DataFrame(data=pred, index=test_idx, columns=['target'], dtype='Float64')
    logger.info('pred_df created')

    # WRITE TO CSV
    pred_df.to_csv('pred' + iteration + '_' + str(bag) + '.csv')
    logger.info('pred' + iteration + '_' + str(bag) + '.csv created')


# CREATE PREDICTIONS
test_idx = pd.read_csv('sample_submission.csv')['ID']
# test_idx = pd.read_csv('sample_sub.csv')['ID']
pred_bag = pred_bag / bag_size
pred_bag_df = pd.DataFrame(data=pred_bag, index=test_idx, columns=['target'], dtype='Float64')
pred_bag_df.to_csv('pred' + iteration + 'bag' + '.csv')
logger.info('pred' + iteration + '_' + 'bag' + '.csv created')
