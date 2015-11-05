# Create dummy variables for categorical features with less than 5 unique values
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import gc
import datetime
import calendar
import xgboost as xgb

# import logger.py
from logger import logger

# set iteration
iteration = '3'


logger.info('Start data_prep_full' + iteration + '.py')

# read in data
train = pd.read_table('train.csv', sep=',')
test = pd.read_table('test.csv', sep=',')

# train = pd.read_table('train_sample.csv', sep=',')
train_idx = range(0, train.shape[0])
logger.info('Train data read, dimensions: %d x %d' % (train.shape[0], train.shape[1]))
# test = pd.read_table('test_sample.csv', sep=',')
test_idx = range(train.shape[0], train.shape[0] + test.shape[0])
logger.info('Test data read, dimensions: %d x %d' % (test.shape[0], test.shape[1]))

# create target series
target = train['target']

# append train and test for data prep
df = train.append(test)
logger.info('Train and Test appended, dimensions: %d x %d' % (df.shape[0], df.shape[1]))

# combine train and test for label encoding and preprocessing
gc.collect()

# remove duplicates
# load colummns to remove from csv file
cols_to_rm = list(pd.read_csv('cols_to_rm.csv'))

# add ID col to columns to remove
cols_to_rm += ['ID']

# add the following cols that are duplicative
cols_to_rm += ['VAR_0044']
cols_to_rm += ['target']

# remove duplicate columns and ID
df.drop(labels=cols_to_rm, axis=1, inplace=True)  # disable for now
logger.info('Redundant columns removed, dimensions: %d x %d' % (df.shape[0], df.shape[1]))

# remove garbage
gc.collect()

# Clean up dates
# Initialize list of datetime columns
date_cols = ['VAR_0073', 'VAR_0075',
             'VAR_0156', 'VAR_0157', 'VAR_0158', 'VAR_0159',
             'VAR_0166', 'VAR_0167', 'VAR_0168', 'VAR_0169',
             'VAR_0176', 'VAR_0177', 'VAR_0178', 'VAR_0179', 'VAR_0204']


# function to get week of month
def week_of_month(date):
    if not pd.isnull(date):
        days_this_month = calendar.mdays[int(date.month)]
        for i in range(1, days_this_month):
            d = datetime.datetime(date.year, date.month, i)
            if d.day - d.weekday() > 0:
                startdate = d
                break

        # now we can use the modulo 7 appraoch
        return (date - startdate).days // 7 + 1


def format_dates(df):
    for col in date_cols:
        year_col = col + '_yr'
        month_col = col + '_mth'
        quart_col = col + '_q'
        day_col = col + '_day'
        doy_col = col + '_doy'
        woy_col = col + '_woy'
        dow_col = col + '_dow'
        wom_col = col + '_wom'
        df[col] = pd.to_datetime(df[col], format='%d%b%y:%H:%M:%S')
        df[year_col] = df[col].dt.year
        df[month_col] = df[col].dt.month
        df[day_col] = df[col].dt.day
        df[quart_col] = df[col].dt.quarter
        df[doy_col] = df[col].dt.dayofyear
        df[woy_col] = df[col].dt.weekofyear
        df[dow_col] = df[col].dt.dayofweek+1  # +1 so monday = 1
        df[wom_col] = df[col].apply(week_of_month)+1  # +1 so wom starts from 1


# run format dates
format_dates(df)
    
# get hour of day for VAR_0204
df['VAR_0204_hr'] = df['VAR_0204'].dt.hour+1  # +1 first hour = 1
logger.info('Dates formatted')

# save date_cols created for future identification
date_cols_full = []

for date in date_cols:
    date_cols_full += [date+'_yr'] + [date+'_mth'] + [date+'_q'] + [date+'_day'] + [date+'_doy'] \
                      + [date+'_woy'] + [date+'_dow'] + [date+'_wom']

date_cols_full += ['VAR_0204_hr']
date_cols_full = pd.DataFrame(date_cols_full, columns=['date_cols'])
date_cols_full.to_csv('date_cols_full.csv', index=False)
logger.info('date_cols_full.csv saved')


# initialize label encoder
le = LabelEncoder()


# define function to encode categorical columns via apply
def col_cleanup(col):
    if col.dtype.name == 'object':
        le.fit(col)
        return le.transform(col).astype(int)
    else:
        return col.fillna(-1) 
    
# apply col_cleanup to all columns
df = df.apply(col_cleanup)
logger.info('Columns encoded')

gc.collect()


# convert datetime cols to integer difference from first date in that col
# this comes after col_cleanup as we need to convert NaT date values to -1 first
def date_to_int(df):
    for col in date_cols:
        df[col] = (df[col] - df[col].min()).astype('timedelta64[D]').astype(int)
        
# convert dates to integers
date_to_int(df)
gc.collect()

#  CREATE DUMMY VARIABLES FOR CATEGORICAL COLUMNS
cat_columns = []
for col, values in df.iteritems():
    if len(values.unique()) <= 5:
        cat_columns.append(col)

cat_columns_11 = ['VAR_0283', 'VAR_0305', 'VAR_0325', 'VAR_0342']
# 404 and 493 contain occupations, don't use for now as too many unique values
# 200, 237, and 274 contain location variables, don't use for now as too many unique values

# cat_columns_11 = ['VAR_0200', 'VAR_0237', 'VAR_0274', 'VAR_0283', 'VAR_0305', 'VAR_0325',
#                   'VAR_0342', 'VAR_0404', 'VAR_0493']

cat_columns += cat_columns_11
cat_columns.remove('VAR_0204')  # remove this date column

# create df of cat_columns
df_cat = df[cat_columns]
logger.info('Dataframe of categorical features created, dimensions: %d x %d' % (df_cat.shape[0], df_cat.shape[1]))

# convert column types to object
df_cat = df_cat.astype('object')

# create dummy columns
df_cat = pd.get_dummies(df_cat)
logger.info('Dataframe of dummy variables created, dimensions: %d x %d' % (df_cat.shape[0], df_cat.shape[1]))

# get list df of numeric columns only
df = df.drop(cat_columns, axis=1)
logger.info('Dataframe of categorical features created, dimensions: %d x %d' % (df.shape[0], df.shape[1]))

# concatenate df of numeric features and df of dummy features
df = pd.concat([df, df_cat], axis=1)
logger.info('Concatenated df of numerics and df of dummy variables; dimensions: %d x %d' % (df.shape[0], df.shape[1]))


# split back into train and test set
train = df.iloc[train_idx, :]
train = pd.concat([train, target], axis=1)
test = df.iloc[test_idx, :]


# SAVE PROCESSED CSV
# save train_proc_full
logger.info('Saving train_proc_full%s.csv: %d rows x %d col' % (iteration, train.shape[0], train.shape[1]))
train.to_csv('train_proc_full' + iteration + '.csv', index=False)  # to save full vars
logger.info('train_proc_full' + iteration + '.csv saved')  # to save full vars

# remove target from train
train.drop(['target'], axis=1, inplace=True)
logger.info('target dropped from train: %d rows x %d col' % (train.shape[0], train.shape[1]))

# save test_proc_full
logger.info('Saving test_proc_full%s.csv: %d rows x %d col' % (iteration, test.shape[0], test.shape[1]))
test.to_csv('test_proc_full' + iteration + '.csv', index=False)  # to save full vars
logger.info('test_proc_full' + iteration + '.csv saved')  # to save full vars



# CREATE XGB MATRIX
logger.info('Start create_xgb_matrix.py')

# train = pd.read_csv('train_proc_full' + iteration + '.csv')
# logger.info('train_proc_full' + iteration + '.csv read')
# logger.info('Dimensions of train with target: %d x %d' % (train.shape[0], train.shape[1]))
#
# # create target series
# target = train['target']
#
# # drop target col from train
# train.drop('target', axis=1, inplace=True)
# logger.info('Dimensions of train: %d x %d' % (train.shape[0], train.shape[1]))

gc.collect()

# create xgb matrix
train_xgb = xgb.DMatrix(data=train, label=target)
logger.info('train xgbDMatrix created')

train_xgb.save_binary('train_proc_full' + iteration + '.buffer')
logger.info('train_proc_full' + iteration + '.buffer saved')

# create xgb matrix
test_xgb = xgb.DMatrix(data=test)
logger.info('train xgbDMatrix created')

test_xgb.save_binary('test_proc_full' + iteration + '.buffer')
logger.info('test_proc_full' + iteration + '.buffer saved')
