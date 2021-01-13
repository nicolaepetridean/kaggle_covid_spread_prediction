import os
import warnings
warnings.filterwarnings('ignore')

import numpy   as np
import pandas  as pd

import seaborn as sns
from matplotlib import pyplot as plt

from tqdm             import tqdm
from IPython.display  import display

from core.data      import compare_countries as cc
from core.data      import utils             as dataUtils
from datetime import datetime, timedelta

from core.nn.loss   import l1_norm_error

import scripts.write_to_elk as elk_service
import scripts.config as config




COUNTRY      = 'Romania'
DEVICE       = 'cpu'
TRAIN_UP_TO  = pd.to_datetime('2020-10-10')

allData = pd.read_csv('../assets/covid_spread.csv', parse_dates=['Date'])
#allData.head()


allData = dataUtils.preprocess_data(allData)
#allData.head()


allData['Org_ConfirmedCases'] = allData['ConfirmedCases']
for country in set(allData.Province_State):
    idx_loc = allData.Province_State == country
    allData.loc[idx_loc, 'ConfirmedCases']=allData.loc[idx_loc, 'ConfirmedCases'].rolling(window=8).mean()
allData = allData.fillna(method='backfill')

allData['ConfirmedCases'] = allData.groupby(['Province_State'])['ConfirmedCases'].diff(periods=1)
allData['ConfirmedCases'] = allData['ConfirmedCases'].fillna(0)
allData['ConfirmedCases'] = np.where(allData['ConfirmedCases'] < 0, 0, allData['ConfirmedCases'])

allData['ConfirmedCases'] = np.log(allData["ConfirmedCases"] + 1)

errorData  = cc.get_nearest_sequence(allData, COUNTRY,
                                     alignThreshConf = 0.5,
                                     alignThreshDead = 0.5,
                                     errorFunc       = l1_norm_error
                                    )

display(errorData.sort_values(by='confirmedError').head())
display(errorData.sort_values(by='deathError').head())

confData = dataUtils.get_target_data(allData, errorData,
                                     errorThresh = 0.45,
                                     country     = COUNTRY,
                                     target      = 'confirmed')
deadData = dataUtils.get_target_data(allData, errorData,
                                     errorThresh = 0.45,
                                     country     = COUNTRY,
                                     target      = 'fatalities')

idx = confData.Province_State == COUNTRY
vConfData = confData[idx]

confScaler = dataUtils.get_scaler(confData, 'confirmed')
deadScaler = dataUtils.get_scaler(deadData, 'fatalities')

winSize       = 15
obsSize       = 14
futureSteps   = 1
supPredSteps  = winSize - obsSize
uPredSteps    = futureSteps - supPredSteps
allPredSteps  = futureSteps + obsSize

confTrainData = dataUtils.get_train_data(confData, 'confirmed',
                                  step       = 1,
                                  winSize    = winSize,
                                  trainLimit = TRAIN_UP_TO,
                                  scaler     = None,
#                                   shuffle    = True
                                        )
confTrainData.shape

confValidData = dataUtils.get_train_data(vConfData, 'confirmed',
                                  step       = 1,
                                  winSize    = winSize,
                                  trainLimit = TRAIN_UP_TO,
                                  scaler     = None,
#                                   shuffle    = True
                                        )
confValidData.shape

X_train = confTrainData[:, :obsSize]
y_train = confTrainData[:, -1]

X_valid = confValidData[:, :obsSize]
y_valid = confValidData[:, -1]


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from tensorflow.compat.v1 import set_random_seed
np.random.seed(1)
set_random_seed(2)


def lstm_model_builder(n_features, n_targets, window_size):
    n_lstm_units = 10 * n_features

    seq_inputs = Input(shape=(window_size, n_features))
    seq_branch = LSTM(5 * n_lstm_units, return_sequences=False)(seq_inputs)
    seq_branch = Dense(2 * n_lstm_units, activation='relu')(seq_branch)
    out_branch = Dense(2 * n_features, activation='relu')(seq_branch)
    output = Dense(n_targets)(out_branch)

    model = Model([seq_inputs], output)

    return model

def rmsle(y, y0):
    return K.sqrt(K.mean(K.pow(K.log(y + 1) - K.log(y0 + 1), 2)))

adam = Adam(learning_rate=3e-3)
reduce_lr = ReduceLROnPlateau(factor=0.8, patience=5, min_lr=3e-4, verbose=1, monitor='val_loss')
ckpt = ModelCheckpoint('rmsle_model1.h5', save_best_only=True, monitor='val_loss')
model = lstm_model_builder(n_features=1, n_targets=1, window_size=obsSize)
model.compile(adam, loss=rmsle, metrics=['mse'])

model.fit(X_train, y_train, validation_data=([X_valid], y_valid),
          batch_size=16, epochs=25, callbacks=[reduce_lr, ckpt])

model = load_model('rmsle_model1.h5', custom_objects={'rmsle': rmsle})

adam = Adam(learning_rate=3e-5)
model.compile(adam, loss='mae', metrics=[rmsle])
reduce_lr = ReduceLROnPlateau(factor=0.8, patience=5, min_lr=3e-5, verbose=1, monitor='rmsle')
ckpt = ModelCheckpoint('mae_model1.h5', save_best_only=True, monitor='rmsle')


model.fit(X_train, y_train, validation_data=([X_valid], y_valid),
          batch_size=32, epochs=5, callbacks=[reduce_lr, ckpt])

model = load_model('mae_model1.h5', custom_objects={'rmsle': rmsle})

confValData, confValLabel = dataUtils.get_val_data(allData, 'confirmed',
                                                   COUNTRY,
                                                   TRAIN_UP_TO,
                                                   obsSize)

def predict_future(n_days, arr):
    pred_death_cases = []

    for _ in range(n_days):
        pred = model.predict([arr])
        pred_death_cases.append(pred)
        new_element = np.expand_dims(pred, axis=0)
        arr = np.concatenate([arr[:,1:,:], new_element], axis=1)
    pred_death_cases = np.concatenate(pred_death_cases)

    return np.squeeze(np.expm1(pred_death_cases))


# from sklearn.utils import check_arrays
def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = check_arrays(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def write_data_to_df(data_to_write):
    config_set = config.read_yml_config_file('run_local.yaml')
    elk_client = elk_service.elk_connect(config_set)
    elk_service.check_mapping_exists(config_set['confirmed_prediction_index'], elk_client)

    elk_service.write_data_from_dataframe(dataframe=data_to_write, es_index=config_set['confirmed_prediction_index'],
                                          es_client=elk_client)

# get figure
fig, ax = plt.subplots(1, 1, figsize = (9, 4))
ax.tick_params(axis='x', rotation=45)

start_cases = allData.loc[(allData.Date == TRAIN_UP_TO) & (allData.Province_State == COUNTRY), 'Org_ConfirmedCases'].values

# make prediction
# n_days = confValLabel.shape[0]
n_days = 13
pred   = predict_future(n_days, arr=confValData)
pred   = start_cases + np.cumsum(pred)

prediction_date = datetime.now()
prediction_start_date = TRAIN_UP_TO
prediction_start_date = prediction_start_date + timedelta(1)
# df = pd.DataFrame({'ConfirmedPrediction': pred})
df = pd.DataFrame()
df['Date'] = pd.date_range(start=prediction_start_date, periods=len(df), freq='D')


# prediction
predDate = pd.date_range(start = TRAIN_UP_TO, periods=pred.shape[0])

# plot train data
showTrainData = allData[allData['Province_State'] == COUNTRY]
showTrainData = showTrainData[(showTrainData['Date'] <= prediction_start_date)]

merged = showTrainData.merge(df, how='outer', on="Date")
merged = merged.drop('Country_Region', axis=1)
merged = merged.drop('ConfirmedCases', axis=1)
merged = merged.drop('Fatalities', axis=1)
#merged = merged.drop('ConfirmedPrediction', axis=1)
merged = merged.rename(columns={"Org_ConfirmedCases": "TrainConfirmations"})
merged['DaysFromZero'] = np.arange(len(merged))
merged['PredictionDate'] = prediction_date
merged = merged.rename(columns={"Province_State": "ProvinceState"})
write_data_to_df(merged)


df = pd.DataFrame()
df['Date'] = pd.date_range(start=prediction_start_date, periods=len(df), freq='D')

# plot val data
showValData = allData[allData['Province_State'] == COUNTRY]
showValData = showValData[showValData['Date'] > TRAIN_UP_TO]
sns.lineplot(y = 'Org_ConfirmedCases', x ='Date', data = showValData, ax = ax, linewidth=4.5, alpha=0.7);

showValData = showValData.drop('Country_Region', axis=1)
showValData = showValData.drop('ConfirmedCases', axis=1)
showValData = showValData.drop('Fatalities', axis=1)
merged = showValData.merge(df, how='outer', on="Date")
merged = merged.rename(columns={"Org_ConfirmedCases": "ValidationConfirmations"})

# merged = merged.drop('Province_State_y', axis=1)
# merged = merged.drop('Province_State_x', axis=1)
# merged = merged.fillna(0)
merged['DaysFromZero'] = np.arange(len(showTrainData)-1, len(showTrainData) + len(showValData)-1)
merged['PredictionDate'] = prediction_date
merged['ProvinceState'] = COUNTRY
merged = merged.drop('Province_State', axis=1)
write_data_to_df(merged)

df = pd.DataFrame({'ConfirmedPrediction': pred})
df['Date'] = pd.date_range(start=prediction_start_date, periods=len(df), freq='D')
df['DaysFromZero'] = np.arange(len(showTrainData)-1, len(showTrainData) + len(pred)-1)
df['PredictionDate'] = prediction_date
df['ProvinceState'] = COUNTRY

write_data_to_df(df)
