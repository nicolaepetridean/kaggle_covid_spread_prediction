import os

import torch
import numpy  as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data      import Dataset


# =============================================== DATASET ==============================================================
class CovidDataset(Dataset):
    colMapping = {'Date'            : 'date',    'Province_State': 'state',
                    'Country_Region': 'country', 'ConfirmedCases': 'confirmed',
                    'Fatalities'    : 'deaths',  'Id'            : 'id',}

    def __init__(self, root, windowSize = 10, predictSize = 4, batchSize = 64, device = 'cpu', trainProc = 0.9):
        '''
        Utility class for Kaggle Covid 19 dataset. The output data will be sorted descending.
        :param root: folder root where train.csv is stored
        :param windowSize: window size (total time of prediction)
        :param predictSize: how much of window size will have to be predicted
        :param batchSize: batch size
        :param device: string of where the data will be loaded
        :param trainProc: the percentage of data used for trainin (the rest is for test)
        '''
        self.root        = root
        self.batchSize   = batchSize
        self.predictSize = predictSize
        self.windowSize  = windowSize
        self.trainProc   = trainProc

        self.scaler   = MinMaxScaler([-1, 1])
        self.trainDf  = pd.read_csv(os.path.join(root, 'train.csv'), parse_dates=['Date'])
        self.testDf   = pd.read_csv(os.path.join(root, 'test.csv'))
        self.trainDf  = self.preprocess_train(self.trainDf)

        # self.casualties = trainDf
        self.parsedData = self._parse_countries(self.trainDf)

        # augment data
        self.data = self.parsedData
        self.data[:,:,0] = np.log10(self.data[:,:,0])

        # scale data [-1, 1]
        batch, time = self.data.shape[:2]
        self.scaler.fit(self.data.reshape(time*batch,-1))
        self.data = self.scaler.transform(self.data.reshape(batch*time, -1)).reshape(batch, time, -1)

        # convert to tensor
        self.data = torch.from_numpy(self.data).float()
        self.data.to(device)

        #shuffle
        order = np.arange(self.data.shape[0])
        np.random.shuffle(order)
        self.data[np.arange(self.data.shape[0]), :, :] = self.data[order, :, :]

        # split
        trainStep     = int(trainProc * self.data.shape[0])
        self.testData = self.data[trainStep:, :, :]
        self.data     = self.data[:trainStep, :, :]

    # =============================================== GETITEM =====================================================
    def __getitem__(self, idx):
        '''
        Return batch idx
        :param idx:
        :return:
        '''
        idx   = idx % len(self)
        batch = slice(idx * self.batchSize, (idx +1) * self.batchSize, 1)

        return self.data[batch, :self.windowSize-self.predictSize,:], \
               self.data[batch, self.windowSize-self.predictSize:,:]

    # =============================================== LEN =====================================================
    def __len__(self):
        return self.data.shape[0]//self.batchSize

    # =============================================== PREPROCESS =====================================================
    def preprocess_train(self, df):
        # fill the state field with name of the country (if it is null)
        df = df.rename(columns=CovidDataset.colMapping)
        renameState     = df['state'].fillna(0).values
        renameCountries = df['country'].values
        renameState[renameState == 0] = renameCountries[renameState == 0]
        df['state'] = renameState

        return df

    # =============================================== PARSE BATCHES =====================================================
    def _parse_countries(self, df):
        trainData = []
        for region in df['state'].unique():
            # get country data
            d = df[df['state'] == region].sort_values(by='date', ascending=False)

            # get only values with a value larger than 1
            d = d[d['confirmed'] > 0]

            # make each batch (sliding window)
            if d.shape[0] >= self.windowSize:
                for i in range(d.shape[0] - self.windowSize):
                    dat = d[['confirmed', 'deaths']].values[i:i+self.windowSize]
                    trainData.append(dat)

        # batch, time, size
        data   = np.array(trainData).astype(np.float64)
        return data

    # =============================================== GET EVAL BATCHES =================================================
    def get_test_data(self):
        '''
        Get a list of batches test data
        :return:
        '''
        noBatches = self.testData.shape[0] // self.batchSize
        sliceData = lambda x : slice(x*self.batchSize, (x+1)*self.batchSize, None)

        data   = list()
        labels = list()
        for i in range(noBatches):
            data.append(self.testData[sliceData(i),:self.windowSize - self.predictSize,:])
            labels.append(self.testData[sliceData(i),self.windowSize - self.predictSize:,:])

        return data, labels

    # =============================================== AUGMENT ==========================================================
    def _random_add(self, data, maxRange):
        data = data.copy()
        for i in range(data.shape[0]):
            data[i,0,:] = data[i, :, :] + np.random.randint(0, maxRange)
        return data

    def _random_mul(self, data, maxRange):
        for i in range(data.shape[0]):
            data[i,:,:] = data[i, :, :] * np.random.randint(1, maxRange)

        return data
    # =============================================== TRANSFORM INVERSE ================================================
    def transform_inverse(self, data):
        # inverse transform the normalized data
        batch, time = data.shape[:2]
        data = self.scaler.transform(data.reshape(batch * time, -1)).reshape(batch, time, -1)
        data[:, :, 0] = np.pow(10, data[:, :, 0])
        return  data


if __name__ == '__main__':
    d = CovidDataset('C:\\Users\\beche\\Documents\\GitHub\\kaggle_covid_spread_prediction\\src\\assets')
    t = d[10]
    print(t[0].shape)
    # print(t[0])
