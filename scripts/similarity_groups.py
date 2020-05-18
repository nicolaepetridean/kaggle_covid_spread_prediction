import os
os.chdir('..')

import numpy   as np
import pandas  as pd

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import scripts.write_to_elk as elk_service
import scripts.config as config

def preprocess_data(df):
    renameState     = df['Province_State'].fillna(0).values
    renameCountries = df['Country_Region'].values
    renameState[renameState == 0] = renameCountries[renameState == 0]
    df['Province_State'] = renameState

    return df


def compare_sequence(source, candidate, errorFunc):
    minError = np.inf
    minIdx = -1

    # only check the countries that can influence
    if len(candidate) > len(source):
        noWindows = len(candidate) - len(source)
        windowSize = len(source)

        # sliding window over candidate country
        for i in range(0, noWindows):

            # compute loss
            error = errorFunc(source, candidate[i:i + windowSize])

            # save the min error and its location
            if error <= minError:
                minError = error
                minIdx = i

        return minError, minIdx

    # return none if invalid
    return None, None


def get_nearest_sequence(df, state, alignThreshConf=50, alignThreshDead=10, errorFunc=mean_absolute_error):
    resDf = pd.DataFrame(columns=['Province_State', 'deathError', 'confirmedError', 'deathIdx', 'confirmedIdx'])
    confDf = df[df['ConfirmedCases'] > alignThreshConf]
    deadDf = df[df['Fatalities'] > alignThreshDead]

    # get source region data
    regionDfConf = confDf[confDf['Province_State'] == state].sort_values(by='Date', ascending=True)
    regionDfDead = deadDf[deadDf['Province_State'] == state].sort_values(by='Date', ascending=True)

    regionConf = regionDfConf['ConfirmedCases'].values
    regionDead = regionDfDead['Fatalities'].values

    # check all possible candidates
    for neighbour in df['Province_State'].unique():

        # skip comparing with the same country
        if neighbour == state:
            continue

        # get country candidate
        confNeighDf = confDf[confDf['Province_State'] == neighbour].sort_values(by='Date', ascending=True)
        deadNeighDf = deadDf[deadDf['Province_State'] == neighbour].sort_values(by='Date', ascending=True)

        neighConf = confNeighDf['ConfirmedCases'].values
        neighDead = deadNeighDf['Fatalities'].values

        # get error for confirmed and neighbour
        confErr, confIdx = compare_sequence(regionConf, neighConf, errorFunc)
        deadErr, deadIdx = compare_sequence(regionDead, neighDead, errorFunc)

        # the candidate will be ignored if it does not have enough data
        if confErr is None or deadErr is None:
            continue

        # append result
        res = {'Province_State': neighbour, 'deathError': deadErr, 'confirmedError': confErr,
               'deathIdx': deadIdx, 'confirmedIdx': confIdx}

        resDf = resDf.append(res, ignore_index=True)

    return resDf


def l1_norm_error(source, candidate):
    error = (abs(source - candidate))
    source[source == 0] = 1e-30  # add for numerical stability
    error = error / source  # normalize the error
    error = error.mean()

    return error


def rmsle_error(source, candidate):
    candidate += 1e-30

    error = np.log10((source + 1) / (candidate + 1))  # 1 is added for numerical stability
    error = error * error
    error = error.mean()
    error = np.sqrt(error)

    return error


def show_country_nn(data, sourceState, alignThreshConf, alignThreshDead, listErrorDf, errorNames):
    SHOW_FIRST = 3  # only show the first top neighbours

    # setup plot figures
    fig, axes = plt.subplots(len(listErrorDf), 2,
                             figsize=(15, len(listErrorDf) * 3),
                             gridspec_kw={'hspace': 0.3})
    # get rid of the annoying
    axes = axes.flatten()

    fig.suptitle(sourceState.title(), fontsize=20)
    colors = sns.color_palette()[:SHOW_FIRST + 1]

    # only keep aligned data
    showDataConf = data[data['ConfirmedCases'] > alignThreshConf].copy()
    showDataDead = data[data['Fatalities'] > alignThreshDead].copy()
    showData = [showDataConf, showDataDead]

    for i, (attr, err) in enumerate(zip(['ConfirmedCases', 'Fatalities'],
                                        ['confirmedError', 'deathError'])):

        for j, (error, name) in enumerate(zip(listErrorDf, errorNames)):
            legend = []
            axIdx = j * 2 + i
            tempError = error.sort_values(by=err, ascending=True)

            # only show available neighbours (if they are less than SHOW_FIRST)
            show = min(SHOW_FIRST, tempError.shape[0])

            for k in range(1, show + 1):

                # plot neighbours
                neighbour = tempError['Province_State'].iloc[k - 1]
                tempShow = showData[i][showData[i]['Province_State'] == neighbour][[attr, 'Date']]
                xAxisValues = [z for z in range(tempShow.shape[0])]

                if len(xAxisValues) > 0:
                    legend.append(neighbour)

                sns.lineplot(x=xAxisValues, y=tempShow, color=colors[k],
                             ax=axes[axIdx], linewidth=4.5)

            # plot source country
            tempShow = showData[i][showData[i]['Province_State'] == sourceState][attr]
            xAxisValues = [z for z in range(tempShow.shape[0])]
            sns.lineplot(x=xAxisValues, y=tempShow, color=colors[0],
                         ax=axes[axIdx], linewidth=4.5)

            # final touches to figure
            axes[axIdx].legend(legend + [sourceState])
            axes[axIdx].set_title(name.title() + ' error')
            axes[axIdx].grid(True)
            axes[axIdx].box = True
    return axes


def generate_country_data(data, sourceState, alignThreshConf, alignThreshDead, listErrorDf, errorNames):
    # only keep aligned data
    showDataConf = data[data['ConfirmedCases'] > alignThreshConf].copy()
    showDataDead = data[data['Fatalities'] > alignThreshDead].copy()
    showData = [showDataConf, showDataDead]
    day_of_year = datetime.now().timetuple().tm_yday

    for i, (attr, err) in enumerate(zip(['ConfirmedCases', 'Fatalities'],
                                        ['confirmedError', 'deathError'])):
        for j, (error, name) in enumerate(zip(listErrorDf, errorNames)):
            tempError = error.sort_values(by=err, ascending=True)

            for k in range(1, len(tempError) + 1):
                # plot neighbours
                neighbour = tempError['Province_State'].iloc[k - 1]
                tempShow = showData[i][showData[i]['Province_State'] == neighbour][[attr, 'Date']]
                tempShow['SourceState'] = sourceState
                tempShow['Neighbour'] = neighbour
                tempShow['NeighbourRank'] = k
                tempShow['NeighbourOfType'] = attr
                tempShow['CovidDay2020'] = day_of_year

                # write data to ELK
                if attr == 'ConfirmedCases':
                    tempShow = tempShow.rename(columns={"ConfirmedCases": "cases"})
                    tempShow['confirmedError'] = tempError['confirmedError'].iloc[k - 1]
                    tempShow['confirmedIdx'] = tempError['confirmedIdx'].iloc[k - 1]
                else:
                    tempShow = tempShow.rename(columns={"Fatalities": "cases"})
                    tempShow['deathError'] = tempError['deathError'].iloc[k - 1]
                    tempShow['deathIdx'] = tempError['deathIdx'].iloc[k - 1]

                tempShow['daysFromAlignment'] = np.arange(len(tempShow))

                write_data_to_df(tempShow)


def write_data_to_df(data_to_write):
    config_set = config.read_yml_config_file('scripts\\run_local.yaml')
    elk_client = elk_service.elk_connect(config_set)
    elk_service.check_mapping_exists(config_set['country_grouping_es_index'], elk_client)

    elk_service.write_data_from_dataframe(dataframe=data_to_write, es_index=config_set['country_grouping_es_index'],
                                          es_client=elk_client)


def test_metrics(trainData, sourceCountry, alignThreshConf, alignThreshDead):
    results = []
    errorNames = ['MAPE', 'MSE', 'RMSLE']
    errors = [l1_norm_error, mean_absolute_error, rmsle_error]

    # compute error df for each metric
    for error in errors:
        r = get_nearest_sequence(trainData, sourceCountry, alignThreshConf, alignThreshDead, error)
        results.append(r)

    # diplay for both fatalities and confirmed cases
    show_country_nn(trainData, sourceCountry, alignThreshConf, alignThreshDead, results, errorNames)


if __name__ == '__main__':
    allData = pd.read_csv('assets/covid_spread.csv', parse_dates=['Date'])
    allData = preprocess_data(allData)
    r = get_nearest_sequence(allData, 'Germany', 40, 10)
    for country in ['Romania', 'Germany', 'Czechia']:
        r = get_nearest_sequence(allData, country, 500, 40, l1_norm_error)
        ##show_country_nn(allData, country, 500, 40, [r], ['MAPE'])
        generate_country_data(allData, country, 500, 40, [r], ['MAPE'])
    print('a')