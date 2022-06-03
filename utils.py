# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import ARIMA, AutoARIMA
from orbit.models.lgt import LGT
from orbit.models.dlt import DLT
from prophet import Prophet
import statsmodels.formula.api as smf
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sktime.forecasting.ets import AutoETS

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import random

# Data Manipulation

# метод для агрегирования данных по неделям/месяцам и т.д.
def group_data(data: pd.DataFrame, date_interval: str):
    return data.groupby(
        ['DFU', 'Customer', data['Period'].dt.to_period(date_interval)]
    ).agg(
        {'BPV': sum, 'Total Sell-in': sum}
    ).reset_index()


# метод для удаления выбросов
def clean_data(data: pd.DataFrame, replace_with: str = '3sigma', comments: bool = False):
    '''
    replace_with: 'mean' / '3sigma'
    '''
    sigma = data['BPV'].std()
    mean = data['BPV'].mean()

    if comments:
        print(f"STD: {sigma}; 3 * std: {3 * sigma}")
        print(f'Mean: {mean}')

    data_cleaned = data.copy()
    replacement = mean if replace_with == 'mean' else 3 * sigma
    data_cleaned.loc[data_cleaned['BPV'] > 3 * sigma, 'BPV'] = replacement
    return data_cleaned


# метод для создания новых признаков (день, месяц, год, квартал)
def make_date_features(X: pd.DataFrame, date_col_name='Period') -> pd.DataFrame:
    X_new = X.copy()
    date_info = X[date_col_name].dt
    X_new['year'] = date_info.year
    X_new['month'] = date_info.month
    X_new['day'] = date_info.day
    X_new['week'] = date_info.week
    X_new['Q'] = date_info.quarter

    return X_new


# Metrics

# подсчет метрики wape
def wape_metric(actual: np.array, predicted: np.array):
    return np.abs(np.array(actual) - np.array(predicted)).sum() / np.array(actual).sum()


# подсчет метрики 1-wape
def quality(actual: np.array, predicted: np.array):
    return 1 - wape_metric(actual, predicted)


# метод для сохранения прогнозов в csv файл
def save_results(actual: np.array, predicted: np.array, comment=None):
    prefix = '' if comment is None else f'{comment}-'
    filename = f'{prefix}results.csv'
    
    pd.DataFrame({
        'actual': actual,
        'predicted': predicted
    }).to_csv(filename, index=False)


# метод для визулизации прогнозов
def plot_forecast(series_train, series_test, forecast, forecast_int=None):

    wape = wape_metric(series_test, forecast)
    quality = 1 - wape

    plt.figure(figsize=(12, 6))
    plt.title(f"WAPE: {wape:.2f}  Quality: {quality:.2f}")
    series_train.plot(label="train", color="b")
    series_test.plot(label="test", color="g")
    # forecast.index = series_test.index
    forecast.plot(label="forecast", color="r")
    if forecast_int is not None:
        plt.fill_between(
            series_test.index,
            forecast_int["lower"],
            forecast_int["upper"],
            alpha=0.2,
            color="dimgray",
        )
    plt.legend(prop={"size": 16})
    plt.show()


# Repository

class Repo:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        if 'Unnamed: 0' in self.df.columns:
            self.df.drop(columns=['Unnamed: 0'], inplace=True)

    def get_data(self, cust: int = None, dfu: str = None, date_from: str = None, date_to: str = None, date_interval: str = None):
        data = self.df
        
        if cust is not None:
            data = data[data['Customer'] == cust]

        if dfu is not None: 
            data = data[data['DFU'] == dfu]

        if 'Period' in data.columns:
            if date_from is not None:
                data = data[data['Period'] >= date_from]

            if date_to is not None:
                data = data[data['Period'] <= date_to]

            data['Period'] = pd.to_datetime(data['Period'])

            if date_interval is not None:
                data = group_data(data, date_interval)

        return data

    def show_data_for(self, cust: int, dfu: str, save_plot: bool = False):
        data = self.get_data(cust, dfu)
        data_cleaned = clean_data(data)
        max_bpv = data['BPV'].max() + 10

        fig, axs = plt.subplots(2, 1, figsize=(21, 16))
        top, bottom = axs[0], axs[1]

        top.plot(
            data["Period"].values.astype(np.datetime64),
            data['BPV']
        )
        bottom.plot(
            data_cleaned["Period"].values.astype(np.datetime64),
            data_cleaned['BPV'],
            color='green'
        )

        top.set_title(f'Продажи Клиенту {cust} - {dfu}')
        bottom.set_title(f'Продажи Клиенту {cust} - {dfu} (с обработкой выбросов)')
        top.set_ylim(0, max_bpv)
        bottom.set_ylim(0, max_bpv)

        if save_plot:
            # does not work if folder not exists
            plt.savefig(f'pictures/generated/cust{cust}_{dfu}_processed.pdf')

        plt.show()
