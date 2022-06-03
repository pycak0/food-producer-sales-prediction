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

def group_data(data: pd.DataFrame, date_interval: str):
    return data.groupby(
        ['DFU', 'Customer', data['Period'].dt.to_period(date_interval)]
    ).agg(
        {'BPV': sum, 'Total Sell-in': sum}
    ).reset_index()


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

def wape_metric(actual: np.array, predicted: np.array):
    return np.abs(actual - predicted).sum() / actual.sum()


def quality(actual: np.array, predicted: np.array):
    return 1 - wape_metric(actual, predicted)


def save_results(actual: np.array, predicted: np.array, comment=None):
    prefix = '' if comment is None else f'{comment}-'
    filename = f'{prefix}results.csv'
    
    pd.DataFrame({
        'actual': actual,
        'predicted': predicted
    }).to_csv(filename, index=False)


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
