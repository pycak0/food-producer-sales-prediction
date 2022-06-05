"""Файл с общими полезными методами.
"""

from urllib.request import urlretrieve
from zipfile import ZipFile

from itertools import product
from typing import Tuple
from typing import Iterable

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### Data Loading

def get_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Получить все данные.
    Включает в себя загрузку архивов, распаковку, чтение и подготовку данных.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: [train_sales, test_sales, promo]
    """

    url = 'https://drive.google.com/uc?export=download&id=1ndm03Vd4-gW2Q2iKkCjFkGnMf5X98RGy&confirm=t'
    file_name = 'all_data.zip'

    print('⬇️ Loading data...')
    load(url, file_name)

    print('🔄 Unzipping data...')
    unzip(file_name)

    print('🧹 Cleaning up...')
    remove(file_name)

    print('📜 Creating DataFrames & filling zeros...')
    all_data = read_all_data()

    print('✅ Completed.')
    return all_data


def load(url: str, file_name: str):
    """Скачать файл.

    Args:
        url (str): URL файла.
        file_name (str): Имя файла для сохранения.
    """

    urlretrieve(url, file_name)


def unzip(zip_path: str, target_dir: str = ''):
    """Распаковать zip архив.

    Args:
        zip_path (str): Путь до архива.
        target_dir (str, optional): Путь, куда распаковать файл.
            По умолчанию '' (в текущей директории).
    """

    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)


def remove(path: str):
    """Удалить по заданному пути.

    Args:
        path (str): Путь до файла/папки.
    """

    try:
        os.remove(path)
    except OSError:
        pass


def read_all_data(path: str = '') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Создать датафреймы для трейн, тест и промо данных
    и заполнить трейн и тест недостающими нулями.

    Args:
        path (str, optional): Путь до папки с файлами.
            По умолчанию '' (текущая директория).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: [train_sales, test_sales, promo]
    """

    return (
        fill_data(pd.read_excel(f'{path}train_sales.xlsx')),
        fill_data(pd.read_excel(f'{path}test_sales.xlsx')),
        pd.read_excel(f'{path}train_promo.xlsx')
    )


def fill_data(data: pd.DataFrame) -> pd.DataFrame:
    """Заполнить данные недостающими нулями.

    Args:
        data (pd.DataFrame): Исходный датафрейм.

    Returns:
        pd.DataFrame: Новый датафрейм, дополненный недостающими нулями.
    """

    dfus = data['DFU'].unique()
    customers = data['Customer'].unique()
    dtr = pd.date_range(
        data['Period'].min(),
        data['Period'].max(),
        freq='W-MON'
    )
    index_cols = ['Period', 'DFU', 'Customer']

    zeros_df = pd.DataFrame(
        product(dfus, customers, dtr.values, [0], [0]),
        columns=data.columns
    ).set_index(index_cols)
    new_data = data.copy().set_index(index_cols)

    index_diff = zeros_df.index.difference(
        new_data.index
    )

    return new_data.append(zeros_df.loc[index_diff])\
        .sort_values(by="Period")\
        .reset_index()


### Data Manipulation

def group_data(data: pd.DataFrame, date_interval: str) -> pd.DataFrame:
    """Агрегировать данные по неделям/месяцам и т.д.

    Args:
        data (pd.DataFrame): Данные для агрегации, должны содержать столбцы
            'DFU', 'Customer', 'Period', 'BPV' и 'Total Sell-in'.
        date_interval (str): Интервал агрегации.
            Допустимые значения см. в разделе 'Offset aliases' документации pandas.

    Returns:
        pd.DataFrame: Датафрейм со значениями BPV и Total Sell-in,
            агрегированными по заданному интервалу.
    """

    return data.groupby(
        ['DFU', 'Customer', data['Period'].dt.to_period(date_interval)]
    ).agg(
        {'BPV': sum, 'Total Sell-in': sum}
    ).reset_index()


def clean_data(data: pd.DataFrame, replace_with: str = '3sigma') -> pd.DataFrame:
    """Удалить выбросы в данных.

    Args:
        data (pd.DataFrame): Исходные данные.
        replace_with (str, optional): 'mean' или '3sigma'. По умолчанию '3sigma'.

    Returns:
        pd.DataFrame: Копия исходных данных с обработкой выбросов.
    """

    sigma = data['BPV'].std()
    mean = data['BPV'].mean()

    data_cleaned = data.copy()
    replacement = mean if replace_with == 'mean' else 3 * sigma
    data_cleaned.loc[data_cleaned['BPV'] > 3 * sigma, 'BPV'] = replacement
    return data_cleaned


def make_date_features(X: pd.DataFrame, date_col_name='Period') -> pd.DataFrame:
    """Создать новые признаки (день, месяц, год, квартал).

    Args:
        X (pd.DataFrame): Исходный датафрейм.
        date_col_name (str, optional): Название столбца с датой в X. По умолчанию 'Period'.

    Returns:
        pd.DataFrame: Копия переданного датафрейма с добавленными признаками.
    """

    X_new = X.copy()
    date_info = X[date_col_name].dt
    X_new['year'] = date_info.year
    X_new['month'] = date_info.month
    X_new['day'] = date_info.day
    X_new['week'] = date_info.week
    X_new['Q'] = date_info.quarter

    return X_new


### Metrics

def wape_metric(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Подсчитать метрику WAPE.

    Args:
        actual (Iterable[float]): Последовательность реальных значений целевой переменной.
        predicted (Iterable[float]): Последовательность предсказанных значений.

    Returns:
        float: Подсчитанная метрика WAPE.
    """

    return np.abs(np.array(actual) - np.array(predicted)).sum() / np.array(actual).sum()


def quality(actual: np.array, predicted: np.array) -> float:
    """Подсчитать метрику (1 - WAPE).

    Args:
        actual (np.array): Последовательность реальных значений целевой переменной.
        predicted (np.array): Последовательность предсказанных значений.

    Returns:
        float: Подсчитанная метрика (1 - WAPE).
    """

    return 1 - wape_metric(actual, predicted)


def save_results(actual: np.array, predicted: np.array, comment=None):
    """Сохранить прогнозы модели и реальные значения целевой переменной в формате .csv.

    Args:
        actual (np.array): Последовательность реальных значений целевой переменной.
        predicted (np.array): Последовательность предсказанных значений.
        comment ([type], optional): Префикс для названия файла. По умолчанию None (не используется).
    """

    prefix = '' if comment is None else f'{comment}-'
    filename = f'{prefix}results.csv'

    pd.DataFrame({
        'actual': actual,
        'predicted': predicted
    }).to_csv(filename, index=False)


def plot_forecast(
    series_train: pd.Series,
    series_test: pd.Series,
    forecast: pd.Series,
    forecast_int=None
):
    """Визуализировать прогноз модели.

    Args:
        series_train (pd.Series): Тренировочные данные BPV.
        series_test (pd.Series): Тестовые данные BPV.
        forecast (pd.Series): Предсказанные данные BPV.
        forecast_int ([type], optional): Доверительный интервал. По умолчанию None.
    """

    wape = wape_metric(series_test, forecast)
    qual = 1 - wape

    plt.figure(figsize=(12, 6))
    plt.title(f"WAPE: {wape:.2f}  Quality: {qual:.2f}")

    series_train.plot(label="train", color="b")
    series_test.plot(label="test", color="g")
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


### Repository

class Repo:
    """Класс-обертка над DataFrame для удобного выбора нужных данных.

    init args:
        df (pd.DataFrame): Исходные данные со столбцами
            'Period', 'DFU' и 'Customer'.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        if 'Unnamed: 0' in self.df.columns:
            self.df.drop(columns=['Unnamed: 0'], inplace=True)

    def get_data(
        self,
        cust: int = None,
        dfu: str = None,
        date_from: str = None,
        date_to: str = None,
        date_interval: str = None
    ) -> pd.DataFrame:
        """Получить данные по заданным параметрам.
        Если параметр равен None, то фильтрация по нему не производится.

        Args:
            cust (int, optional): Идентификатор клиента. По умолчанию None.
            dfu (str, optional): DFU - название продукта. По умолчанию None.
            date_from (str, optional): С какой даты выбрать данные. По умолчанию None.
            date_to (str, optional): По какую дату выбрать данные. По умолчанию None.
            date_interval (str, optional): Интервал для агрегации. По умолчанию None.
                Допустимые значения см. в методе group_data.

        Returns:
            pd.DataFrame: Результат выбора данных по заданным параметрам.
        """

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
        """Выбрать из данных пару [DFU x Customer] и построить для них график.

        Args:
            cust (int): Идентификатор клиента.
            dfu (str): DFU - название продукта.
            save_plot (bool, optional): Сохранить изображение графика или нет.
                По умолчанию False (не сохранять).
        """

        data = self.get_data(cust, dfu)
        data_cleaned = clean_data(data)
        max_bpv = data['BPV'].max() + 10

        _, axs = plt.subplots(2, 1, figsize=(21, 16))
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
