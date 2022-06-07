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

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


### Data Loading

def get_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, 
                            pd.DataFrame, pd.DataFrame]:
    """Получить все данные.
    Включает в себя загрузку архивов, распаковку, чтение и подготовку данных.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            [train_sales, test_sales, train_promo, test_promo]
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


def read_all_data(path: str = '') -> Tuple[pd.DataFrame, pd.DataFrame, 
                                           pd.DataFrame, pd.DataFrame]:
    """Создать датафреймы для трейн, тест и промо данных
    и заполнить трейн и тест недостающими нулями.

    Args:
        path (str, optional): Путь до папки с файлами.
            По умолчанию '' (текущая директория).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            [train_sales, test_sales, train_promo, test_promo]
    """

    return (
        fill_data(pd.read_excel(f'{path}train_sales.xlsx')),
        fill_data(pd.read_excel(f'{path}test_sales.xlsx')),
        pd.read_excel(f'{path}train_promo.xlsx'),
        pd.read_excel(f'{path}test_promo.xlsx')
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

    agg_dict = dict()
    for col in ['BPV', 'Total Sell-in', 'Promo_period']:
        if col in data.columns:
            agg_dict[col] = sum

    return data.groupby(
        ['DFU', 'Customer', pd.Grouper(key='Period', freq=date_interval)]
    ).agg(agg_dict).sort_values(by="Period").reset_index()


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


### Feature Engineering

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


def add_days(date: np.datetime64, num_days: int) -> np.datetime64:
    """Добавить к дате указанное число дней.

    Args:
        date (np.datetime64): Дата.
        num_days (int): Число дней.

    Returns:
        np.datetime64: Дата, увеличенная на указанное число дней.
    """

    return date + np.timedelta64(num_days, 'D')


def calculate_feature(df_sales: pd.DataFrame, df_promo: pd.DataFrame) -> pd.DataFrame:
    """Вспомогательный метод для генерации промо-признака

    Args:
        df_sales (pd.DataFrame): Датафрейм sales
        df_promo (pd.DataFrame): Датафрейм promo

    Returns:
        pd.DataFrame: Новый датафрейм со сгенерированными признаками
    """

    sales_new = df_sales.copy()

    for i, row_sales in sales_new.iterrows():
        for _, row_promo in df_promo.iterrows():
            for day in range(7):
                first_date = row_promo['First Date of shipment']
                end_date = row_promo['End Date of shipment']
                if first_date <= add_days(row_sales['Period'], day) <= end_date:
                    sales_new.loc[i, f'D{day}'] = 1

    return sales_new


def generate_feature(
    df_sales: pd.DataFrame,
    df_promo: pd.DataFrame,
    use_sod_correction=True
) -> pd.DataFrame:
    """Сгенерировать промо-признак.

    Args:
        df_sales (pd.DataFrame): Датафрейм sales.
        df_promo (pd.DataFrame): Датафрейм promo.
        use_sod_correction (bool, optional): Использовать ли коррекцию на SoD.
            По умолчанию True.

    Returns:
        [pd.DataFrame]: Новый датафрейм sales со сгенерированным промо-признаком.
    """

    cols_to_drop = list(set(df_promo.columns).difference(
        set(['Customer', 'DFU', 'First Date of shipment',
             'End Date of shipment', 'Units SoD'])
    ))
    promo = df_promo.copy()
    promo = promo.drop(columns=cols_to_drop)

    sales = df_sales.copy()
    added_cols = [f'D{i}' for i in range(7)]
    sales.loc[:, added_cols] = 0

    sales_new = calculate_feature(sales, promo)

    sales_new['Promo_period'] = sales_new[added_cols].sum(axis=1) / 7

    if use_sod_correction:
        sales_new['SOD_percentage'] = (
            sales_new['Total Sell-in'] - sales_new['BPV']) / sales_new['Total Sell-in']
        sales_new.loc[sales_new['SOD_percentage'] == 0, 'Promo_period'] = 0
        sales_new.loc[
            (sales_new['Promo_period'] == 0) &
            (sales_new['SOD_percentage'] != 0),
            'Promo_period'
        ] = sales_new['SOD_percentage']

    return sales_new.drop(columns=added_cols)


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


### Training

def forecast_simple(
    forecaster: any,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    X_test: pd.DataFrame,
    X_train: pd.DataFrame,
    show_plot: bool = True
) -> Tuple[np.array, np.array]:
    """Метод для предсказания простыми моделями из sktime.

    Args:
        forecaster (any): Модель.
        y_train (pd.DataFrame): Значения целевой переменной на трейне.
        y_test (pd.DataFrame): Значения целевой переменной на тесте.
        X_test (pd.DataFrame): Матрица признаков для трейна.
        X_train (pd.DataFrame, optional): Матрица признаков для теста.
        show_plot (bool, optional): Показать ли график с предсказанием.
            По умолчанию True (показать).

    Returns:
        Tuple[np.array, np.array]: Фактические и предсказанные значения целевой переменной.
    """

    forecaster.fit(y_train)
    y_pred = forecaster.predict(X_test.index)

    if show_plot:
        plt.figure(figsize=(12, 4))

        plt.plot(X_train['Period'], y_train, label='train')
        plt.plot(X_test['Period'], y_test, label='true test')
        plt.plot(X_test['Period'], y_pred, label='predicted')

        plt.legend()
        plt.title(f'{type(forecaster).__name__} forecast')
        plt.show()

    qual = quality(y_test.values, y_pred.values) * 100
    print(f"\n📝 Quality (1 - WAPE) : {qual}")

    return (y_test.values, y_pred.values)


def forecast_ml(
    model: BaseEstimator,
    y_train: np.array,
    X_train: np.array,
    y_test: np.array,
    X_test: np.array,
    scaler=StandardScaler(),
    save_plot: bool=False,
    save_results_csv: bool=False,
    file_prefix: str=None
) -> np.array:
    """Предсказание моделями машинного обучения.

    Args:
        model (RegressorMixin | BaseEstimator): Модель машиннго обучения.
        y_train (np.array): Значения целевой переменной для трейна.
        X_train (np.array): Матрица признаков для трейна.
        y_test (np.array): Значения целевой переменной для теста.
        X_test (np.array): Матрица признаков для трейна.
        scaler (BaseEstimator, optional): Скейлер для данных. По умолчанию StandardScaler().
            Если передать None, данные не будут скейлиться.
        save_plot (bool, optional): Сохранить картинку предсказания (True) или нет (False).
            По умолчанию False.
        save_results_csv (bool, optional): Сохранить результаты предсказания (True) или нет (False).
            По умолчанию False.
        file_prefix (_type_, optional): Префикс для картинки и файла с предсказанием.
            По умолчанию None (отсутствует).

    Returns:
        np.array: Предсказания целевой переменной.
    """

    period_test = X_test['Period']
    period_train = X_train['Period']

    X_train = X_train.drop(columns=['Period'])
    X_test = X_test.drop(columns=['Period'])

    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if save_results_csv:
        save_results(y_test, y_pred, comment=file_prefix)

    plt.figure(figsize=(12, 4))
    plt.plot(period_train, y_train, label='train')
    plt.plot(period_test, y_test, label='true test')
    plt.plot(period_test, y_pred, label='predicted')

    plt.legend()
    plt.title(f'{type(model).__name__} Model forecast')

    img_prefix = f'{file_prefix}-' if file_prefix is not None else ''
    if save_plot:
        plt.savefig(f'{img_prefix}plot.png')

    plt.show()

    print(f"\n📝 Quality (1 - WAPE) : {quality(y_test, y_pred) * 100}")

    return y_pred
