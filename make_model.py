
"""
Module for make pipline with model to predict user conversion action
"""
import datetime

# import pandas as pd
import numpy as np
import pickle
import dill
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

TARGET_ACTION_LIST = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
                      'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
                      'sub_submit_success', 'sub_car_request_submit_click']

FEATURE_LIST = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
                'device_category', 'device_os', 'device_brand', 'device_model', 'device_screen_resolution',
                'device_browser', 'geo_country', 'geo_city']


def filter_data(df):
    columns_to_drop = ['device_model']
    df_out = df.drop(columns=columns_to_drop).fillna('other').copy()
    return df_out


def get_decrease_unconv_dic(df):
    FEATURES_FOR_UNCONV = ('utm_campaign', 'utm_adcontent', 'utm_keyword', 'device_brand', 'device_screen_resolution',
                           'device_browser', 'geo_country', 'geo_city')
    df_out = df.copy()
    target_series = 'target'
    unconv_dic = {}
    for column in FEATURES_FOR_UNCONV:
        df_out[column] = df_out[column].fillna('other')
        unconv_set = set(df_out[column][df_out[target_series] == 1].to_list())
        unconv_dic[column] = unconv_set
    return unconv_dic


# функция для обобщения признаков без целевого действия
def decrease_unconv(df, unconv_dic):
    df_out = df.copy()
    for column, unconv_set in unconv_dic.items():
        df_out[column] = df_out[column].fillna('other')
        df_out[column] = df_out[column].apply(lambda x: x if x in unconv_set else 'out')
    return df_out


def get_increase_dic(df):
    FEATURES_FOR_BOUND = ('utm_campaign', 'utm_adcontent', 'utm_keyword', 'device_brand', 'device_screen_resolution',
                          'geo_country', 'geo_city')
    increase_dic = {}
    for column in FEATURES_FOR_BOUND:
        values_dic = {}
        for v, c in zip(df[column].value_counts().index, df[column].value_counts()):
            values_dic[v] = c
        increase_dic[column] = values_dic
    return increase_dic


# Функция для объединения редких категорий в одну
def category_increase_by_bound(df, column_value_count_dic):
    df_out = df.copy()
    bound = 30
    for column, value_count_dic in column_value_count_dic.items():
        value_count_dic['other'] = bound + 1
        value_count_dic['out'] = bound + 1
        df_out[column] = df_out[column].apply(lambda x: x if value_count_dic[x] > bound else 'bounded')
    return df_out


# Целевая обработка по каналу и типу привлечения
def utm_main_convertor(df):
    SOURCE_TUPLE = ('QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs', 'IZEXUFLARCUMynmHNBGo',
                    'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm')
    df_out = df.copy()
    df_out.utm_source = df_out.utm_source.fillna('other')
    df_out['utm_source'] = df_out['utm_source'].apply(lambda x: 1 if x in SOURCE_TUPLE else 0)
    df_out['utm_medium'] = df_out['utm_medium'].apply(lambda x: 1 if x in ('organic', 'referral', '(none)') else 0)
    return df_out


# Объединение браузера с содержанием названия Instagram
def browser_insta_group(df):
    df_out = df.drop(columns=['device_browser']).copy()
    df_out['device_browser'] = df.device_browser.apply(
        lambda x: 'Instagram' if x.split(' ')[0] == 'Instagram' else x)
    return df_out


# Загрузка данных и подготовка к обучению
def collect_data():
    # Загрузка данных по сессиям
    with open('data/ga_sessions.pkl', 'rb') as file:
        sessions = pickle.load(file)

    # Загрузка данных по действиям
    with open('data/ga_hits.pkl', 'rb') as file:
        ga_hits = pickle.load(file)

    # Вытаскиваем целевые данные
    ga_hits['target'] = ga_hits['event_action'].apply(lambda x: x in TARGET_ACTION_LIST)
    ga_hits = ga_hits.groupby(['session_id']).agg({'target': 'max'})

    sessions = sessions.merge(ga_hits, on='session_id', how='inner')

    sessions['target'] = sessions['target'].apply(lambda x: int(x))

    # Формируем таблицу для работы
    use_list = FEATURE_LIST.copy()
    use_list.append('target')
    sessions = sessions[use_list].drop_duplicates()
    return sessions


def main():
    # import pandas as pd

    RS = 42

    print('User Action Prediction Pipeline')

    # sessions.to_csv('prepared_sessions.csv')
    # sessions = pd.read_csv('prepared_sessions.csv', index_col=0)

    sessions = collect_data()

    increase_dic = get_increase_dic(sessions)
    conv_dic = get_decrease_unconv_dic(sessions)

    X = sessions.drop('target', axis=1)
    y = sessions['target']

    x_tr, x_test, y_tr, y_test = train_test_split(X, y, random_state=RS)

    print('Start models fitting')

    prepare_transformer = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        # ('imputer', SimpleImputer(strategy='constant', fill_value='other')),
        ('utm_main_convertor', FunctionTransformer(utm_main_convertor)),
        ('browser_insta_group', FunctionTransformer(browser_insta_group)),
        ('decrease_unconv', FunctionTransformer(decrease_unconv,
                                                kw_args={'unconv_dic': conv_dic})),
        ('category_increase_by_bound', FunctionTransformer(category_increase_by_bound,
                                                           kw_args={'column_value_count_dic': increase_dic}))
    ])

    numerical_transformer = Pipeline(steps=[
        # ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))  # sparse_output=False,
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include=object))
    ])

    models = (
        LogisticRegression(class_weight='balanced', random_state=RS),
        RandomForestClassifier(class_weight='balanced', random_state=RS),
        SVC(class_weight='balanced', random_state=RS, verbose=100)
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preporator', prepare_transformer),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, x_tr, y_tr, cv=4, scoring='roc_auc', verbose=100)
        print(f'model: {type(model).__name__}, roc_auc_score: {score.mean():.4f}')
        # print(f'model: {0}, roc_auc_score: {score.mean():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(x_tr, y_tr)
    prediction = best_pipe.predict(x_test)
    score = roc_auc_score(y_test, prediction)
    print(f'best model test: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc_score: {score:.4f}')

    best_pipe.fit(X, y)
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc_score: {best_score:.4f}')

    # print(confusion_matrix(y, prediction))

    with open('sber_user_action_pipe.dill', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                "name": "Sberbank user action prediction model",
                "author": "F Salih",
                "version": 1,
                "date": datetime.datetime.now(),
                "type": type(best_pipe.named_steps["classifier"]).__name__,
                "roc_auc_score": best_score
            }
        }, file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


