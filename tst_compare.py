
import datetime

import pandas as pd
import numpy as np
import pickle
import dill
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer

RS = 42
TARGET_ACTION_LIST = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
                      'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
                      'sub_submit_success', 'sub_car_request_submit_click']

FEATURE_LIST = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
                'device_category', 'device_os', 'device_brand', 'device_model', 'device_screen_resolution',
                'device_browser', 'geo_country', 'geo_city']

SOURCE_TUPLE = ('QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs', 'IZEXUFLARCUMynmHNBGo',
                'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm')

FEATURES_FOR_UNCONV = ('utm_campaign', 'utm_adcontent', 'utm_keyword', 'device_brand', 'device_screen_resolution',
                       'device_browser', 'geo_country', 'geo_city')

FEATURES_FOR_BOUND = ('utm_campaign', 'utm_adcontent', 'utm_keyword', 'device_brand', 'device_screen_resolution',
                      'geo_country', 'geo_city')


def filter_data(df):
    # import pandas as pd
    # print(type(df))
    columns_to_drop = ['device_model', 'utm_source', 'utm_medium', 'device_browser']
    df_out = df.drop(columns=columns_to_drop).fillna('other').copy()
    # print(df_out.isna().sum())

    df_out['utm_source'] = df['utm_source'].apply(lambda x: 1 if x in SOURCE_TUPLE else 0)
    df_out['utm_medium'] = df['utm_medium'].apply(lambda x: 1 if x in ('organic', 'referral', '(none)') else 0)

    df_out['device_browser'] = df.device_browser.apply(
        lambda x: 'Instagram' if x.split(' ')[0] == 'Instagram' else x)

    target_series = 'target'
    for column in FEATURES_FOR_UNCONV:
        # df_out[column] = df[column].fillna('other')
        on_conv_set = set(df_out[column][df[target_series] == 1].to_list())
        df_out[column] = df_out[column].apply(lambda x: x if x in on_conv_set else 'out')

    bound = 30
    for column in FEATURES_FOR_BOUND:
        values_dic = {}
        for v, c in zip(df_out[column].value_counts().index, df_out[column].value_counts()):
            values_dic[v] = c
        df_out[column] = df_out[column].apply(lambda x: x if values_dic[x] > bound else 'bounded')

    columnst_to_transform = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
                             'utm_keyword', 'device_category', 'device_os', 'device_brand',
                             'device_screen_resolution', 'device_browser', 'geo_country', 'geo_city']
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(df_out[columnst_to_transform])
    df_out[ohe.get_feature_names_out()] = ohe.transform(df_out[columnst_to_transform])
    df_out = df_out.drop(columns=columnst_to_transform)

    # df_out.to_csv('test_out.csv')
    return df_out  # .drop(columns=['target'])


sessions = pd.read_csv('prepared_sessions.csv', index_col=0)

sessions = filter_data(sessions)

x_tr, x_test, y_tr, y_test = train_test_split(sessions.drop(columns='target'), sessions['target'], random_state=RS)

# ### Logistic Regression

logreg = LogisticRegression(class_weight='balanced',
                           random_state=RS)

logreg.fit(x_tr, y_tr)

X, y = sessions.drop(columns='target'), sessions['target']

y_pred_lrx = logreg.predict(X)
print(roc_auc_score(y, y_pred_lrx))

y_pred_lr = logreg.predict(x_test)
print('test', roc_auc_score(y_test, y_pred_lr))