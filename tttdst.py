import sys
# print(sys.version)

import pandas as pd
import pickle
# import missingno as msno
# import matplotlib.pyplot as plt

from scipy import stats
# import geopy
# from geopy.geocoders import Nominatim


RS = 42

sessions = pd.read_csv('prepared_sessions.csv', index_col=0)


# функция для обобщения признаков без целевого действия
def decrease_unconv(in_series, target_series):
    series = in_series.fillna('other')
    on_conv_set = set(series[target_series==1].to_list())
    return series.apply(lambda x: x if x in on_conv_set else 'out')


# функция для объединения редких категорий в одну
def category_increase_by_bound(data_series, bound=30):
    values_dic = {}
    for v, c in zip(data_series.value_counts().index, data_series.value_counts()):
        values_dic[v] = c
    return data_series.apply(lambda x: x if values_dic[x] > bound else 'bounded' )



# ### 'utm_source' — канал привлечения

# Реклама в социальных сетях — все визиты с ga_sessions.utm_source in
# ('QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs', 'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw',
# 'gVRrcxiDQubJiljoTbGm').

# Рассматривал разные варианты работы с фичами. Некоторые подходы оставил как комментарии

#

sessions.utm_source = sessions.utm_source.fillna('other')

# Делаем признак канала привлечения только как "Реклама в социальных сетях"

sessions.utm_source = sessions.utm_source.apply(lambda x: 1 if x in ('QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs', 'IZEXUFLARCUMynmHNBGo',
                        'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm') else 0)


# ### 'utm_medium' —  тип привлечения

# Тим привлечения делаем как "Органический трафик"

sessions.utm_medium = sessions.utm_medium.apply(lambda x: 1 if x in ('organic', 'referral', '(none)') else 0)

# # category_increase_by_bound('utm_medium', bound=300)
# decrease_unconv('utm_medium')

# ### 'utm_campaign' — рекламная кампания

sessions.utm_campaign = decrease_unconv(sessions.utm_campaign, sessions.target)

sessions.utm_campaign = category_increase_by_bound(sessions.utm_campaign, bound=30)

# sessions.utm_adcontent = non_zero_user_transform('utm_adcontent')
sessions.utm_adcontent = decrease_unconv(sessions.utm_adcontent, sessions.target)

sessions.utm_adcontent = category_increase_by_bound(sessions.utm_adcontent, bound=30)


# ### utm_keyword — ключевое слово

sessions.utm_keyword = decrease_unconv(sessions.utm_keyword, sessions.target)

sessions.utm_keyword = category_increase_by_bound(sessions.utm_keyword, bound=30)

# ### device_category — тип устройства

# ### device_os — ОС устройства

# Заполним пустые значения значением other
columns_to_fill_other = ['device_os']
sessions[columns_to_fill_other[0]] = sessions[columns_to_fill_other[0]].fillna('other')

# ### device_brand — марка устройства

sessions.device_brand = decrease_unconv(sessions.device_brand, sessions.target)
sessions.device_brand = category_increase_by_bound(sessions.device_brand, bound=30)

# ### device_model — модель устройства

columns_to_drop = ['device_model']

sessions = sessions.drop(columns=columns_to_drop)

# ### device_screen_resolution — разрешение экрана

sessions.device_screen_resolution = decrease_unconv(sessions.device_screen_resolution, sessions.target)

sessions.device_screen_resolution = category_increase_by_bound(sessions.device_screen_resolution, bound=30)

# ### device_browser — браузер

sessions.device_browser = sessions.device_browser.apply(lambda x: 'Instagram' if x.split(' ')[0]=='Instagram' else x)

sessions.device_browser = decrease_unconv(sessions.device_browser, sessions.target)

# ### geo_country — страна

sessions.geo_country = decrease_unconv(sessions.geo_country, sessions.target)

sessions.geo_country = category_increase_by_bound(sessions.geo_country, bound=30)

# ### geo_city — город

# #### v2

sessions.geo_city = decrease_unconv(sessions.geo_city, sessions.target)

# V.2
columnst_to_transform = [ 'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
       'utm_keyword', 'device_category', 'device_os', 'device_brand',
       'device_screen_resolution', 'device_browser', 'geo_country', 'geo_city']


sessions = sessions.drop_duplicates()


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


ohe = OneHotEncoder(sparse=False)

ohe.fit(sessions[columnst_to_transform])

ohe_names = ohe.get_feature_names_out()

ohe_data = ohe.transform(sessions[columnst_to_transform])

sessions[ohe_names] = ohe_data

sessions = sessions.drop(columns=columnst_to_transform)

# ### Разбивка датасета

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
