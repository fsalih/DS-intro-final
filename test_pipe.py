import dill
import pandas as pd


with open('sber_user_action_pipe.dill', 'rb') as file:
    model = dill.load(file)

df = pd.read_csv('prepared_sessions.csv', index_col=0)

data = df.iloc[:5, :]

print(data)
y = model['model'].predict(data)

print(y)

