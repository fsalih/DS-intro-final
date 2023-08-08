
from fastapi import FastAPI
import pandas as pd
import dill
from pydantic import BaseModel

app = FastAPI()

with open('sber_user_action_pipe.dill', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    Result: int


@app.get('/status')
def status():
    return "Working"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.model_dump()])
    # df = pd.DataFrame.from_dict([form.dict()])
    print(df)
    y = model['model'].predict(df)

    return {
        'Result': y[0]
    }


def intro():
    print('Final work on the course "Introduction to Data Science"')
    print('To start application use command: "uvicorn main:app --reload"')


if __name__ == '__main__':
    intro()





