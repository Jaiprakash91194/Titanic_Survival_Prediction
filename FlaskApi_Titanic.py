# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 21:11:14 2022

@author: Jaiprakash
"""

import pickle
from flask import Flask, request
import flasgger
from flasgger import Swagger
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

pickle_in = open("rf_model.pkl", "rb")
rf_model = pickle.load(pickle_in)

df = pd.read_csv("titanicsurvival.csv")
df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Sex"] = df["Sex"].map({"male": 1, "female": 0}).astype(int)

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
ss = StandardScaler().fit(x_train)

app = Flask(__name__)
Swagger(app)


@app.route("/", methods=["Get"])
def predict_survival():
    """Let's predict Titanic Survival.
    Please provide input of Passenger class,Sex,Age,Fare
    ---
    parameters:
      - name: Gender
        in: query
        type: number
        required: True
      - name: Passenger
        in: query
        type: number
        required: True
      - name: Age
        in: query
        type: number
        required: True
      - name: Fare
        in: query
        type: number
        required: True
    responses:
        200:
            description: output the values
    """
    gender = request.args.get("Gender")
    p_class = request.args.get("Passenger")
    age = request.args.get("Age")
    fare = request.args.get("Fare")

    input_values = ss.transform([[gender, p_class, age, fare]])
    output = rf_model.predict(input_values)[0]
    if output == 1:
        return "The Passenger Survived in Titanic"
    else:
        return "The Passenger died in Titanic ship"


if __name__ == "__main__":
    app.run()
