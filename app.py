# app.py
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)


class PositiveClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.where(np.isfinite(X), X, 0)
        return np.clip(X, a_min=0, a_max=None)


model = joblib.load("models/model_pipeline.pkl")

JOBS       = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
              'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed']
CONTACTS   = ['cellular', 'telephone', 'unknown']
POUTCOMES  = ['success', 'failure', 'other', 'non_contacté']
EDUCATIONS = ['primary', 'secondary', 'tertiary']
MONTHS     = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
CLUSTERS   = [0, 1, 2]


@app.route("/")
def index():
    return render_template("index.html",
        jobs=JOBS, contacts=CONTACTS, poutcomes=POUTCOMES,
        educations=EDUCATIONS, months=MONTHS, clusters=CLUSTERS
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    client_input = pd.DataFrame([{
        'duration':  int(data['duration']),
        'balance':   int(data['balance']),
        'campaign':  int(data['campaign']),
        'pdays':     int(data['pdays']),
        'previous':  int(data['previous']),
        'job':       data['job'],
        'education': data['education'],
        'contact':   data['contact'],
        'month':     data['month'],
        'poutcome':  data['poutcome'],
        'cluster':   int(data['cluster']),
    }])

    proba = float(model.predict_proba(client_input)[0, 1])
    pred  = int(model.predict(client_input)[0])

    return jsonify({
        "proba": round(proba * 100, 2),
        "pred":  pred,
        "label": "Susceptible de souscrire" if pred == 1 else "Pas intéressé"
    })


if __name__ == "__main__":
    app.run(debug=True)
