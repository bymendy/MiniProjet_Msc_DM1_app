# train_and_save_model.py
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, Binarizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from preprocessing import PositiveClipper  # ← import externe (plus de définition ici)

# Variables
num_log  = ['duration', 'balance', 'campaign']
num_bin  = ['pdays', 'previous']
cat_vars = ['job', 'education', 'contact', 'month', 'poutcome']
extra_vars = []

# Pipelines
log_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('clip',   PositiveClipper()),
    ('log',    FunctionTransformer(np.log1p, validate=True)),
    ('scale',  StandardScaler())
])

bin_pipeline = Pipeline([
    ('impute',  SimpleImputer(strategy='constant', fill_value=0)),
    ('binary',  Binarizer(threshold=0.0))
])

cat_pipeline = Pipeline([
    ('impute',  SimpleImputer(strategy='most_frequent')),
    ('encode',  OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Prétraitement
preprocessor = ColumnTransformer(transformers=[
    ('log',     log_pipeline,  num_log),
    ('bin',     bin_pipeline,  num_bin),
    ('cat',     cat_pipeline,  cat_vars),
    ('cluster', 'passthrough', extra_vars)
])

# Pipeline complet
model_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('clf',        LogisticRegression(max_iter=1000))
])

# Chargement des données
df = pd.read_csv("data/processed/bank_clean.csv")  # ← chemin mis à jour
X  = df.drop(columns='y')
y  = df['y'].map({'no': 0, 'yes': 1})

# Entraînement
model_pipeline.fit(X, y)

# Sauvegarde
joblib.dump(model_pipeline, "models/model_pipeline.pkl")  # ← chemin mis à jour
print("✅ Modèle entraîné et sauvegardé avec succès.")
