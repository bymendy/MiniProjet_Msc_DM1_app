# train_and_save_model.py
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, Binarizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

# Classe custom
class PositiveClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.where(np.isfinite(X), X, 0)
        return np.clip(X, a_min=0, a_max=None)

# Variables
num_log = ['duration', 'balance', 'campaign']
num_bin = ['pdays', 'previous']
cat_vars = ['job', 'education', 'contact', 'month', 'poutcome']
extra_vars = []

# Pipelines
log_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('clip', PositiveClipper()),
    ('log', FunctionTransformer(np.log1p, validate=True)),
    ('scale', StandardScaler())
])

bin_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value=0)),
    ('binary', Binarizer(threshold=0.0))
])

cat_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Prétraitement
preprocessor = ColumnTransformer(transformers=[
    ('log', log_pipeline, num_log),
    ('bin', bin_pipeline, num_bin),
    ('cat', cat_pipeline, cat_vars),
    ('cluster', 'passthrough', extra_vars)
])

# Pipeline complet
model_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('clf', LogisticRegression(max_iter=1000))
])

# Chargement des données
df = pd.read_csv("bank_clean.csv")  # Remplace par ton fichier nettoyé
X = df.drop(columns='y')
y = df['y'].map({'no': 0, 'yes': 1})

# Entraînement
model_pipeline.fit(X, y)

# Sauvegarde
joblib.dump(model_pipeline, "model_pipeline.pkl")
print("✅ Modèle entraîné et sauvegardé avec succès.")

