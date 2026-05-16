import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("cleaned_earthquake_data.csv")

# ==============================
# FEATURE ENGINEERING
# ==============================
# Extract state from place
df['state'] = df['place'].str.split(',').str[-1].str.strip().str.lower()

# Features & target
X = df[['latitude', 'longitude', 'depth', 'year', 'month', 'state']]
y = df['mag']

# ==============================
# PREPROCESSING
# ==============================
num_cols = ['latitude', 'longitude', 'depth', 'year', 'month']
cat_cols = ['state']

preprocessor = ColumnTransformer([
    ('num', 'passthrough', num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# ==============================
# MODEL PIPELINE
# ==============================
pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestRegressor(n_estimators=150, random_state=42))
])

# ==============================
# TRAIN
# ==============================
pipeline.fit(X, y)

# ==============================
# SAVE MODEL
# ==============================
joblib.dump(pipeline, "earthquake_pipeline.pkl")

print("✅ Model trained and saved successfully!")