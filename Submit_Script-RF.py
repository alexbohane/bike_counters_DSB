import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

def _encode_dates(X):
    X = X.copy()
    X['date'] = pd.to_datetime(X['date'])
    X['year'] = X['date'].dt.year
    X['month'] = X['date'].dt.month
    X['day'] = X['date'].dt.day
    X['weekday'] = X['date'].dt.weekday
    X['hour'] = X['date'].dt.hour
    return X.drop(columns=['date'])

def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]

    preprocessor = ColumnTransformer(
        [
            ("date", date_encoder, ["date"]),
            ("cat", categorical_encoder, categorical_cols),
        ]
    )
    regressor = RandomForestRegressor(max_depth=7)

    pipe = make_pipeline(preprocessor, regressor)
    return pipe

# Load data
df_train = pd.read_parquet("../input/mdsb-2023/train.parquet")
df_test = pd.read_parquet("../input/mdsb-2023/final_test.parquet")

# Extract features and target
X_train = df_train.drop(columns=['log_bike_count'])
y_train = df_train['log_bike_count']

# Train the model
pipeline = get_estimator()
pipeline.fit(X_train, y_train)

# Predict on test data
X_test = df_test  
y_pred = pipeline.predict(X_test)

# Create submission file
results = pd.DataFrame({
    'Id': np.arange(y_pred.shape[0]),
    'log_bike_count': y_pred
})
results.to_csv("submission.csv", index=False)