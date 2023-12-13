import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from datetime import datetime

def _encode_dates(X):
    X = X.copy()
    X['date'] = pd.to_datetime(X['date'])
    X['year'] = X['date'].dt.year
    X['month'] = X['date'].dt.month
    X['day'] = X['date'].dt.day
    X['weekday'] = X['date'].dt.weekday
    X['hour'] = X['date'].dt.hour
    return X

def _distance_from_paris_centre(X):
    X = X.copy()  # modify a copy of X
    X.loc[:, 'distance_paris_centre'] = np.sqrt((X['latitude']-48.8565)**2 + (X['longitude']-2.3426)**2)
    
    return X


conf = [(datetime(2020, 3, 17), datetime(2020, 5, 10)), (datetime(2020, 10, 30), datetime(2020, 11, 27)), (datetime(2021, 4, 3), datetime(2021, 5, 2))]
conf_a = [(datetime(2020, 5, 11), datetime(2020, 6, 1)), (datetime(2020, 11, 28), datetime(2020, 12, 15))]
c_v = [(datetime(2020,10,15), datetime(2020,10,29)), (datetime(2020, 12, 16), datetime(2021, 4, 2)), (datetime(2021, 5, 3), datetime(2021, 6, 20))]

def is_in_range(date):
    for start, end in conf:
        if start <= date <= end:
            return 1
    for start, end in conf_a:
        if start <= date <= end:
            return 1
    for start, end in c_v:
        if start <= date <= end:
            return 1
    if date == datetime(2020, 12, 25):
        return 2
    else:
        return 0

def apply_confinement_status(X):
    X = X.copy()  
    X['Confinement'] = X['date'].apply(is_in_range)
    return X

def covid_scaled_count(X):
    X = X.copy()

    # Define the COVID impact factors
    covid_impact_conf = 1.23
    covid_impact_cv = 1.16

    # Define date ranges for confinements (conf) and couvre-feu (c_v)
    conf = [(datetime(2020, 10, 30), datetime(2020, 12, 15)), (datetime(2021, 4, 3), datetime(2021, 5, 2))]
    c_v = [(datetime(2020, 10, 15), datetime(2020, 10, 29)), (datetime(2020, 12, 16), datetime(2021, 4, 2)), (datetime(2021, 5, 3), datetime(2021, 6, 20))]

    # Initialize a column for scaled bike counts
    X['scaled_bike_count'] = X['bike_count']

    # Apply the conf impact factor
    for start, end in conf:
        conf_mask = (X['date'] >= start) & (X['date'] <= end)
        X.loc[conf_mask, 'scaled_bike_count'] *= covid_impact_conf

    # Apply the c_v impact factor
    for start, end in c_v:
        cv_mask = (X['date'] >= start) & (X['date'] <= end)
        X.loc[cv_mask, 'scaled_bike_count'] *= covid_impact_cv

    return X

def covid_scaled_log_count(X):
    X = X.copy()
    # Apply ln(x+1) transformation to the scaled_bike_count
    X['scaled_log_bike_count'] = np.log1p(X['scaled_bike_count'])
    return X


def column_to_drop(X):
    #return X.drop(['date'], axis=1)
    return X.drop(['date', 'longitude', 'latitude', 'counter_installation_date',
                  'coordinates', 'counter_technical_id','counter_id', 'site_name', 'site_id'], axis=1)



###########

def combined_transformer(X):
    X = _encode_dates(X)
    X = _distance_from_paris_centre(X)
    X = apply_confinement_status(X)

    X = column_to_drop(X)
    print(X.info())
    return X




def get_estimator():
    
    data_encoder = FunctionTransformer(combined_transformer)
    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name"]

    preprocessor = ColumnTransformer(
        [
#             ("prepro", data_encoder, ["date",'longitude', 'latitude']),
            ("cat", categorical_encoder, categorical_cols),
        ],remainder='passthrough'
    )
    
#     data_capture_step = DataCaptureTransformer()
    
    regressor = XGBRegressor(
        objective='reg:squarederror',
        reg_alpha = 0.005,
        n_estimators = 200,
        max_depth = 7,
        learning_rate=0.2,
    )

    
    pipe = make_pipeline(data_encoder, preprocessor, regressor)
    return pipe

# Load data
df_train = pd.read_parquet("../input/mdsb-2023/train.parquet")
df_test = pd.read_parquet("../input/mdsb-2023/final_test.parquet")

df_train['site_name'] = df_train['site_name'].astype(str)
df_train['site_name'] = df_train['site_name'].replace(r'^Pont des Invalides (S-N|N-S)$', 'Pont des Invalides', regex=True)
df_train['site_name'] = df_train['site_name'].astype('category')

df_test['site_name'] = df_test['site_name'].astype(str)
df_test['site_name'] = df_test['site_name'].replace(r'^Pont des Invalides (S-N|N-S)$', 'Pont des Invalides', regex=True)
df_test['site_name'] = df_test['site_name'].astype('category')

### IMPLEMENT SCALED FACTOR FOR COVID on df_train
df_train = covid_scaled_count(df_train)
df_train = covid_scaled_log_count(df_train)

df_train.head()
# Extract features and target
X_train = df_train.drop(columns=['log_bike_count', 'bike_count', 'scaled_bike_count', 'scaled_log_bike_count'])
y_train = df_train['scaled_log_bike_count']

# Train the model with cross-validation
pipeline = get_estimator()


# Fit the model on entire training data
pipeline.fit(X_train, y_train)

#transformed_data = pipeline.named_steps['datacapturetransformer'].data

# Predict on test data
X_test = df_test
y_pred = pipeline.predict(X_test)


# Create submission file
results = pd.DataFrame({
    'Id': np.arange(y_pred.shape[0]),
    'log_bike_count': y_pred
})
results.to_csv("submission.csv", index=False)