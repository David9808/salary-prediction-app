import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('salary_data.csv')

# Columns to encode with label encoding
label_features = ['seniority', 'Size']
categorical_features = ['job_simp', 'job_state', 'Sector', 'Type of ownership']
boolean_features = ['aws', 'excel', 'python_yn', 'R_yn', 'spark', 'hourly']

# Apply label encoding
for col in label_features:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df[categorical_features + boolean_features + label_features]
y = df['avg_salary']

# Preprocess features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('bool', 'passthrough', boolean_features),
        ('label', 'passthrough', label_features)
    ]
)

# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, 'salary_prediction_pipeline.pkl')

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
