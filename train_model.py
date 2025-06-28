
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("data.csv")

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'sqft_above',
            'sqft_basement', 'yr_built', 'yr_renovated']
X = df[features]
y = df['price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor()
model.fit(X_scaled, y)

joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
