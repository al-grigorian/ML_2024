import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_car_data():
    data = pd.read_csv('car2.csv')
    return data

@st.cache_resource
def preprocess_car_data(data_in):
    '''
    Масштабирование признаков и обработка категориальных переменных
    '''
    data_out = data_in.copy()

    cat_cols = ['CarName', 'fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation', 
            'cylindernumber', 'enginetype', 'fuelsystem', 'doornumber']
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(data_out[cat_cols])
    encoded_feature_names = encoder.get_feature_names_out(cat_cols)
    data_out = pd.concat([data_out.drop(cat_cols, axis=1), pd.DataFrame(encoded_features.toarray(), columns=encoded_feature_names)], axis=1)
    
    scaler = StandardScaler()
    data_out[data_out.columns.difference(['price'] + encoded_feature_names)] = scaler.fit_transform(data_out[data_out.columns.difference(['price'] + encoded_feature_names)])
    
    return data_out

# Загрузка и предварительная обработка данных
data = load_car_data()
data_preprocessed = preprocess_car_data(data)

X = data_preprocessed.drop('price', axis=1)
y = data_preprocessed['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.sidebar.header('Car Price Predictor')
n_estimators_slider = st.sidebar.slider('Количество деревьев:', min_value=10, max_value=200, value=100, step=10)
max_depth_slider = st.sidebar.slider('Глубина дерева:', min_value=1, max_value=20, value=10, step=1)

model = RandomForestRegressor(n_estimators=n_estimators_slider, max_depth=max_depth_slider, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.subheader('Оценка качества модели')
# st.write(f'Среднеквадратическая ошибка: {mean_squared_error(y_test, y_pred):.2f}')

# st.subheader('Матрица ошибок')
errors = y_test - y_pred
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(errors, bins=30, kde=False, ax=ax)
ax.axvline(x=0, color='r', linestyle='--')  # Линия среднего значения ошибок
ax.set_title('Distribution of Prediction Errors')
ax.set_xlabel('Prediction Error')
ax.set_ylabel('Frequency')

# Использование st.pyplot для отображения графика в Streamlit
st.pyplot(fig)
