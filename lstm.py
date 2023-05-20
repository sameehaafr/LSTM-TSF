import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
import numpy as np
import tensorflow as tf
from numpy import array
import time
from tensorflow.python.keras import regularizers
from geopy.geocoders import Nominatim



DATA_URLS = ["data/LA_pm10_2020.csv", "data/LA_pm10_2021.csv", "data/LA_pm10_2022.csv"]
DATE = "Date"
DATA_COL = "Daily Mean PM10 Concentration"

def load_data(url):
    df = pd.read_csv(url)
    df[DATE] = pd.to_datetime(df[DATE]).dt.date
    return df

def merge_data():
    merged = pd.DataFrame()
    for url in DATA_URLS:
        dat = load_data(url)
        merged = pd.concat([merged, dat])
    merged[DATA_COL] = (merged[DATA_COL] - merged[DATA_COL].mean()) / merged[DATA_COL].std()
    return merged

def split_data(sequence, nsteps):
	X, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + nsteps
		if end_ix > len(sequence)-1:
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


def input_seq():
    merged = merge_data()
    daily_pm10 = merged[DATA_COL].values.tolist()
    daily_pm10 = daily_pm10[0:len(daily_pm10)-1]
    x_train, y_train = split_data(daily_pm10, 10)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train

def create_lstm(units, activation, nsteps, nfeatures, reg_input, dropout):
    model = Sequential()
    model.add(LSTM(units, activation=activation, input_shape=(nsteps, nfeatures), kernel_regularizer=regularizers.l2(reg_input)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', run_eagerly=True)
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()
    return model

#50, 'elu', 10, 1, 0.02, 0.6
def build_lstm(units, activation, nsteps, nfeatures, reg_input, dropout, trial):
    xtrain, ytrain = input_seq()
    model = create_lstm(units, activation, nsteps, nfeatures, reg_input, dropout)
    time1 = time.perf_counter()
    st.text("model training...")
    model.fit(xtrain, ytrain, epochs=25, verbose=0)
    time2 = time.perf_counter()
    st.text("model running time: " + str(time2-time1))
    model.save('models/lstm_model_{}.h5'.format(trial))
    return model


def load():
    model = tf.keras.models.load_model('models/lstm_model_10.h5')
    return model

def make_prediction(start, stop):
    start = int(start)
    stop = int(stop)
    print(start)
    model = load()
    merged = merge_data()
    merged[DATE] = pd.to_datetime(merged[DATE])
    #predict
    yhat = model.predict(merged[DATA_COL][start:stop], verbose=0)
    #normalize
    merged['daily_pm10_normalized'] = (merged[DATA_COL] - merged[DATA_COL].mean()) / merged[DATA_COL].std()
    yhat = (yhat - yhat.mean()) / yhat.std()
    yhat = np.array(yhat).flatten().tolist()
    actual = (merged['daily_pm10_normalized'][start:stop]).to_list()
    #display as table
    data = pd.DataFrame({'yhat': yhat, 'actual': actual, 'diff': np.abs(np.array(yhat) - np.array(actual)), 'date': merged[DATE][start:stop]})
    combined = pd.DataFrame(data, columns=['yhat', 'actual', 'diff', 'date'])
    return combined

def map():
    geolocator = Nominatim(user_agent="MyApp")
    merged = merge_data()
    coords = pd.DataFrame()
    latitude = []
    for i in range(len(merged)):
        location = geolocator.geocode(merged['Site Name'][i])
        latitude = latitude.append(location.latitude)
        st.text(latitude)

make_prediction(0,10)




#DISPLAY ----------------------------------------------------------------------------------------------------------------------
st.markdown('# LSTM for Time Series Forecasting')
st.caption('The objective of this project is to build an LSTM model that can forecast PM10 values in LA, California over X amount of time. The data used for this project was obtained from the EPA website.')
st.caption('This project was worked on during the 2022-23 school year as a part of the club ML@P (Machine Learning at Purdue). Check us out here: https://ml-purdue.github.io/')
st.caption('The code for this project can be found here: https://github.com/sameehaafr/LSTM-TSF/tree/master')


#DATA ----------------------------------------------------------------------------------------------------------------------
st.markdown('## Data')
st.caption('''To download the exact data I used:
1. Head to https://www.epa.gov/outdoor-air-quality-data
2. Click on "Download Daily Data"
3. Select "PM10" for "Pollutant"
4. Select "2020" for "Year"
5. Select "Los Angeles-Long Beach-Anaheim, CA" for "County" -> Make sure all sites are included
6. Click "Get Data"
7. Repeat for years 2021 and 2022''')
           
st.caption('We chose PM10 (Particulate Matter with a diameter of 10 micrometers or less) as our primary air quality metric because of its impact and presence. PM10 has a significant impact on human health and especially causing respiratory and cardiovascular issues. PM10 also has various emission sources, including from industrial activities, construction, vehicles, dust, etc. PM10 data is also widely available due to a plethora of air quality monitering stations and government agencies.' )
st.caption('Since this model is focused on forecasting PM10 values, I subsetted the data to only include the date and PM10 values. I also normalized the PM10 values to make it easier for the model to train on the data')
col1, col2 = st.columns(2)

with col1:
   st.text("Full Merged Dataset")
   merged_full = merge_data()
   st.dataframe(merged_full, use_container_width=True)

with col2:
   st.text("LA PM10 Values - Subset & Normalized")
   merged_sub = merged_full[[DATE, DATA_COL]]
   st.dataframe(merged_sub, use_container_width=True)

#MODEL ----------------------------------------------------------------------------------------------------------------------
st.markdown('## LSTM Model')
st.caption('We chose LSTM as our primary time series forecasting model for various reasons. Air pollution data often involves non-linear relationships and intricate patterns that may be difficult for linear models to capture. An LSTM is more flexible with this kind of task as it is designed to capture long term dependences in time series data and retain information from previous time steps. ')
st.caption('We used the Keras library to build our LSTM model. We used a single LSTM layer with 50 units, a dropout rate of 0.2, and a regularization rate of 0.02. We used the Adam optimizer and mean squared error as our loss function. We trained our model for 25 epochs.')
st.caption('''After multiple trials of training, we saw that our model's main problem was it was overfitting the data (trends were too accurate). To prevent overfiting we did the following: 
1. **Reduced the number of training epochs (25)**: Helps prevent overfitting by limiting the model's exposure to the training data. Stopping the training earlier can improve the model's ability to generalize.
2. **Chose the Exponential Linear Unit (ELU) activation function**: ELU can capture both positive and negative input regions (unlike ReLU which sets all negative input to 0) and returns non-zero outputs for negative input. By providing non-zero gradients for both positive and negative values, ELU helps maintain a more balanced and stable gradient flow throughout the network.
3. **Added L2 regularizer**: The L2 regularizer encourages the weights to be small, which reduces the model's complexity and prevents hte model from becoming too specialized to the training data (better generalized performance).
4. **Increased Dropout rate (0.6)**: We increased the drop out rate to reduce the model's reliance on specific inputs/features and encourages generalization.
''')
model = load()
st.markdown("## Model Summary")
model.summary(print_fn=lambda x: st.text(x))
st.code('''
def create_lstm(nsteps, nfeatures, units, activation, dropout):
    model = Sequential()
    model.add(LSTM(units, activation=activation, input_shape=(nsteps, nfeatures), kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', run_eagerly=True)
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()
    return model
''')

#PREDICTION ----------------------------------------------------------------------------------------------------------------------
st.markdown('## Make Predictions')
start = st.number_input('Insert a start value for the range', format='%i', min_value=0, value=0)
stop = st.number_input('Insert a stop value for the range', format='%i', min_value=1, value=8)
combined = make_prediction(start,stop)
combined['date'] = pd.to_datetime(combined['date']).dt.date
combined.index = combined['date']
st.dataframe(combined, use_container_width=True)
st.line_chart(combined[['yhat', 'actual']])

#METRICS ----------------------------------------------------------------------------------------------------------------------
st.header('Metrics')
mse = mean_squared_error(np.array(combined['actual']), np.array(combined['yhat']))
st.text("mean squared error: " + mse.astype(str))

map()