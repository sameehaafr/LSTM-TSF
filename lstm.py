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

def create_lstm(nsteps, nfeatures, units, activation, dropout):
    model = Sequential()
    model.add(LSTM(units, activation=activation, input_shape=(nsteps, nfeatures), kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', run_eagerly=True)
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()
    return model

def build_lstm(nsteps, nfeatures, units, activation, dropout, trial):
    xtrain, ytrain = input_seq()
    model = create_lstm(nsteps, nfeatures, units, activation, dropout)
    time1 = time.perf_counter()
    st.text("model training...")
    model.fit(xtrain, ytrain, epochs=25, verbose=0)
    time2 = time.perf_counter()
    st.text("model running time: " + str(time2-time1))
    model.save('models/lstm_model_{}.h5'.format(trial))

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

make_prediction(0,10)




#DISPLAY ----------------------------------------------------------------------------------------------------------------------
st.title('LSTM for Time Series Forecasting')
st.subheader('WIP...')
st.subheader('Forecasting PM10 in LA, California Over X Amount of Time')


#DATA ----------------------------------------------------------------------------------------------------------------------
st.header('Data')
col1, col2 = st.columns(2)

with col1:
   st.text("Full Dataset")
   merged_full = merge_data()
   st.dataframe(merged_full, use_container_width=True)

with col2:
   st.text("LA PM10 Values - Subset & Normalized")
   merged_sub = merged_full[[DATE, DATA_COL]]
   st.dataframe(merged_sub, use_container_width=True)

#MODEL ----------------------------------------------------------------------------------------------------------------------
st.header('LSTM Model')
model = load()
st.text("Model Summary")
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
st.header('Make Predictions')
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