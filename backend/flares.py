import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
import warnings
from PIL import Image
from plotly import graph_objects as go
warnings.filterwarnings("ignore")
import streamlit as st
from keras.models import load_model
st.set_page_config(page_title="Solar Activity", page_icon=":tada", layout = "wide")

with st.container():
    st.subheader("The below graphs are displaying the variation in the solar activity by using the FLARE data")

st.title("Visualising the data")
solar_flare = pd.read_csv("backend/hessi.solar.flare.2002to2016.csv", parse_dates=["start.date"],
                          dtype={"energy.kev": "category", "total.counts": "float64", "active.region.ar": "category"})
solar_flare.head()

solar_flare['start.date'] = solar_flare['start.date'].dt.strftime('%Y-%m-%d')
solar_flare['start_datetime'] = pd.to_datetime(solar_flare['start.date'] + ' ' + solar_flare['peak'])
solar_flare['start_datetime']
solar_flare.set_index('start_datetime', inplace=True)
columns_to_drop = ['start.date', 'start.time', 'peak', 'end']
solar_flare = solar_flare.drop(columns=columns_to_drop)

# solar_flare.info()

# solar_flare.describe()
# duplicates_count = solar_flare['flare'].duplicated(keep=False).sum()

# value_counts = solar_flare['flare'].value_counts()

# duplicates = value_counts[value_counts>1]
# print(duplicates)
# print(f"Number of duplicates in the 'flare' column: {len(duplicates)}")

# fig0=plt.figure(figsize=(12, 6))
# plt.plot(solar_flare.index, solar_flare['peak.c/s'], label='Peak Count Rate')
# plt.title('Solar Flare Time Series')
# plt.xlabel('Time')
# plt.ylabel('Peak.c/s')
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.savefig('backend/flaresGraph1.png')
# st.pyplot(fig0)


# def model_forecast(model, series, window_size):
#     ds = tf.data.Dataset.from_tensor_slices(series)
#     ds = ds.window(window_size, shift=1, drop_remainder=True)
#     ds = ds.flat_map(lambda w: w.batch(window_size))
#     ds = ds.batch(batch_size).prefetch(1)
#     forecast = model.predict(ds)  # To predict
#     return forecast

# # Preparing the data
# import pandas as pd
# import datetime
# # solar_flare = pd.read_csv("hessi.solar.flare.2002to2016.csv", parse_dates=["start.date"],
# #                           dtype={"energy.kev": "category", "total.counts": "float64", "active.region.ar": "category"})
# series = solar_flare['peak.c/s'].to_numpy()
# # date = data['Date'].values
# # date=np.array([datetime.datetime.strptime(x, '%Y-%m-%d') for x in date])
# # time=np.array([x.month for x in date])

# time= solar_flare.index
# # Splitting the data into train and test
# split_time = int(len(series)*0.9)          # 90% of the original data is for training
# time_train = time[:split_time]
# x_train = series[:split_time]
# time_valid = time[split_time:] #date
# x_valid = series[split_time:] #monthly sunspot number
# print(f"There are {len(x_train)} training samples and {len(x_valid)} validation samples.")

# print(time_valid)

# # Parameters
# delta = 1                      # Huber loss
# window_size = 10               # For dataset
# batch_size = 145               # For dataset
# shuffle_buffer_size= 900       # Shuffling the dataset randomly
# epochs = 100                   # For optimal learning rate
# train_epochs = epochs + 0    # Training epochs
# momentum_sgd = 0.9
# LSTM_model=load_model('backend/FLARES1.h5')

# rnn_forecast = model_forecast(LSTM_model, series[:, np.newaxis], window_size)
# rnn_forecast = rnn_forecast[split_time - window_size:-1, 0]  # rnn_forecast[-328:-1, 0]

# # Plots
# fig1=plt.figure(figsize=(15, 6))
# plt.plot(time_valid, x_valid)
# plt.plot(time_valid, rnn_forecast)

# # plt.grid(visible=True, axis='both',)
# plt.title("")
# plt.legend(["Validation Data", "Predicted Data"])
# plt.show()
# # plt.savefig('flaresGraph1.png')
# st.pyplot(fig1)

# image = Image.open('backend/flaresGraph1.png')
# with open("backend/flaresGraph1.png", "rb") as file:
#     btn=st.download_button(label="Download graph", data=file, file_name='/flaresGraph1.png',mime="image/png")



# GRU_model=load_model('backend/GRUFLARES1.h5')
# import tensorflow as tf
# rnn_forecast = model_forecast(GRU_model, series[:, np.newaxis], window_size)
# rnn_forecast = rnn_forecast[split_time - window_size:-1, 0]  # rnn_forecast[-328:-1, 0]

# # Plots
# fig2=plt.figure(figsize=(15, 6))
# plt.plot(time_valid, x_valid)
# plt.plot(time_valid, rnn_forecast)

# # plt.grid(visible=True, axis='both',)
# plt.title("")
# plt.legend(["Validation Data", "Predicted Data"])
# plt.show()
# # plt.savefig('flaresGraph2.png')
# st.pyplot(fig2)

# image = Image.open('backend/flaresGraph2.png')
# with open("backend/flaresGraph2.png", "rb") as file:
#     btn=st.download_button(label="Download graph", data=file, file_name='backend/flaresGraph2.png',mime="image/png")



fig = go.Figure()
fig.add_trace(go.Scatter(x=solar_flare.index, y=solar_flare['peak.c/s'], name ='LSTM'))
fig.layout.update(title_text="Solar Flares data")
st.plotly_chart(fig)



plotly_html = fig.to_html()

    # Open the existing HTML file and read its content
with open('frontend/flares.html', 'r') as f:
        html_content = f.read()

    # Find the location where you want to insert the Plotly graph
    # For example, let's assume you want to insert it within a <div> with id="plotly-graph"
    # Replace 'plotly-graph' with the id of the <div> in your HTML file
insert_location = '<div id="flares-plotly"></div>'

    # Insert the Plotly HTML code into the existing HTML content
updated_html_content = html_content.replace(insert_location, plotly_html)

    # Write the updated HTML content back to the file
with open('frontend/updated-flares.html', 'w',encoding='utf-8') as f:
    f.write(updated_html_content)
