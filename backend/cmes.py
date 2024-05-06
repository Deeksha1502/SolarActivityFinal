
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

from keras.models import load_model

st.set_page_config(page_title="Solar Activity", page_icon=":tada", layout = "wide")

with st.container():
    st.subheader("The below graphs are displaying the variation in the solar activity by using the CMES data")

st.title("Visualising the data")

data = pd.read_csv("CME Data.csv")
data.head()

data.isnull().sum()

# Get descriptive statistics
data.describe()

# Check for duplicate rows
data.duplicated().sum()

# Get the correlation matrix
cor=data.corr()

fig1= plt.figure(figsize=(20, 15))
sns.heatmap(cor, annot=False, cmap="coolwarm")

plt.show()

st.pyplot(fig1)

# Assuming y is your target variable
X = data[['Total unsigned flux',	'Mean gradient of total field',	'Mean current helicity (Bz contribution)'	,'Mean photospheric magnetic free energy',	'Fraction of Area with Shear > 45 deg'	,'Total unsigned current helicity',	'Mean gradient of horizontal field',	'Mean characteristic twist parameter, alpha'	,'Mean angle of field from radial',	'Mean gradient of vertical field',	'Mean vertical current density',	'Total unsigned vertical current',	'Sum of the modulus of the net current per polarity',	'Total photospheric magnetic free energy density',	'Mean shear angle',	'Area of strong field pixels in the active region',	'Sum of flux near polarity inversion line',	'Absolute value of the net current helicity']]
y = data['target']

# Use RandomForestRegressor for feature selection
model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=4)
fit = rfe.fit(X, y)
data.info()
# Display the selected features
selected_features = X.columns[fit.support_]
print("Selected Features:", selected_features)

new_data = data[['Total unsigned flux',
       'Total photospheric magnetic free energy density',
       'Area of strong field pixels in the active region',
       'Absolute value of the net current helicity','target']]
new_data.describe()

new_data.head()

X = new_data[['Total unsigned flux', 'Total photospheric magnetic free energy density', 'Area of strong field pixels in the active region', 'Absolute value of the net current helicity']]
y = new_data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the input data for GRU
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

lrs = 1e-8 * (10**(np.arange(100)/20))
lrs

from keras.layers import LSTM, Dense
from keras.models import Sequential
# Build the GRU model
Lmodel = Sequential()
Lmodel.add(LSTM(units=50, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
Lmodel.add(Dense(units=1, activation='sigmoid'))
Lmodel.summary()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20), verbose = 0) # lr --> starting lr * 10^(0/20), starting lr * 10^(1/20), so on..

# Stochastic Gradient Desect as the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9)


# Compile the model
Lmodel.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# Train the model
history=Lmodel.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[lr_schedule])
min_loss = min(history.history['loss'])
idx_min_loss = history.history['loss'].index(min_loss)
opt_lr = lrs[idx_min_loss]

# Convert the optimal learning rate to scientific notation using format
x = "{:e}".format(opt_lr)

print(f"Optimal Learning Rate was --> {x}.")
fig2=plt.figure(figsize=(28, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss (LSTM)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
st.pyplot(fig2)

# Plot training and validation accuracy
fig3=plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy (LSTM)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
st.pyplot(fig3)
plt.tight_layout()
plt.show()
predictions_lstm = Lmodel.predict(X_test_reshaped)
binary_predictions_lstm = (predictions_lstm > 0.5).astype(int) 


# Calculate ROC curve
# fpr_lstm, tpr_lstm, _ = roc_curve(y_test, predictions_lstm)
# roc_auc_lstm = auc(fpr_lstm, tpr_lstm)

# Plot ROC curve
# fig4=plt.figure(figsize=(8, 6))
# plt.plot(fpr_lstm, tpr_lstm, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_lstm:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve (LSTM)')
# plt.legend(loc='lower right')
# plt.show()
# st.pyplot(fig4)