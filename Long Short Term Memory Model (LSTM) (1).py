#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import MinMaxScaler
get_ipython().system('pip install keras')
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout


# In[101]:


google_df = pd.read_csv('Google Data.csv')


# In[102]:


google_df.head()


# In[103]:


google_df.info()


# In[104]:


google_df['Date'] = pd.to_datetime(google_df['Date'])
google_df['Date'] = google_df['Date'].astype('int64') // 10**9                                     


# In[105]:


#Rescale our values for better result/performances
sc = MinMaxScaler(feature_range=(0,1))
google_df = sc.fit_transform(google_df)
google_df.shape


# In[106]:


X_train = []
y_train = []
for i in range (60,1149): #60 : timestep//1149 : Length of the data
    X_train.append(google_df[i-60:i,0])
    y_train.append(google_df[i,0])
    
X_train,y_train = np.array(X_train),np.array(y_train)


# In[107]:


#Adding the batch_size axis
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
X_train.shape


# In[108]:


#Building Long Short Term Memory Model (LSTM)
model = Sequential()

model.add(LSTM(units=100, return_sequences = True, input_shape = (X_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = False))
model.add(Dropout(0.2))

model.add(Dense(units=1))
model.compile(optimizer='adam',loss="mean_squared_error")


# In[109]:


hist = model.fit(X_train, y_train, epochs = 20, batch_size = 32, verbose=2)


# In[110]:


plt.plot(hist.history['loss'])
plt.title('Training Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()


# In[148]:


sc = MinMaxScaler(feature_range=(0, 1))
sc.fit(inputClosing)


# In[149]:


sc = MinMaxScaler(feature_range=(0,1))
google_df = sc.fit_transform(google_df)
google_df.shape


# In[150]:


X_test = []
length = len(google_df)
timestep = 60 
for i in range(timestep, length):
    X_test.append(inputClosing_scaled[i-timestep:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_test.shape


# In[151]:


y_pred = model.predict(X_test)


# In[152]:


y_pred


# In[153]:


predicted_price = sc.inverse_transform(y_pred)


# In[154]:


plt.plot(y_test, color = 'red', label = 'Actual Stock Price')
plt.plot(predicted_price, color = 'green', label = 'Predicted Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock  Price')
plt.legend()
plt.show()


# In[ ]:




