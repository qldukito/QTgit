# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
from keras import *
from keras.callbacks import *
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from keras.layers import *
from sklearn.pipeline import Pipeline
import tensorflow as tf

PATH_TO_DATA = "E:/Code/QT/BTC/tsfc/datasets/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv"
data = pd.read_csv(PATH_TO_DATA)
data_last1year = data.iloc[-555600:-1,:].fillna(method='pad')
data_last2year = data.iloc[-(555600*2):-1,:].fillna(method='pad')


X = data_last1year.iloc[:-1,1:]
y = data_last1year.iloc[1:,4]

X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.2, train_size=0.8, shuffle=False, random_state=7)

estimators=[]
estimators.append(['robust',RobustScaler()])
estimators.append(['mixmax',MinMaxScaler()])

scale=Pipeline(estimators,verbose=True)

X_train=scale.fit_transform(X_train)
X_test=scale.transform(X_test)

X_train=np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
X_test=np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))

y_train=y_train.values
y_train=np.reshape(y_train,(y_train.shape[0],1,1))

y_test=y_test.values
y_test=np.reshape(y_test,(y_test.shape[0],1,1))

# **Model Architecture
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

adam= tf.keras.optimizers.Adam(lr=lr_schedule(0),amsgrad=True)
model = Sequential()
model.add(Bidirectional(LSTM(350, return_sequences=True, activation='relu'),input_shape=(1, X_train.shape[2])))
model.add(Bidirectional(LSTM(350, return_sequences=True, activation='relu')))
model.add(Dense(1))
model.compile(loss="logcosh", optimizer=adam, metrics=['mae'])

mcp_save = ModelCheckpoint('E:/Code/QT/BTC/tsfc/results/LSTM_reg_new.hdf5', save_best_only=True, monitor='val_loss', mode='auto')
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')

# **Training
model.fit(X_train,y_train, epochs=1000, batch_size=128, validation_data=(X_test,y_test), callbacks=[mcp_save,earlyStopping])

# **Testing Model**
from tensorflow.keras.models import load_model
prediction_model = load_model('E:/Code/QT/BTC/tsfc/results/LSTM_reg_new.hdf5',compile=False)

y_pred = np.ravel(prediction_model.predict(X_test))
y_pred_perc = y_pred[1:]

y_test=np.ravel(y_test)
y_test_perc = y_test[:-1]

pred_perc = ((y_pred_perc-y_test_perc)/y_test_perc)*100
real_perc = ((y_test[1:]-y_test_perc)/y_test_perc)*100

r2=r2_score(y_test,y_pred) #testing score/ r^2
mae=mean_absolute_error(y_test,y_pred) #mae
rmse=np.sqrt(mean_squared_error(y_test,y_pred)) #rmse
mape=mean_absolute_percentage_error(y_test,y_pred) #mape

result = pd.DataFrame({'y_pred':y_pred,'y_test':y_test})
result.to_csv('E:/Code/QT/BTC/tsfc/results/result.csv')
result_visual = result.iloc[-300:,:] # 将最近一分钟的预测值与真实值可视化


result_perc = pd.DataFrame({'pred_perc':pred_perc,'real_perc':real_perc})
result_perc.to_csv('E:/Code/QT/BTC/tsfc/results/result_perc.csv')
result_visual_perc = result_perc.iloc[-300:,:] # 将最近一分钟百分比可视化

error = pd.DataFrame(zip(['MAE','RMSE','MAPE','R^2'],[mae,rmse,mape,r2])).transpose()
error.to_csv('E:/Code/QT/BTC/tsfc/results/error.csv')

print(error)
result_visual.plot(linewidth=2)
plt.savefig('E:/Code/QT/BTC/tsfc/results/result.png')
plt.show()

result_visual_perc.plot(linewidth=2)
plt.savefig('E:/Code/QT/BTC/tsfc/results/result_perc.png')
plt.show()