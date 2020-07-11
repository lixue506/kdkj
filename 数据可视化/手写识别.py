import sklearn
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
train_data=pd.read_excel(r"C:\Users\1\Desktop\数据可视化\train.xlsx")
test_data=pd.read_excel(r"C:\Users\1\Desktop\数据可视化\test.xlsx")
x_train=train_data.iloc[:-1,1:]
y_train=train_data.iloc[:-1,:1]
x_test=test_data.iloc[:-1,:]
model=keras.models.Sequential()
model.add(keras.layers.Dense(30,input_shape=x_train.shape[1:]))
for _ in range(20):
    model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.AlphaDropout(rate=0.5))
model.add(keras.layers.Dense(10,activation='softmax'))
model.summary()
model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
history=model.fit(x_train,np.array(y_train),epochs=100)
y_test=model.predict(x_test)
for i in range(len(y_test)):
    y_test[i]=np.argmax(y_test[i])
y_predict=y_test[:,1]
def plt_learning_curve(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(2,3)
    plt.show()
plt_learning_curve(history)
