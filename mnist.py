# -*- coding: utf-8 -*-

# 3. Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility
import time
#import keras.callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras import layers
#from keras.utils import visualize_util


 
# 4. Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# 5. Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
         
# 6. Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
 
# 7. Define model architecture
model = Sequential() 
 
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28), dim_ordering='th'))
# 32*3*3*32
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
# 32*256
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(1, activation='linear'))
# 256*10
model.add(Dense(10, activation='softmax'))


 
# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#visualize_util.plot(model, to_file='model.png')

#earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')

t1 = time.time()

#model.fit(X_train, Y_train, batch_size=32, nb_epoch=100, verbose=2, callbacks=[earlyStopping], validation_data=(X_test, Y_test), shuffle=True)





# 9. Fit model on training data


#model.fit(X_train, Y_train, 
#          batch_size=32, nb_epoch=10, verbose=2)
print ("time")
t2 = time.time()
print t2-t1
# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=1)

print score