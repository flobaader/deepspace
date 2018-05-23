import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [0], [0], [1]])

model = Sequential()
model.add(Dense(1, activation='sigmoid', input_dim=2))

sgd = SGD(lr=0.1)

model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X, y, batch_size=1, epochs=300)

print(model.predict_proba(X))

#kerasimo.ToSVG('and', model, X)









