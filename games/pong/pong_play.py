from keras.models import Sequential
from keras.layers import Flatten, Dense
from lib.qlearning4k.games import Pong
from keras.optimizers import *
from lib.qlearning4k import Agent

width = 30
height = 20
grid_size = 10
hidden_size = 100
nb_frames = 4

model = Sequential()
model.add(Flatten(input_shape=(nb_frames, height, width)))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(3))
model.compile(sgd(lr=.2), "mse")

pong = Pong(width, height)
agent = Agent(model=model, memory_size=-1, nb_frames=nb_frames)
agent.train(pong, batch_size=10, nb_epoch=10, epsilon=.1)
agent.play(pong)


