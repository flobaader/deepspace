from keras.models import Sequential
from keras.layers import *
from lib.qlearning4k.games import Snake
from keras.optimizers import *
from lib.qlearning4k import Agent
from keras import backend as keras
keras.set_image_dim_ordering('th')

grid_size = 10
nb_frames = 4
nb_actions = 5
epochs = 5000

model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(nb_frames, grid_size, grid_size)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(nb_actions))
model.compile(RMSprop(), 'MSE')
snake = Snake(grid_size)

agent = Agent(model=model, memory_size=-1, nb_frames=nb_frames)
agent.train(snake, batch_size=64, nb_epoch=epochs, gamma=0.8)

model.save('nets/snake' + str(epochs) + '.h5')
