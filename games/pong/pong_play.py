from keras import backend
from keras.layers import Flatten, Dense, Conv2D
from keras.models import Sequential, load_model
from keras.optimizers import *

from lib.qlearning4k import Agent
from lib.qlearning4k.games import Pong

width = 30
height = 20
grid_size = 10
hidden_size = 256
nb_frames = 4
nb_actions = 3

train = True
model_file = 'pong900.h5'

backend.set_image_dim_ordering('th')

if train:
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(nb_frames, height, width)))
    model.add(Flatten())
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(nb_actions))
    # model.compile(sgd(lr=.2), "mse")
    model.compile(RMSprop(), 'MSE')
    pong = Pong(width, height, live_view=True, maxScore=1)
    agent = Agent(model=model, memory_size=-1, nb_frames=nb_frames)

    for i in range(1, 10):
        agent.train(pong, batch_size=64, nb_epoch=100, epsilon=.1, gamma=.8)
        model.save('pong' + str(i * 100) + '.h5')
else:
    model = load_model(model_file)
    pong = Pong(width, height, live_view=False, maxScore=10, print_score=True)
    agent = Agent(model=model, memory_size=-1, nb_frames=nb_frames)
    agent.play(pong, nb_epoch=50)


