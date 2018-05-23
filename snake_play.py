from keras.models import Sequential
from keras.layers import *
import matplotlib
from lib.qlearning4k.games import Snake
from keras.optimizers import *
from lib.qlearning4k import Agent
from keras.models import load_model



from keras import backend as K
K.set_image_dim_ordering('th')

grid_size = 10
nb_frames = 4
nb_actions = 5

snake = Snake(grid_size)
model = load_model('snake5000.h5')

agent = Agent(model=model, memory_size=4, nb_frames=nb_frames)

agent.play(snake, nb_epoch=10, visualize=True)
