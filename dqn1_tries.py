import numpy as np
import time
import random

from tqdm import tqdm#simply to visualize the progress bars of training
import cv2
import os
from PIL import Image
from collections import deque

import matplotlib.pyplot as plt

import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam

#----------------------------------------------------------------WHEN LOADING A Q TABLE MODEL----------------------------------------------------------------:
'''
    - EPSILON (e) MUST BE DOWNGRADED TO 0 SO IT ONLY EXPLOITS FROM THAT Q-TABLE, if not, = 1
    - MIN_REWARD MUST BE DOWNGRADED TO 0
    - SHOW_EACH must be set to True to visualize your model

'''


#tf.config.list_physical_devices('GPU')
#config.gpu_options.per_process_gpu_memory_fraction
#tf.config.gpu_options.allow_growth = True
#C:\Program Files\NVIDIA Corporation\NvStreamSrv


#This must be the filepath for the model, eg: "models/..... .model" or None for model creation
#MODEL_LOADED = "models/2x256___300.00max__246.90avg__105.00min__1595101534.model"
MODEL_LOADED = None


TIMES_HIT_FOOD = 0

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000  #last steps to keep for training the model
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training

#Basically from Nnets theory, the weights that the model creates adjust to the batch size inputs. So if the batch size is 1, thr weights will adjust only to that 1 thing,
#and when another batch size is inputted, the weights will all change to fit that other 1 thing.
#(So basically we have to have a big enough, but not extremly large, batch size)
#Deals with overfitting

MINI_BATCH_SIZE = 64  # How many steps (samples) to use for training

UPDATE_TARGET_EVERY = 5  # This is used to see each how many steps to update the target network
MODEL_NAME = '2x256'

MIN_REWARD = 0 #SAVED model
#MIN_REWARD = -200  #TO TRAIN model

MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 10
eps = list(range(1, EPISODES + 1)) #[1, 10]

# Lists to keep track of stats
epsilon_list = []
average_reward_list = []
min_reward_list = []
max_reward_list = []


# Exploration settings
e = 1  #epsilon, will be decayed at the end
e_DECAY = 0.99975
MIN_e = 0.0005#threshold of epsilon

#  Stats settings
GATHER_STATS_EACH = 50# each x episodes stats will be gathered (the lists)
SHOW_EACH = True#True if you want to render, False if not

#EXPLORE/EXPLOIT
def e_greedy(state, e):
    if (np.random.random() > e):
        action = np.argmax(agent.get_qs(current_state))
        #print("-EXPLOIT")
    else:
        action = np.random.randint(0, env.ACTION_SPACE_SIZE)
        #print("                                            EXPLORE")
    return action


def data_gathering(e, ep_rewards, GATHER_STATS_EACH, MIN_REWARD):
    
    average_reward = sum(ep_rewards[-GATHER_STATS_EACH:]) / len(ep_rewards[-GATHER_STATS_EACH:])
    average_reward_list.append(average_reward)

    min_reward = min(ep_rewards[-GATHER_STATS_EACH:])
    min_reward_list.append(min_reward)

    max_reward = max(ep_rewards[-GATHER_STATS_EACH:])
    max_reward_list.append(max_reward)

    if average_reward >= max(average_reward_list):
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')


def data_visualization(eps, epsilon_list, average_reward_list, min_reward_list, max_reward_list):
    plt.plot(eps, epsilon_list, 'bs')
    plt.ylabel("Epsilon value")
    plt.xlabel("Episodes")
    plt.show()

    plt.plot(eps, average_reward_list, 'r--')
    plt.ylabel("Average reward")
    plt.xlabel("Episodes")
    plt.show()


#BOXES for player, food and enemy creation
class Box:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)
    
    def __str__(self):
        return f"Box ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    #9 moving options [0, 8]
    def action(self, choice):

        #DIAGONAL MOVES
        if choice == 0:
            self.move(x = 1, y = 1)
        elif choice == 1:
            self.move(x = - 1, y = - 1)
        elif choice == 2:
            self.move(x = -1, y = 1)
        elif choice == 3:
            self.move(x = 1, y = - 1)

        #LEFT/RIGHT
        elif choice == 4:
            self.move(x = 1, y = 0)
        elif choice == 5:
            self.move(x =- 1, y = 0)

        #UP/DOWN
        elif choice == 6:
            self.move(x = 0, y = 1)
        elif choice == 7:
            self.move(x = 0, y =- 1)

        #STATIC
        elif choice == 8:
            self.move(x = 0, y = 0)


    def move(self, x = False, y = False):
        #If there's no value for x or y, choose a random option to move
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y


        #If it's going out of frame, stick to position x = 0 (instead of going outside the box)
        if self.x < 0:
            self.x = 0

        elif self.x > self.size-1:
            self.x = self.size-1
        #same as previous comment but for y coordinates
        if self.y < 0:
            self.y = 0

        elif self.y > self.size-1:
            self.y = self.size-1


class Box_environment:
    SIZE = 10

    #These values seem to work fine, they could be changed to see if more efficiency can be obtained.
    #(Previously had move penalty set to 1 and agent wasn't very good)
    PENALTY_MOVE = 15
    PENALTY_ENEMY = 300
    REWARD_FOOD = 300

    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)
    ACTION_SPACE_SIZE = 9

    PLAYER_DICT = 1  # player key for dictionary below
    FOOD_DICT = 2  # food key for dictionary below
    ENEMY_DICT = 3  # enemy key for dictionary below

    # {key : (bgr colour)}
    env_components_colours = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)}

    def reset(self):
        self.player = Box(self.SIZE)
        self.food = Box(self.SIZE)

        while self.food == self.player:
            self.food = Box(self.SIZE)

        self.enemy = Box(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Box(self.SIZE)

        self.episode_step = 0

        observation = np.array(self.get_image())

        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)
        TIMES_HIT_FOOD = 0

        '''
        Enable this to make enemy or food move
        enemy.move()
        food.move()
        '''
        new_observation = np.array(self.get_image())

        if self.player == self.enemy:
            reward = -self.PENALTY_ENEMY
        elif self.player == self.food:
            TIMES_HIT_FOOD += 1
            print("Times hit the food: %d" %TIMES_HIT_FOOD)

            reward = self.REWARD_FOOD
        else:
            reward = -self.PENALTY_MOVE

        done = False
        #done if it hits the FOOD, ENEMY, or hasn't reached any of those in 200 steps
        if reward == self.REWARD_FOOD or reward == -self.PENALTY_ENEMY or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    #SHOW THE ENVIRONMENT
    def render(self):
        img = self.get_image()
        img = img.resize((600, 600))  #600 x 600 the image of the env.
        cv2.imshow("Environment", np.array(img))  #show the environment
        cv2.waitKey(1)

    # FOR THE NNet
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size

        #These 3 lines will set the colours to the respective boxes in the environment
        env[self.food.x][self.food.y] = self.env_components_colours[self.FOOD_DICT]
        env[self.enemy.x][self.enemy.y] = self.env_components_colours[self.ENEMY_DICT]
        env[self.player.x][self.player.y] = self.env_components_colours[self.PLAYER_DICT]

        img = Image.fromarray(env, 'RGB')  #read from the environment into an image
        return img


env = Box_environment()

# For stats
ep_rewards = [-200]

#store models in models folder
if not os.path.isdir('models'):
    os.makedirs('models')


#Agent class: Basically here we have 2 models, the main one and the target one. We are doing .fit in the main model and .predict in the target model.
#Main model is the one getting trained. We do this system to handle the chaos that the random actions do
class DQNAgent:
    def __init__(self):

        # Main
        self.model = self.create_model()

        # Target
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        #Array with the last n steps left for training
        #This is a double-ended queue with length REPLAY_MEMORY_SIZE. Transitions will be appended in this array to keep track of the last REPLAY_MEMORY_SIZE trans.
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        #This will increment each episode, when its > UPDATE_TARGET_EVERY, we will replace target with the current main (and set counter to 0)
        self.target_update_counter = 0 

    #cnn using keras
    #Dropout is done to overcome overfitting (layer by layer as we can see above) --> Each pass of that layer an amount of neurons (the number spcified when calling the 
    #function) is dropped (not taken into acount) to avoid overfitting. In this case 30% of them (0.3).
    def create_model(self):
        #LOAD MODEL
        if MODEL_LOADED != None:
            model = tf.keras.models.load_model(MODEL_LOADED)
            print("MODEL LOADED:%s " %MODEL_LOADED)
        #CREATE MODEL
        else:
            model = Sequential()
            model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size = (2, 2)))
            model.add(Dropout(0.3))

            model.add(Conv2D(256, (3, 3)))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size = (2, 2)))
            model.add(Dropout(0.3))

            model.add(Flatten())
            model.add(Dense(64))

            model.add(Dense(env.ACTION_SPACE_SIZE, activation = "linear"))
            model.compile(loss = "mse", optimizer = Adam(lr = 0.001), metrics = ['accuracy'])
        return model
        '''

        model = Sequential()
        #model.add(Input(shape = (env.OBSERVATION_SPACE_VALUES)))# OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Conv2D(4, 2, 1, activation = "sigmoid", input_shape = env.OBSERVATION_SPACE_VALUES))
        model.add(Conv2D(8, 2, 1, activation = "sigmoid", input_shape = env.OBSERVATION_SPACE_VALUES))
        model.add(Conv2D(8, 1, 1, activation = "sigmoid", input_shape = env.OBSERVATION_SPACE_VALUES))

        model.add(Flatten())
        model.add(Dense(64, activation = "relu"))
        model.add(Dense(64))
        model.compile(loss = "mse", optimizer = Adam(lr = 0.001), metrics = ['accuracy'])
        '''
        #return model


    #variables returned by step() are appended in replay_memory
    #These are appended to replay memory which is a dequeu
    def update_replay_memory(self, trans):
        self.replay_memory.append(trans)#(observation space, action, reward, new observation space, done)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        #Training starts if there are several transs already appended
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        #from replay_memory table, get MINI_BATCH_SIZE size of random samples (64)
        mini_batch = random.sample(self.replay_memory, MINI_BATCH_SIZE)

        # Get current states from mini_batch, then query NN model for Q values
        #trans[0] = Observation space
        current_states = np.array([trans[0] for trans in mini_batch])/255
        current_qs_list = self.model.predict(current_states)

        
        #We get trans[3] to .predict the future qs list
        #trans[3] = new observation space
        new_current_states = np.array([trans[3] for trans in mini_batch])/255
        new_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(mini_batch):

            #if its not terminal state, get the new max q
            if not done:
                max_future_q = np.max(new_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q

            #if its terminal state, max_future_q is 0 and the new_q is just reward
            else:
                new_q = reward

            #update
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            #append for training afterwards
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as 1 batch
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINI_BATCH_SIZE, verbose=0, shuffle=False)

        #+1 target network counter when its terminal state
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    #.predict on main net given the state
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]#unpacks state, divs by 255 because RGB


agent = DQNAgent()

#MAIN LOOP for iterations
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    episode_reward = 0
    step = 1
    current_state = env.reset()
    done = False

    while not done:

#CHOOSE ACTION
#------------------------------------------------------------------------------------------------------------------------------
        action = e_greedy(current_state, e)
#------------------------------------------------------------------------------------------------------------------------------

#TAKE STEP
#------------------------------------------------------------------------------------------------------------------------------
        new_state, reward, done = env.step(action)
#------------------------------------------------------------------------------------------------------------------------------

#ADD THE REWARD TO THIS EPISODE'S TOTAL REWARD + render or not
#------------------------------------------------------------------------------------------------------------------------------
        episode_reward = episode_reward + reward

        if SHOW_EACH and not episode % 1:
            env.render()
#------------------------------------------------------------------------------------------------------------------------------

#CNN
#------------------------------------------------------------------------------------------------------------------------------
        #each step we update the replay memory + train main
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)
#------------------------------------------------------------------------------------------------------------------------------
        current_state = new_state
        step += 1

    #append episode_reward + gather data
    ep_rewards.append(episode_reward)

    data_gathering(e, ep_rewards, GATHER_STATS_EACH, MIN_REWARD)

    # Decay epsilon for convergence
    #Only works in training, when loading a model it won't store anything on the list, as epsilon = 0
    if e > MIN_e:
        e = e * e_DECAY
        e = max(MIN_e, e)
        epsilon_list.append(e)
print(epsilon_list)
data_visualization(eps, epsilon_list, average_reward_list, min_reward_list, max_reward_list)