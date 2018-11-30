import random
import curses

import numpy as np
from sklearn import preprocessing
import pandas as pd
#import matplotlib.pyplot as plt
import re
from keras.models import Sequential,Model
from keras.layers import Dense, Input, Dot
from keras.models import load_model, model_from_json
from keras.optimizers import Adam

from collections import deque

import time

''' ------------- SNAKE RL -------------- '''

def create_model(n_states, n_actions):
    # Maximum future discounted reward
    # Q(S_t)
    state = Input(shape=(n_states,))
    x1 = Dense(4, activation='relu')(state)
    x2 = Dense(4, activation='relu')(x1)
    out = Dense(n_actions)(x2)
    # Q(S_t)(a_t)
    actions = Input(shape=(n_actions,))
    out2 = Dot(axes=-1)([out, actions])
    
    # wrap the above in Keras Model class
    model = Model(inputs=[state, actions], outputs=out2)
    model.compile(loss='mse', optimizer='rmsprop')
    
    model2 = Model(inputs=state, outputs=out)

    return model, model2

def train_data(minibatch, model):
    s_j_batch = np.array([d[0] for d in minibatch])
    a_batch = np.array([d[1] for d in minibatch])
    r_batch = np.array([d[2] for d in minibatch])
    s_j1_batch = np.array([d[3] for d in minibatch])
    terminal_batch = np.array([d[4] for d in minibatch])

    readout_j1_batch = model.predict(s_j1_batch, batch_size=BATCH)
    y_batch = r_batch + GAMMA * np.max(readout_j1_batch, axis=1)
    y_batch[terminal_batch] = r_batch[terminal_batch]
    return s_j_batch, a_batch, y_batch

#--------------------game parameters----------------------

total_games = 200
max_moves = 500
frames = 100
sh, sw = 15,15
threshold_score = 4

STATES, ACTIONS = 4,4
model, out = create_model(STATES, ACTIONS)
INITIAL_EPSILON = 1e-1
FINAL_EPSILON = 1e-4
DECAY = 0.9
GAMMA = 0.9 # decay rate of past observations
OBSERVE = 5000. # timesteps to observe before training
REPLAY_MEMORY = 5000 # number of previous transitions to remember
TIME_LIMIT = 100000
BATCH = 128

#-------------------------------------

score = 0
moves = 0

s = curses.initscr()
curses.curs_set(0)
#sh, sw = s.getmaxyx()
#print(sh,sw)
w = curses.newwin(sh, sw, 0, 0)
w.keypad(1)
w.timeout(frames)

#w.getch()
snk_x = sw/4
snk_y = sh/2

def reset():
    global w
    global snake
    global food
    global score 
    score = 0
    w = curses.newwin(sh, sw, 0, 0)
    w.keypad(1)
    w.timeout(frames)
    snk_x = sw/4
    snk_y = sh/2
    snake = [
        [snk_y, snk_x],
        [snk_y, snk_x-1],
        [snk_y, snk_x-2]
    ]

    food = [sh/2, sw/2]
    w.addch(food[0], food[1],curses.ACS_PI) 
    w.addch(snake[0][0], snake[0][1], curses.ACS_CKBOARD)
    return np.array([snake[0][0],snake[0][1],food[0],food[1]])

def step(key):
    global snake
    global food
    global w
    global score
    new_head = [snake[0][0], snake[0][1]]
    #moves += 1
    if key == curses.KEY_DOWN:
        
        new_head[0] += 1
        if (new_head[0]==sh-1): 
            new_head[0]=1 
    if key == curses.KEY_UP:
        
        new_head[0] -= 1
        if (new_head[0]==0): 
            new_head[0]=sh-2
    if key == curses.KEY_LEFT:
        
        new_head[1] -= 1
        if (new_head[1]==0): 
            new_head[1]=sw-2
    if key == curses.KEY_RIGHT:
        
        new_head[1] += 1
        if (new_head[1]==sw-1): 
            new_head[1]=1
    
    snake.insert(0, new_head)

    if snake[0] == food:
        score += 1
        #print(score)
        food = None
        while food is None:
            nf = [
                random.randint(2, sh-2),
                random.randint(2, sw-2)
            ]
            food = nf if nf not in snake else None
        w.addch(food[0], food[1], curses.ACS_PI)
    else:
        tail = snake.pop()
        w.addch(tail[0], tail[1], ' ')

    w.addch(snake[0][0], snake[0][1], curses.ACS_CKBOARD)

snake = [
    [snk_y, snk_x],
    [snk_y, snk_x-1],
    [snk_y, snk_x-2]
]

food = [sh/2, sw/2]
w.addch(food[0], food[1],curses.ACS_PI)


initial_state = reset()

key = 0

moves = 0

while 1:
    
    moves = moves + 1

    if moves>50: 
        #curses.endwin()
        #quit()
        reset()  
        moves = 0  

    next_key = w.getch()

    key = key if next_key == -1 else next_key

    step(key)

    
    
curses.endwin()
print(initial_state)


