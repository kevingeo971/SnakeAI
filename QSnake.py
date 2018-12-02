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
import math

''' ------------- SNAKE RL -------------- '''

def create_model(n_states, n_actions):
    # Maximum future discounted reward
    # Q(S_t)
    state = Input(shape=(n_states,))
    x1 = Dense(8, activation='relu')(state)
    x2 = Dense(12, activation='relu')(x1)
    x3 = Dense(8, activation='relu')(x2)
    out = Dense(n_actions)(x3)
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
frames = 5
sh, sw = 15,15
threshold_score = 20

STATES, ACTIONS = 4,4
model, out = create_model(STATES, ACTIONS)
INITIAL_EPSILON = 1e-1
FINAL_EPSILON = 1e-3
DECAY = 0.9
GAMMA = 0.9 # decay rate of past observations
OBSERVE = 5000 # timesteps to observe before training
REPLAY_MEMORY = 5000 # number of previous transitions to remember
TIME_LIMIT = 100000
BATCH = 2048

#-------------------------------------

score = 0
moves = 0
games = 0
target_reached = 0

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
    global games
    games+=1
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

def step(action):
    
    global snake
    global food
    global w
    global score
    global s
    global moves
    global target_reached
    #moves += 1

    dead = False     

    new_head = [snake[0][0], snake[0][1]]

    key_val=[261,260,258,259]

    key = w.getch()
    key = key_val[action]

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

    if snake[0][0] in [1, sh-2] or snake[0][1]  in [1, sw-2]:
        dead = True  

    #------------  reward   -----------------------------------
   
    distance = float( math.sqrt( (snake[0][0] - food[0])**2 + (snake[0][1] - food[1])**2 ) )

    if snake[0] == food :
        food_reward = 20
    else:
        food_reward = 0

    reward = food_reward - 1*distance

    if dead == True:
        reward = (-40)

    #------------  reward   -----------------------------------

    if snake[0] == food:
        score += 1
        #print(score)
        food = None
        while food is None:
            nf = [
                random.randint(3, sh-3),
                random.randint(3, sw-3)
            ]
            food = nf if nf not in snake else None
        w.addch(food[0], food[1], curses.ACS_PI)
    else:
        tail = snake.pop()
        w.addch(tail[0], tail[1], ' ')

    w.addch(snake[0][0], snake[0][1], curses.ACS_CKBOARD)

    state = np.array( [ snake[0][0],snake[0][1],food[0],food[1] ] )

    terminal = False
    if score >= threshold_score : 
        terminal = True
        target_reached += 1

    s.addstr(16,0,"Moves = " + str(moves))
    s.addstr(17,0,"Score = " + str(score))
    s.addstr(18,0,"Games = " + str(games))
    #s.addstr(19,0,"Target Reached = " + str(target_reached))
    s.addstr(19,0,"Reward = " + str(reward))
    s.addstr(20,0,"")
    s.refresh()

    if dead:
        reset()

    return state, reward, terminal

snake = [
    [snk_y, snk_x],
    [snk_y, snk_x-1],
    [snk_y, snk_x-2]
]

food = [sh/2, sw/2]
w.addch(food[0], food[1],curses.ACS_PI)

reset()

D = deque(maxlen=REPLAY_MEMORY)
loss = []

actions = np.zeros(ACTIONS)
actions[np.random.choice(ACTIONS)] = 1
state, reward, terminal = step(np.argmax(actions))

epsilon = INITIAL_EPSILON

points_array = [0]
start = time.time()

key = 0

l=[]

for moves in range(0,TIME_LIMIT):
       
    readout_t = out.predict(state[None, :])

    actions = np.zeros([ACTIONS])
    if np.random.random() <= epsilon:
        actions[np.random.choice(ACTIONS)] = 1
    else:
        actions[np.argmax(readout_t)] = 1

    if epsilon > FINAL_EPSILON and moves > OBSERVE*3:
        epsilon *= DECAY
    
    next_state,reward,terminal= step(np.argmax(actions))
    D.append((state,actions,reward,next_state,terminal))

    if terminal:
        points_array.append(0)
        reset()
    else:
        points_array[-1] += score

    if moves > OBSERVE:
        # sample a minibatch to train on
        idx = np.random.choice(REPLAY_MEMORY, BATCH, replace=False)
        minibatch = [D[i] for i in idx]
        # get the batch variables
        s_t_batch, a_batch, y_batch = train_data(minibatch, out)
        # perform gradient step
        loss.append(model.train_on_batch([s_t_batch, a_batch], y_batch))

    state = next_state

    #if t%(TIME_LIMIT/20)==0:
        #print('Episode :', moves,'s, average up time: ',np.mean(points_array[-100:]))
        

    '''if moves>50: 
        #curses.endwin()
        #quit()
        reset()  
    
    #next_key = w.getch()
    next_key = random.choice([0,1,2,3])

    key = key if next_key == -1 else next_key

    state, reward, terminal = step(key)

    l.append([state,reward,terminal])
    '''

curses.endwin()

for i in D:
    print(i)

