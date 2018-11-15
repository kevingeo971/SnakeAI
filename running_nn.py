import curses
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
#from statistics import median, mean
from collections import Counter

LR=1e-3
def initial_population():
    #game parameters------------------

    total_games = 10000      
    max_moves = 500
    frames = 1
    sh, sw = 15,15
    threshold_score = 7

    #---------------------------------

    score_list = []
    training_data = []

    for i in range(0,total_games):

        score = 0
        moves = 0
    
        moves_and_actions_list = []   # [[[sy,sx,fy,fx],[L,R,U,D]] , [[sy,sx,fy,fx],[L,R,U,D]] , [[sy,sx,fy,fx],[L,R,U,D]] ...... ]
        previous_observation = []     # [sy,sx,fy,fx]
        temp = []

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
        snake = [
            [snk_y, snk_x],
            [snk_y, snk_x-1],
            [snk_y, snk_x-2]
        ]

        food = [sh/2, sw/2]
        w.addch(food[0], food[1],curses.ACS_PI)

        key=0
        while moves<=max_moves:
            
            previous_observation = [snake[0][0],snake[0][1],food[0],food[1]]
            #next_key = w.getch()
            next_key = random.choice([261,260,258,259])
            action=[]
            #temp = [previous_observation,next_key]
            #moves_and_actions_list.append(temp)

            key = key if next_key == -1 else next_key

            new_head = [snake[0][0], snake[0][1]]
            moves += 1
            if key == curses.KEY_DOWN:
                action = [0,0,0,1]
                new_head[0] += 1
                if (new_head[0]==sh-1): 
                    new_head[0]=1 
            if key == curses.KEY_UP:
                action=[0,0,1,0]
                new_head[0] -= 1
                if (new_head[0]==0): 
                    new_head[0]=sh-2
            if key == curses.KEY_LEFT:
                action=[1,0,0,0]
                new_head[1] -= 1
                if (new_head[1]==0): 
                    new_head[1]=sw-2
            if key == curses.KEY_RIGHT:
                action=[0,1,0,0]
                new_head[1] += 1
                if (new_head[1]==sw-1): 
                    new_head[1]=1

            temp = [previous_observation,action]
            moves_and_actions_list.append(temp)
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
        curses.endwin()
        score_list.append(score)

        if score >= threshold_score:
            for i in moves_and_actions_list:
                training_data.append(i)

    score_list=sorted(score_list)
    #print(score_list[total_games-5:total_games-1])
    #print(len(training_data))
    #print(training_data[0])
    return training_data


def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 4, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input': X}, {'targets': y}, n_epoch=10, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model

training_data=initial_population()
print("done populationg")
print(len(training_data))
model = train_model(training_data)

'''print(len(training_data))
print(training_data[0])
print(training_data[0][0])
print(training_data[0][1])'''