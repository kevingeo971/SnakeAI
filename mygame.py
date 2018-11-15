import random
import curses

''' ---- SNAKE RL ---- '''

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
    
    moves_and_actions_list = []   # [[[sy,sx,fy,fx],[action]] , [[sy,sx,fy,fx],[action]] , [[sy,sx,fy,fx],[action]] ...... ]
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
        
        temp = [previous_observation,next_key]
        moves_and_actions_list.append(temp)

        key = key if next_key == -1 else next_key

        new_head = [snake[0][0], snake[0][1]]
        moves += 1
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
    curses.endwin()
    score_list.append(score)

    if score >= threshold_score:
        for i in moves_and_actions_list:
            training_data.append(i)

score_list=sorted(score_list)
print(score_list[total_games-5:total_games-1])
print(len(training_data))
print(training_data[0])

