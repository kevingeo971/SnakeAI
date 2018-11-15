import random
import curses

''' Snake RL '''

#game parameters------------------
total_games = 20000         
max_moves = 500
frames = 1
sh, sw = 8,8
threshold_score = 0
#---------------------------------

score_list = []

for i in range(0,total_games):

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
    snake = [
        [snk_y, snk_x],
        [snk_y, snk_x-1],
        [snk_y, snk_x-2]
    ]

    food = [sh/2, sw/2]
    w.addch(food[0], food[1],curses.ACS_PI)

    key=0
    while moves<=max_moves:
    
        next_key = w.getch()
        next_key = random.choice([261,260,258,259])

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

score_list=sorted(score_list)
print(score_list[19989:19999])

