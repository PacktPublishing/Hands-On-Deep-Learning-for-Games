#contextual bandit problem in Python
import random

alpha = .1
bandits = [[['bronze' , 1],['gold', 3], ['silver' , 2], ['bronze' , 1]],
           [['bronze' , 1],['gold', 3], ['silver' , 2], ['bronze' , 1]],
           [['bronze' , 1],['gold', 3], ['silver' , 2], ['bronze' , 1]],
           [['bronze' , 1],['gold', 3], ['silver' , 2], ['bronze' , 1]]]
q = [[0,0,0,0],
     [0,0,0,0],
     [0,0,0,0],
     [0,0,0,0]]

for i in range(10):    
    for b in range(len(bandits)):
        arm = random.randint(0,3)
        print('pulling arm {0} on bandit {1}'.format(arm,b))
        q[b][arm] = q[b][arm] + alpha * (bandits[b][arm][1]-q[b][arm])

print(q)
