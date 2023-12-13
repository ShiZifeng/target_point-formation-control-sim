import numpy as np
import matplotlib.pyplot as plt
import math
# number of agents
n = 4
# total time
T = 3000
# dt
dt = 0.01
# distance
distance = np.zeros((T, n+1, n+1))
# velocity
velocity = np.zeros((T, n+1, 2))
# position
position = np.zeros((T, n+1, 2))
# target_distance
target_distance = np.zeros((n+1, n+1))
# clockwise or counter clockwise 1:clockwise -1:counter clockwise
clockwise = np.zeros(n+1)

def init():
    for i in range(0,n):
        for j in range(0,n):
            target_distance[i,j] = -1
    for i in range(1,n):
        target_distance[i,i] = 0
    # velocity init
    get_velocity()
    # target init
    get_target()
    # position init
    get_position()
    # clockwise init
    clockwise_init()
    
##################################need replacement to simulation environment###########
def get_velocity():
    for i in range(1,n+1):
        velocity[0,i,0] = 0
        velocity[0,i,1] = 0

def get_position():
    position[0,1,0] = 0
    position[0,1,1] = 0
    position[0,2,0] = 6
    position[0,2,1] = 0
    position[0,3,0] = 3
    position[0,3,1] = 4
    position[0,4,0] = 0
    position[0,4,1] = 4

def get_target():
    target_distance[2,1] = 3
    target_distance[3,1] = 4
    target_distance[3,2] = 5
    target_distance[4,1] = 5
    target_distance[4,2] = 4

def clockwise_init():
    clockwise[1] = 0
    clockwise[2] = 0
    clockwise[3] = -1
    clockwise[4] = 1
##################################

# set the velocity of leader
def get_leader_velocity():
    return np.random.uniform(low=0.0, high=5.0, size=None), np.random.uniform(low=0.0, high=1.0, size=None)

def get_first_follower_v(i):
    v = -(np.square(distance[i,2,1]) -np.square(target_distance[2,1])) * (position[i,2,:] - position[i,1,:]) 
    return v
# distance
def dis(p1, p2):
    distance = math.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))
    return distance

def transforming(dx1, dx2, x1, x2):
    if dx1 > dx2:
        y1 = x1
        y2 = x2
        d1 = dx1
        d2 = dx2
    else :
        y1 = x2
        y2 = x1
        d1 = dx2
        d2 = dx1
    return y1, y2, d1, d2

def cal_s(y1, y2, d1, d2):
    distance = dis(y1, y2)
    if distance > d1 - d2 and distance < d1 + d2:
        y = 1 + ((distance - (d1 + d2)) * (distance - (d1 -d2))) / (2 * d1 * distance)
    else: y = 1
    return y

def draw_gif(i):
        if (i % 1 == 0 and i < 20) or (i % 5 == 0 and i > 20 and i < 100) or (i % 10 == 0 and i >100 or i < 200) or (i % 50 ==0 and i > 200):
            plt.clf()
            plt.plot(position[i,2,0], position[i,2,1], 'c.', 'markersize', 30)
            plt.plot(position[i,3,0], position[i,3,1], 'r.', 'markersize', 30)
            plt.plot(position[i,4,0], position[i,4,1], 'g.', 'markersize', 30)
            plt.plot(position[i,1,0], position[i,1,1], 'b.', 'markersize', 30)
            plt.plot([position[i,3,0],position[i,2,0]],[position[i,3,1],position[i,2,1]],'b-','linewidth',1.5)
            plt.plot([position[i,3,0],position[i,1,0]],[position[i,3,1],position[i,1,1]],'b-','linewidth',1.5)
            plt.plot([position[i,4,0],position[i,2,0]],[position[i,4,1],position[i,2,1]],'b-','linewidth',1.5)
            plt.plot([position[i,4,0],position[i,1,0]],[position[i,4,1],position[i,1,1]],'b-','linewidth',1.5)
            plt.plot([position[i,1,0],position[i,2,0]],[position[i,1,1],position[i,2,1]],'b-','linewidth',1.5)
            plt.text(position[i,1,0]-1.5,position[i,1,1],'agent1')
            plt.text(position[i,2,0]+0.5,position[i,2,1],'agent2')
            plt.text(position[i,3,0]+0.5,position[i,3,1],'agent3')
            plt.text(position[i,4,0]+0.5,position[i,4,1],'agent4')
            plt.xlim(-2,11)
            plt.ylim(-5,6)
            plt.draw()
            plt.pause(0.001)

def update():
    for i in range(0,T-1):
        t = dt * i
        # update distance
        for k in range(1,n+1):
            for j in range(1, n+1):
                distance[i,k,j] = dis(position[i,k,:], position[i,j,:])
        # update velocity
        velocity[i,1,:] = get_leader_velocity()
        velocity[i,2,:] = get_first_follower_v(i)
        for k in range(3,n+1):
            y1,y2,d1,d2 = transforming(target_distance[k,2], target_distance[k,1], position[i,2,:], position[i,1,:])
            s = cal_s(y1, y2, d1, d2)
            Rs = np.array([[s, clockwise[k] * math.sqrt(1 - s*s)],[-clockwise[k] * math.sqrt(1 - s*s), s]])

            velocity[i,k,:] = -distance[i,2,1] ** 2 * (position[i,k,:] - y1).T - np.dot(d1 * distance[i,2,1] * Rs, np.array((y1 - y2).T))
        #update position
        for k in range(1,n+1):
            position[i+1,k,0] = position[i,k,0] + velocity[i,k,0] * dt
            position[i+1,k,1] = position[i,k,1] + velocity[i,k,1] * dt
        # print(t)
        # print(position[i,3,:])
        draw_gif(i)
        
def main():
    init()
    update()


if __name__ == '__main__':
    main()