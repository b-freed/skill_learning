
import numpy as np
import matplotlib.pyplot as plt

class PointmassEnv():
    def __init__(self,m = 1.0,dt = 0.01):

        '''
        Pointmass environment
        State: self.x: (x, y, vx, vy) of point mass
        Control input u: (fx, fy)
        '''

        self.m = m
        self.dt = dt

    def reset(self,x0=np.zeros(4)):
        assert x0.shape[0] == 4
        self.x = x0

        return self.x


    def step(self,u):
        '''
        evolves the state of a point mass, initially in state self.x, according to control input u

         
        '''

        pos = self.x[:2]
        vel = self.x[2:]

        # compute acceleration
        a = u/self.m 

        # compute change in velocity
        delta_v = a*self.dt

        # compute new velocity
        vel_new = vel + delta_v

        # compute new pos
        delta_pos = vel * self.dt

        pos_new = pos+delta_pos

        self.x = np.concatenate([pos_new,vel_new])

        return self.x

    def get_stabilizing_control(self,loc_sp,kp=5,kd=2):
        '''
        Stabilize to a setpoint sp
        loc_sp: [goal_x,goal_y]
        '''

        loc_error = loc_sp - self.x[:2]
        vel = self.x[2:]

        return kp*loc_error - kd*vel
    
if __name__ == '__main__':

    env = PointmassEnv()

    
    start_loc = 2*np.random.uniform(size=2) - 1
    start_state = np.concatenate([start_loc,np.zeros(2)]) 
    goal_loc = 2*np.random.uniform(size=2) - 1
    state = env.reset(start_state)
    states = [state]
    
    
    for t in range(100):
        print('env.x: ', env.x)
        u = env.get_stabilizing_control(goal_loc)
        print('u: ', u)
        state = env.step(u)
        states.append(state)

    states = np.stack(states)

    plt.figure()
    plt.scatter(states[:,0],states[:,1])
    plt.scatter(start_state[0],start_state[1])
    plt.scatter(goal_loc[0],goal_loc[1])
    # plt.xlim(-1.1,1.1)
    # plt.ylim(-1.1,1.1)
    plt.savefig('states')

    
        