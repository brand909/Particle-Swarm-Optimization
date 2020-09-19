import gym
import numpy as np
from collections import deque
import random
import math

env = gym.make('CartPole-v1')

class Particle():
    def __init__(self, s_size=4, a_size=2, env=env):
        self.w = 1e-4*np.random.rand(s_size, a_size)  # weights for simple linear policy: state_space x action_space
        self.bw = self.w # best weight
        self.v = 1e-4*np.random.uniform(-1, 1) # velocity

    def forward(self, state, weights):
        x = np.dot(state, weights)
        return np.exp(x)/sum(np.exp(x))

    def act(self, state, weights):
        probs = self.forward(state, weights)
        #action = np.random.choice(2, p=probs) # option 1: stochastic policy
        action = np.argmax(probs)              # option 2: deterministic policy
        return action

    def episode(self, weights, n_episodes=1, max_t=1000, gamma=1.0):
        for i_episode in range(1, n_episodes+1):
            rewards = []
            state = env.reset()
            for t in range(max_t):
                action = self.act(state, weights)
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                if done:
                    break
            discounts = [gamma**i for i in range(len(rewards)+1)]
            R = sum([a*b for a,b in zip(discounts, rewards)])
            return R

def S1(f):
    if 0 <= f <= 0.4:
        return 0
    if 0.4 < f <= 0.6:
        return 5*f - 2
    if 0.6 < f <= 0.7:
        return 1
    if 0.7 < f <= 0.8:
        return -10*f + 8
    if 0.8 < f <= 1:
        return 0

def S2(f):
    if 0 <= f <= 0.2:
        return 0
    if 0.2 < f <= 0.3:
        return 10*f - 2
    if 0.3 < f <= 0.4:
        return 1
    if 0.4 < f <= 0.6:
        return -5*f + 3
    if 0.6 < f <= 1:
        return 0

def S3(f):
    if 0 <= f <= 0.1:
        return 1
    if 0.1 < f <= 0.3:
        return -5*f + 1.5
    if 0.3 < f <= 1:
        return 0

def S4(f):
    if 0 <= f <= 0.7:
        return 0
    if 0.7 < f <= 0.9:
        return 5*f - 3.5
    if 0.9 < f <= 1:
        return 1
    
def update_state(f, prev_state):
    """Fuzzy state transition implemented with singleton defuzzification and transition logic.

    Params
    ======
        f (float): evolution factor calculated from distance matrices of the particles
        prev_state (string): the previous behavior state
    """
    transitions = {'S1':'S2', 'S2':'S3', 'S3':'S4', 'S4':'S1'} # transition logic
    states = {'S1':S1(f), 'S2':S2(f), 'S3':S3(f), 'S4':S4(f)} # output according to each state
    potential_states = [i for i in states.keys() if states[i] != 0] # candidates for transition
    roots = [i for i in transitions.keys() for j in potential_states if j == transitions[i]] # origins of potential states
    
    if len(potential_states) == 1: # not fuzzy 
        return max(states, key=lambda key: states[key])
    elif prev_state in potential_states: # fuzzy, but no state change for stability
        return prev_state
    elif prev_state in roots: # fuzzy with a connection; follow connection
        return transitions[prev_state]
    else: # fuzzy without a connection; return maximum of potential states
        return max(states, key=lambda key: states[key])
    
def update_c(c, state):
    """Update to the parameters which govern exploration and exploitation.

    Params
    ======
        c (list of floats/ints): c1 and c2
        state (str): 
    """
    delta = np.random.uniform(0.05,0.1)
    if state == 'S1':
        return c[0]+delta, c[1]-delta
    if state == 'S2':
        return c[0]+0.5*delta, c[1]-0.5*delta
    if state == 'S3':
        return c[0]-0.5*delta, c[1]+0.5*delta
    if state == 'S4':
        return c[0]-delta, c[1]+delta
        
def adaptive_pso(p_count, lr):
    """Implementation of adaptive particle swarm optimization.

    Params
    ======
        p_count (int): number of particles
        lr(float): learn rate
    """
    particles = [Particle() for i in range(p_count)]  
    highest_reward = 0
    bsp = random.choice(particles) # best swarm particle
    wv = 0.9
    c1 = 2
    c2 = 2
    state = 'S1'
    for i in particles:
        reward = i.episode(i.bw)
        if reward > highest_reward:
            highest_reward = reward
            bsp.w = i.bw

    for t in range(40): # no. of iterations
        rewards = deque(maxlen=p_count)
        for particle in particles:
            rp = np.random.rand(particle.w.shape[0], particle.w.shape[1])
            rg = np.random.rand(particle.w.shape[0], particle.w.shape[1])
            particle.v = wv*particle.v + c1*rp*(particle.bw-particle.w) + c2*rg*(bsp.w-particle.w)
            particle.w += lr*particle.v

            reward = particle.episode(particle.w)
            rewards.append(reward)
            print(reward)

            if reward > particle.episode(particle.bw):
                particle.bw = particle.w
                if particle.episode(particle.bw) > bsp.episode(bsp.w):
                    bsp.w = particle.bw
        
        # calculate distance matrix and phi
        d_list = [] # distances
        for i in particles:
            d_list.append(np.sum([np.sqrt((j.w - i.w)**2) for j in particles if i not in [j]]))
        d_g = np.sum([np.sqrt((bsp.w - q.w)**2) for q in particles if q not in [bsp]])
        d_g = d_g * 1/(p_count-1) # average distance from global
        d_max = max(d_list) * 1/(p_count-1)
        d_min = min(d_list) * 1/(p_count-1)
        phi = (d_g - d_min) / (d_max - d_min)
        # update wv
        wv = 1 / (1+1.5*math.exp(-2.6*phi))
        # calculate state change and update c parameters
        state = update_state(phi,state)
        c1, c2 = update_c((c1,c2),state)
        
        print("wv: ", wv)
        print('state: ',state)
        print("c1,c2: ",c1,c2)

        print("---------------Iteration through all particles complete---------------")
        if np.mean(rewards)==500:
            print("*All particles converged unto the maximum*")
            return particles[0]
        if t == 39:
            end_rewards = [particle.episode(particle.bw) for particle in particles]
            maximum_reward = max(end_rewards)
            for i in range(len(end_rewards)):
                if end_rewards[i] == maximum_reward:
                    return particles[i]          
            
if __name__ == "__main__":
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)
    y = adaptive_pso(30,0.075)
    state = env.reset()
    for t in range(1000):
        env.render()
        action = y.act(state, y.bw)
        state, reward, done, _ = env.step(action)
        if done:
            env.close()
            break