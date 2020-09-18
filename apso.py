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
         
def adaptive_pso(p_count, phi_p, phi_g, lr):
    """Implementation of adaptive particle swarm optimization.

    Params
    ======
        p_count (int): number of particles
        phi_p (float/int): scalar importance of particle's best weight
        phi_g (float/int): scalar importance of swarm's best weight
        lr(float): learn rate
    """
    particles = [Particle() for i in range(p_count)]  
    highest_reward = 0
    bsp = random.choice(particles) # best swarm particle
    wv = 0.7
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
            particle.v = wv*particle.v + phi_p*rp*(particle.bw-particle.w) + phi_g*rg*(bsp.w-particle.w)
            particle.w += lr*particle.v

            reward = particle.episode(particle.w)
            rewards.append(reward)
            print(reward)

            if reward > particle.episode(particle.bw):
                particle.bw = particle.w
                if particle.episode(particle.bw) > bsp.episode(bsp.w):
                    bsp.w = particle.bw
                    
        d_list = [] # distances
        for i in particles:
            d_list.append(np.sum([np.sqrt((j.w - i.w)**2) for j in particles if i not in [j]]))
        d_g = np.sum([np.sqrt((bsp.w - q.w)**2) for q in particles if q not in [bsp]])
        d_g = d_g * 1/(p_count-1) # average distance from global
        d_max = max(d_list) * 1/(p_count-1)
        d_min = min(d_list) * 1/(p_count-1)
        phi = (d_g - d_min) / (d_max - d_min)
        wv = 1 / (1+1.5*math.exp(-2.6*phi)) # update wv
        print("wv: ", wv)

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
    y = adaptive_pso(30,0.4,0.7,0.15)
    state = env.reset()
    for t in range(1000):
        env.render()
        action = y.act(state, y.bw)
        state, reward, done, _ = env.step(action)
        if done:
            env.close()
            break