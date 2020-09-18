# Particle-Swarm-Optimization
Using PSO to train an agent to balance a pole

![](uploads/CartPole.gif)

PSO is an optimization method in which there is a group of "particles" that each represent an approach to a given problem. As the particles try to find an optimal solution, their behaviors are influenced by the most succesful of the group, causing them to converge, or "swarm", upon such a solution.

![](uploads/ParticleSwarmArrowsAnimation.gif)
###### GIF Source: Wikipedia

![](uploads/eq_1.png)
###### Velocity Update; Source: High Dimensional Adaptive Particle Swarm Optimization on Heterogeneous Systems
![](uploads/eq_2.png)
###### Particle Update; Source: High Dimensional Adaptive Particle Swarm Optimization on Heterogeneous Systems

This method is not guaranteed to converge to the global optimal solution. Behavioral changes are governed by velocities affected by parameter selection, to which this method is very sensitive to. 

I coded the most simple implementation here as well as its adaptive version for a relatively easy environment to solve. Each particle in the swarm is evaluated after playing through an entire episode of the game. A "particle" in this case is a weight matrix that, when multiplied by a given state of the environment outputs, outputs actions to take. 

The adaptive version automatically changes the intertia parameter based on the clustering of particles.

![](uploads/eq_3.png)
###### Inertia Update; Source: High Dimensional Adaptive Particle Swarm Optimization on Heterogeneous Systems

An inertia of <0.3 indicates swarm convergence and a value of >0.7 indicates the swarm needs to escape local optima. Values between are standard eploration and eploitation.

There are many upgrades to this optimization algorithm that I'll add at a later time.

The maximum score for this environment is 500.
