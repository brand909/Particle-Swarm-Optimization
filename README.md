# Particle-Swarm-Optimization
Using PSO to train an agent to balance a pole

![](uploads/CartPole.gif)

PSO is an optimization method in which there is a group of "particles" that each represent an approach to a given problem. As the particles try to find an optimal solution, their behaviors are influenced by the most succesful of the group, causing them to converge, or "swarm", upon such a solution.

![](uploadsParticleSwarmArrowsAnimation.gif)
###### GIF Source: Wikipedia

This method is not guaranteed to converge to the global optimal solution. Behavioral changes are governed by velocities affected by parameter selection, to which this method is very sensitive to. 

I coded the most simple implementation here for a relatively easy environment to solve. Each particle in the swarm is evaluated after playing through an entire episode of the game. A "particle" in this case is a weight matrix that, when multiplied by a given state of the environment outputs, outputs actions to take. There are many upgrades to this optimization algorithm that I'll add at a later time.

The maximum score for this environment is 500.
