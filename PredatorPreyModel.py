#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:08:03 2020

Basic predator-prey agent based model.

    Scheme:
        1) Choose agent randomly
        2) If prey -> move and reproduce with certain probability
        3) If predator -> move -> if prey in vision, kill prey and reproduce
            with certain probability. Else die with certain probability

@author: Jan-Hendrik Niemann
"""


from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector


class RandomWalker(Agent):
    def __init__(self, unique_id, model, pos):
        """
        Parameters
        ----------
        pos : tuple
            x and y coordinates.

        """
        super().__init__(unique_id, model)
        self.pos = pos

    def move(self):
        new_pos = (self.pos[0] + self.model.random.normalvariate(0, 1),
                   self.pos[1] + self.model.random.normalvariate(0, 1))

        self.model.space.move_agent(self, new_pos)


class Prey(RandomWalker):
    def __init__(self, unique_id, model, pos, reproduction_probability):
        """
        Parameters
        ----------
        pos : tuple
            x and y coordinates.
        reproduction_probability : float
            Reproduction probability.

        """
        super().__init__(unique_id, model, pos)
        self.breed = 'Prey'
        self.reproduction_probability = reproduction_probability
        self.dead = False

    def step(self):

        # If dead, do not do anything
        if self.dead:
            return None

        # Move randomly
        self.move()

        # Reproduce
        if self.model.random.uniform(0, 1) < self.reproduction_probability:
            x = self.model.random.uniform(0, self.model.width)
            y = self.model.random.uniform(0, self.model.height)
            if self.model.local_offspring:
                x = self.model.random.normalvariate(self.pos[0], 1)
                y = self.model.random.normalvariate(self.pos[1], 1)
            offspring = Prey(self.model.next_id(),
                             self.model,
                             (x, y),
                             self.reproduction_probability)
            self.model.space.place_agent(offspring, (x, y))
            self.model.schedule.add(offspring)


class Predator(RandomWalker):
    def __init__(self, unique_id, model, pos, reproduction_probability, mortality, vision):
        """
        Parameters
        ----------
        pos : tuple
            x and y coordinates.
        reproduction_probability : float
            Reproduction probability.
        mortality : float
            Probability of dying
        vision : float
            Vision radius for search for prey

        """
        super().__init__(unique_id, model, pos)
        self.breed = 'Predator'
        self.vision = vision
        self.reproduction_probability = reproduction_probability
        self.mortality = mortality
        self.dead = False

    def step(self):

        if self.dead:
            return None

        # Move randomly
        self.move()

        # Look
        self.update_prey_in_vision()

        # Kill prey
        if self.prey_in_vision:
            prey = self.prey_in_vision[self.random.randint(0, len(self.prey_in_vision) - 1)]
            prey.dead = True
            self.model.schedule.remove(prey)
            self.model.space.remove_agent(prey)

            # Reproduce
            if self.model.random.uniform(0, 1) < self.reproduction_probability:
                x = self.model.random.uniform(0, self.model.width)
                y = self.model.random.uniform(0, self.model.height)
                if self.model.local_offspring:
                    x = self.model.random.normalvariate(self.pos[0], 1)
                    y = self.model.random.normalvariate(self.pos[1], 1)
                offspring = Predator(self.model.next_id(),
                                     self.model, (x, y),
                                     self.reproduction_probability,
                                     self.mortality,
                                     self.vision)
                self.model.space.place_agent(offspring, (x, y))
                self.model.schedule.add(offspring)

        # Die
        elif not self.prey_in_vision and self.model.random.uniform(0, 1) < self.mortality:
            self.dead = True
            self.model.schedule.remove(self)
            self.model.space.remove_agent(self)

    def update_prey_in_vision(self):

        self.neighbors = self.model.space.get_neighbors(self.pos,
                                                        radius=self.vision)
        self.prey_in_vision = []

        for agent in self.neighbors:
            if agent.breed == 'Prey' and agent.dead is False:
                self.prey_in_vision.append(agent)


class PredatorPreyModel(Model):
    def __init__(self,
                 height=100,
                 width=100,
                 init_prey=100,
                 prey_reproduction=0.03,
                 init_predator=10,
                 predator_vision=1,
                 predator_reproduction=0.5,
                 predator_death=0.02,
                 local_offspring=False,
                 max_iters=500,
                 seed=None):

        super().__init__()

        self.height = height
        self.width = width
        self.init_prey = init_prey
        self.prey_reproduction = prey_reproduction
        self.init_predator = init_predator
        self.predator_vision = predator_vision
        self.predator_reproduction = predator_reproduction
        self.predator_death = predator_death
        self.local_offspring = local_offspring
        self.iteration = 0
        self.max_iters = max_iters
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(height, width, torus=True)
        model_reporters = {
            'Prey': lambda model: self.count('Prey'),
            'Predator': lambda model: self.count('Predator'),
        }

        self.datacollector = DataCollector(model_reporters=model_reporters)

        # Place prey
        for i in range(self.init_prey):
            x = self.random.uniform(0, self.width)
            y = self.random.uniform(0, self.height)
            # next_id() starts at 1
            prey = Prey(self.next_id(), self, (x, y), self.prey_reproduction)
            self.space.place_agent(prey, (x, y))
            self.schedule.add(prey)

        # Place predators
        for i in range(self.init_predator):
            x = self.random.uniform(0, self.width)
            y = self.random.uniform(0, self.height)
            predator = Predator(self.next_id(),
                                self,
                                (x, y),
                                self.predator_reproduction,
                                self.predator_death,
                                self.predator_vision)
            self.space.place_agent(predator, (x, y))
            self.schedule.add(predator)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
        Advance the model by one step and collect data.

        Returns
        -------
        None.

        """

        self.schedule.step()
        self.iteration += 1

        self.datacollector.collect(self)

        # Stop system if maximum of iterations is reached
        if self.iteration > self.max_iters:
            self.running = False
            return None

    def count(self, breed):
        """
        Count agent by breed.

        Parameters
        ----------
        breed : string
            Breed of agent Can be 'Prey' or 'Predator'.

        Returns
        -------
        count : int
            Number of agents of type breed.

        """

        count = 0
        for agent in self.schedule.agents:
            if agent.breed == breed:
                count += 1

        if breed == 'Predator' and count == 0:
            self.running = False

        return count

    def warm_up(self):
        for agent in self.schedule.agents:
            if agent.breed == 'Prey':
                continue

            neighbors = self.space.get_neighbors(agent.pos, radius=agent.vision)

            for prey in neighbors:
                if prey.breed == 'Prey':
                    x = self.random.uniform(0, self.width)
                    y = self.random.uniform(0, self.height)
                    self.space.move_agent(prey, (x, y))


# %%

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import time

    # Parameter setting
    height = 100
    width = 100
    init_prey = 200
    prey_reproduction = 0.03
    init_predator = 20
    predator_vision = 3
    predator_reproduction = 0.5
    predator_death = 0.02
    iters = 1000
    local_offspring = False
    seed = 0

    PPM = PredatorPreyModel(height=height,
                            width=width,
                            init_prey=init_prey,
                            prey_reproduction=prey_reproduction,
                            init_predator=init_predator,
                            predator_vision=predator_vision,
                            predator_reproduction=predator_reproduction,
                            predator_death=predator_death,
                            local_offspring=local_offspring,
                            max_iters=iters,
                            seed=seed)

    # %% Run model

    start = time.time()
    PPM.run_model()
    stop = time.time()

    # Get simulation data
    model_out = PPM.datacollector.get_model_vars_dataframe()
    trajectory = model_out.to_numpy()

    print('\nElapsed time: %.4f seconds\n' % (stop - start))

    # %% Plot model

    fig = plt.figure()
    plt.step(np.linspace(0, trajectory.shape[0], trajectory.shape[0]), trajectory[:, 0], color='g')
    plt.step(np.linspace(0, trajectory.shape[0], trajectory.shape[0]), trajectory[:, 1], color='r')
    plt.legend(('Prey', 'Predator'))
    plt.xlabel('Time $t$')
    plt.ylabel('Number of agents')

    fig = plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 1], c='k')
    plt.xlabel('Number of prey')
    plt.ylabel('Number of predators')

    # %% Animate model

    PPM = PredatorPreyModel(height=height,
                            width=width,
                            init_prey=init_prey,
                            prey_reproduction=prey_reproduction,
                            init_predator=init_predator,
                            predator_vision=predator_vision,
                            predator_reproduction=predator_reproduction,
                            predator_death=predator_death,
                            local_offspring=local_offspring,
                            max_iters=iters,
                            seed=seed)

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(idx):
        fig.clf()
        PPM.step()

        num_prey = PPM.count('Prey')
        num_predator = PPM.count('Predator')

        for agent in PPM.schedule.agents:
            x, y = agent.pos

            if agent.breed == 'Prey':
                plt.plot(x, y, marker='.', c='g', markersize=10)

            if agent.breed == 'Predator':
                plt.plot(x, y, marker='.', c='r', markersize=10)
                circle = plt.Circle((x, y), predator_vision, color='r', fill=True, alpha=0.2)
                fig.gca().add_artist(circle)

        plt.xlim([0, 100])
        plt.ylim([0, 100])
        plt.title('Prey ' + str(num_prey) + '\nPredators ' + str(num_predator))

    ani = animation.FuncAnimation(fig,
                                  update,
                                  repeat=False,
                                  interval=10,
                                  frames=np.arange(1, iters, 1))
