# 2020 0214 zoe
# This is for meta-learning
# 2 agent cooperation(good blue, dummy green), push 1 agent(adv red), to 1 landmark.

import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        # dummy-adv-good. The order can not be changed!!!
        num_dummies = 1
        num_adversaries = 1 # being pushed
        num_push_agents = 1
        num_agents = num_adversaries + num_push_agents + num_dummies
        num_landmarks = 1
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            
            agent.silent = True
            if i < num_dummies:
                agent.adversary = False
                agent.dummy = True 
                agent.collide = True   
            elif (i < (num_adversaries + num_dummies)):
                agent.adversary = True
                agent.dummy = False  
                agent.collide = False  
            else:
                agent.adversary = False
                agent.dummy = False 
                agent.collide = True 

            agent.size = 0.3 if agent.adversary else 0.1
        
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()

        world.assign_landmark_colors()

        # set random initial states
        '''
        # totally random
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        '''
        # limit
        for agent in world.agents:
            if agent.adversary:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
                adv_size = agent.size
                adv_pos = agent.state.p_pos

        for agent in world.agents:
            if not agent.adversary:
                agent.state.p_pos = np.zeros(world.dim_p)
                costheta = np.random.uniform(-1, +1)
                agent.state.p_pos[0] = adv_pos[0] - (agent.size + adv_size) * costheta
                operation = np.random.randint(-1,1)
                if (operation==-1):
                    sintheta = (-1) * np.sqrt (1 - costheta ** 2)
                else:
                    sintheta = np.sqrt (1 - costheta ** 2)
                
                agent.state.p_pos[1] = adv_pos[1] - (agent.size + adv_size) * sintheta

                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
                      
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            for a in world.agents:
                if a.adversary:
                    dists = np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                    rew -= dists
                    if dists < 0.1:
                        occupied_landmarks += 1

        # TODO
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent) and (not a.adversary) and (not agent.adversary):
                    rew -= 1
                    collisions += 1
        
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        if agent1.collide and agent2.collide:
            delta_pos = agent1.state.p_pos - agent2.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = agent1.size + agent2.size
            return True if dist < dist_min else False
        else:
            return False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            for a in world.agents:
                if a.adversary:
                    dists = np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                    rew -= dists

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent) and (not a.adversary) and (not agent.adversary):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        other_pos = []
        # distance between adv and target
        for entity in world.landmarks:  # world.entities:
            for a in world.agents:
                if a.adversary:
                    entity_pos.append(entity.state.p_pos - a.state.p_pos)  
                    adv_pos = a.state.p_pos

        # entity colors
        # entity_color = []
        # for entity in world.landmarks:  # world.entities:
        #     entity_color.append(entity.color)
        # communication of all other agents
        # comm = []
        
        for a in world.agents:
            if not a.adversary:
                other_pos.append(agent.state.p_pos - a.state.p_pos)
            # comm.append(other.state.c)
            
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [adv_pos] + entity_pos + other_pos)
