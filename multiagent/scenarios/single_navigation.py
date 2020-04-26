import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.num_agents = args.num_agents
        world.num_landmarks = args.num_landmarks
        world.step_unknown = args.step_unknown
        world.unknown_decay = args.unknown_decay
        world.decay_episode = args.decay_episode
        world.critic_full_obs = args.critic_full_obs
        world.num_reset = 0
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        world.select_goal = np.random.randint(0, world.num_landmarks)
        world.agents[0].goal = world.landmarks[world.select_goal]
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()

        world.assign_landmark_colors()

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        
        world.select_goal = np.random.randint(0, world.num_landmarks)
        world.agents[0].goal = world.landmarks[world.select_goal]
        world.world_step = 0
        world.num_reset += 1
        if world.unknown_decay and (world.step_unknown>1) and (world.num_reset%world.decay_episode==0):
            world.step_unknown = world.step_unknown - 1
        

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        if agent1 is agent2:
            return False
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal.state.p_pos)))
        rew -= dist

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world, critic_full_obs):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            
        stable_obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos)

        # index
        index = np.zeros(world.num_landmarks)
        if critic_full_obs:
            # actor
            if world.world_step > world.step_unknown:
                index = np.zeros(world.num_landmarks)
            else:
                index = np.zeros(world.num_landmarks)
                index[world.select_goal] = 1
            # critic
            index_critic = np.zeros(world.num_landmarks)
            index_critic[world.select_goal] = 1
            
            all_obs = np.append(index, stable_obs)
            all_obs_critic = np.append(index_critic, stable_obs)
            
            return all_obs, all_obs_critic       
        else:
            if world.world_step > world.step_unknown:
                index = np.zeros(world.num_landmarks)
            else:
                index = np.zeros(world.num_landmarks)
                index[world.select_goal] = 1
                            
            all_obs = np.append(index, stable_obs)
    
            return all_obs
