"""
Main file. 
"""



if __name__ == "__main__":
    import random
    random.seed(42)
    from jsonargparse import ArgumentParser, ActionConfigFile
    from atcenv import Environment
    import time
    from tqdm import tqdm

    # RL models
    from atcenv.DDPG.DDPG import DDPG
    from atcenv.MADDDPG.maddpg import MADDPG
    import atcenv.DDPG.TempConfig as tc
    from atcenv.SAC.sac import SAC
    import copy
    import numpy as np    
    from sklearn.cluster import KMeans
    from pandas import DataFrame
    from shapely.geometry import Point

    STATE_SIZE = 8
    ACTION_SIZE = 2
    NUMBER_ACTORS_MARL = 10
    EVOLUTION_EPISODE = 10
    
    EVO_REACHED_W = 50
    EVO_CONF_W    = -0.1

    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(Environment, 'env')

    # parse arguments
    args = parser.parse_args()

    # init environment
    env = Environment(**vars(args.env))

    #RL = DDPG()
    RL = MADDPG(NUMBER_ACTORS_MARL, STATE_SIZE, ACTION_SIZE)
    #RL = SAC()

    # increase number of flights
    rew_list = []
    state_list = []

    tot_rew_list = []
    conf_list = []

    obs_list = []
    
    # Genetic stuff
    agent_conflict_storage = []
    agent_conflicts_this_episode = [0] * NUMBER_ACTORS_MARL
    
    agent_reached_storage = []
    
    # run episodes
    for e in tqdm(range(args.episodes)):   
        print('--------------------------------------------------------')     
        episode_name = "EPISODE_" + str(e) 

        # reset environment
        # train with an increasing number of aircraft
        #number_of_aircraft = min(int(e/100)+1,10)
        number_of_aircraft = 10
        obs = env.reset(number_of_aircraft, NUMBER_ACTORS_MARL)

        # set done status to false
        done = False

        # save how many steps it took for this episode to finish
        number_steps_until_done = 0
        # save how many conflics happened in eacj episode
        number_conflicts = 0
        tot_rew = 0
        it = 0
        # execute one episode
        while not done:           
            obs0 = copy.deepcopy(obs)
            
            # for ob in obs0:
            #     obs_list.append(RL.normalizeState(ob,env.max_speed, env.min_speed))
            # print(obs0)
            # get actions from RL model 
            if type(RL) is MADDPG:
                actions = [ [] for _ in range(number_of_aircraft)]
                if number_of_aircraft > NUMBER_ACTORS_MARL: # we need to divide all aircraft into groups based on their current position
                    n_cluster = int(np.ceil(number_of_aircraft/NUMBER_ACTORS_MARL))
                    ids = []
                    x = []
                    y = []
                    for flight_idx in range(env.num_flights):
                        if flight_idx not in env.done:
                            ids.append(flight_idx)
                            x.append(env.flights[flight_idx].position.x)
                            y.append(env.flights[flight_idx].position.y)
                    df = DataFrame( {'id': ids, 'x': x, 'y': y})
                    kmeans = KMeans(n_clusters = n_cluster).fit(df[['x', 'y']])    
                    cluster_indexes = [ [] for _ in range(n_cluster)]
                    # kmeans does not limit the number of points per cluster, so we have to do it ourselfs
                    for flight_idx in range(env.num_flights):
                        distance_to_centers = [0] * n_cluster
                        for idx_center in range(n_cluster):
                            distance_to_centers[idx_center] = env.flights[flight_idx].position.distance(Point(kmeans.cluster_centers_[idx_center]))
                        picked_center = 0
                        while len(cluster_indexes[np.argsort(distance_to_centers)[picked_center]]) >=NUMBER_ACTORS_MARL:
                            picked_center +=1
                        cluster_indexes[np.argsort(distance_to_centers)[picked_center]].append(flight_idx)
                    for cluster_idx in range(n_cluster):            
                        indexes = np.array(cluster_indexes[cluster_idx])
                        obs_cluster = np.array(obs)[indexes]
                        actions_aux = RL.do_step(obs_cluster, episode_name, env.max_speed, env.min_speed)
                        for index in indexes:
                            actions[index] = actions_aux.pop()
                else:
                    actions  = RL.do_step(obs, episode_name, env.max_speed, env.min_speed)
            else:
                for obs_i in obs:
                    actions = RL.do_step(obs_i, episode_name, env.max_speed, env.min_speed)

            # perform step with dummy action
            obs, rew, done, info = env.step(actions, type(RL) is MADDPG, NUMBER_ACTORS_MARL)
            obs2 = copy.deepcopy(obs) # obs will get normalized
            for rew_i in rew:
                rew_list.append(rew_i)
            for obs_i in obs:
                state_list.append(obs_i)
            
            tot_rew += rew
            # train the RL model
            # comment out on testing

            if number_steps_until_done > 0:
                if type(RL) is MADDPG:
                    if number_of_aircraft > NUMBER_ACTORS_MARL:
                        for clusters_idx in range(n_cluster):
                            indexes = np.array(cluster_indexes[cluster_idx])
                            obs0_cluster = np.array(obs0)[indexes]
                            obs_cluster  =  np.array(obs)[indexes]
                            actions_cluster =  np.array(actions)[indexes]
                            rew_cluster = sum(rew[indexes])
                            RL.setResult(episode_name, obs0_cluster, obs_cluster, rew_cluster, actions_cluster, done, env.max_speed, env.min_speed)
                    else:                    
                        rew = sum(rew)             
                        RL.setResult(episode_name, obs0, obs2, rew, actions, done, env.max_speed, env.min_speed)
                else:
                    for it_obs in range(len(obs)):
                        RL.setResult(episode_name, obs0[it_obs], obs2[it_obs], rew[it_obs], actions[it_obs], done, env.max_speed, env.min_speed)

            # comment render out for faster processing
            #if e%10 == 0:
                #env.render()
            number_steps_until_done += 1
            number_conflicts += len(env.conflicts)
            
            #time.sleep(0.05)

        if len(tot_rew_list) < 100:
            tot_rew_list.append(sum(tot_rew)/number_of_aircraft)
            conf_list.append(number_conflicts)
        else:
            tot_rew_list[e%100 -1] = sum(tot_rew)/number_of_aircraft
            conf_list[e%100 -1] = number_conflicts
            
        # save information
        agent_conflict_storage.append(agent_conflicts_this_episode)
        agent_conflicts_this_episode = [0] * NUMBER_ACTORS_MARL
        
        temp_list = [0] * NUMBER_ACTORS_MARL
        
        for ac_id_done in env.done:
            temp_list[ac_id_done] = 1
            
        agent_reached_storage.append(temp_list)
        
        ############## GENETICS ################
        if len(agent_conflict_storage) == EVOLUTION_EPISODE:
            # We perform artificial selection
            agent_conflict_storage = np.array(agent_conflict_storage)
            agent_reached_storage = np.array(agent_reached_storage)
            
            # Get the evaluations for conflicts and reached
            conflicts_per_ac = np.sum(agent_conflict_storage, axis = 0)
            times_reached_per_ac = np.sum(agent_reached_storage, axis = 0)
            
            # Get the reward functions for each
            rewards_per_ac = conflicts_per_ac * EVO_CONF_W + times_reached_per_ac * EVO_REACHED_W
            
            # Find the two best agents
            best_1, best_2 = np.argsort(np.max(rewards_per_ac, axis=0))[[-2, -1]]
            
            print(f'Selected agents number {best_1} and {best_2}.')
            
            # Set all other agents as these ones
            to_set = best_1
            for i_agent in range(NUMBER_ACTORS_MARL):
                if i_agent == best_1 or i_agent == best_2:
                    continue
                
                #Set this agent as one of the bests
                print(f'Replacing agent {i_agent} with agent {to_set}.')
                RL.super_agent.agents[i_agent] = copy.deepcopy(RL.super_agent.agents[to_set])
                
                # Change the to_set
                if to_set == best_1:
                    to_set = best_2
                else:
                    to_set = best_1
                    
                agent_conflict_storage = []
                agent_reached_storage = []
            
        # comment out on testing
        RL.episode_end(episode_name)
        tc.dump_pickle(number_steps_until_done, 'results/save/numbersteps_' + episode_name)
        tc.dump_pickle(number_conflicts, 'results/save/numberconflicts_' + episode_name)

        print(f' {episode_name} ended in {number_steps_until_done} runs, with {number_conflicts} conflicts.')
        print(f'Number of aircraft: {number_of_aircraft}')
        print(f'Done aircraft: {len(env.done)}')  
        print(f'Done aircraft IDs: {env.done}')      

        print('conflicts (rolling av100)', np.mean(np.array(conf_list)), 'reward (rolling av100)=', np.mean(np.array(tot_rew_list)))        
        np.savetxt('rewards.csv', rew_list)
        np.savetxt('states.csv', state_list)
        # close rendering
        env.close()
