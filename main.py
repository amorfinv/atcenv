"""
Example
"""

import numpy as np

if __name__ == "__main__":
    import random
    random.seed(52)
    from jsonargparse import ArgumentParser, ActionConfigFile
    from atcenv import Environment
    import atcenv.settings as cfg
    import time
    from tqdm import tqdm

    # RL model
    import atcenv.TempConfig as tc
    from atcenv.MASAC.masac_agent import MaSacAgent
    from atcenv.CCSP.ccsp_agent import CCSPAgent
    import copy

    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=cfg.NUMBER_EPISODES)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(Environment, 'env')

    # parse arguments
    args = parser.parse_args()

    # init environment
    env_args = vars(args.env)
    env_args['altitude'] = 0 # in meters
    env = Environment(**env_args)
    
    # load settings
    number_of_aircraft = cfg.NUMBER_AIRCRAFT
    use_altitude = cfg.USE_ALTITUDE
    num_intruders_state = cfg.NUMBER_INTRUDERS_STATE
    load_models = cfg.LOAD_MODELS
    test = cfg.TEST

    if use_altitude:
        action_dim = 3 #heading, speed, altitude
        state_dim = 5 + num_intruders_state * 6
    else:
        action_dim = 2 #heading, speed
        state_dim = 4 + num_intruders_state * 5
    
    if cfg.RL_MODEL == "MASAC":
        RL = MaSacAgent(number_of_aircraft, action_dim, state_dim, num_intruders_state, use_altitude)
    elif cfg.RL_MODEL == "CCSP":
        RL = CCSPAgent(number_of_aircraft, action_dim, state_dim, num_intruders_state, use_altitude)
    else:  
        raise Exception("Please select one of the available models from 'settings.cfg'.")

    if load_models:
        RL.load_models()

    tot_rew_list = []
    conf_list = []
    speeddif_list = []
    state_list = []

    for e in tqdm(range(args.episodes)):   
        print('\n-----------------------------------------------------')
    
        episode_name = "EPISODE_" + str(e) 
      
        obs = env.reset(number_of_aircraft, num_intruders_state, use_altitude)
        for obs_i in obs:
            RL.normalizeState(obs_i, env.max_speed, env.min_speed)
        # set done status to false
        done = False

        # save how many steps it took for this episode to finish
        number_steps_until_done = 0
        # save how many conflics happened in each episode
        number_conflicts = 0
        # save different from optimal speed
        average_speed_dif = 0

        tot_rew = 0
        # execute one episode
        while not done:

            actions = RL.do_step(obs,env.max_speed, env.min_speed, test=test)

            obs0 = copy.deepcopy(obs)

            obs, rew, done_t, done_e, info = env.step(actions)

            for obs_i in obs:
               RL.normalizeState(obs_i, env.max_speed, env.min_speed)

            if done_t or done_e:
                done = True

            tot_rew += rew
            
            # Make sure that the state is filled,
            # Different states are used for the different models due to the different Critics

            if cfg.RL_MODEL == "MASAC":
                while len(obs) < len(obs0):
                    obs.append( [0] * RL.statedim)

            elif cfg.RL_MODEL == "CCSP":
                while len(obs) < number_of_aircraft and len(env.done) != number_of_aircraft and not test:
                    i = np.random.randint(0,len(obs))
                    obs = np.vstack((obs,obs[i]))

            if not test:
                RL.setResult(episode_name, obs0, obs, sum(rew), actions, done_e)

            if cfg.RENDER:
                if e%cfg.RENDER_FREQ == 0:
                    env.render()

            number_steps_until_done += 1
            number_conflicts += len(env.conflicts)
            average_speed_dif = np.average([env.average_speed_dif, average_speed_dif])            
                
        if len(tot_rew_list) < 100:
            tot_rew_list.append(sum(tot_rew)/number_of_aircraft)
            conf_list.append(number_conflicts)
            speeddif_list.append(average_speed_dif)
        else:
            tot_rew_list[e%100 -1] = sum(tot_rew)/number_of_aircraft
            conf_list[e%100 -1] = number_conflicts
            speeddif_list[e%100 -1] = average_speed_dif

        if e%100 == 0 and not test:
            RL.save_models()

        tc.dump_pickle(number_steps_until_done, 'results/save/numbersteps_' + episode_name)
        tc.dump_pickle(number_conflicts, 'results/save/numberconflicts_' + episode_name)
        tc.dump_pickle(average_speed_dif, 'results/save/speeddif_' + episode_name)
        print(f'Done aircraft: {len(env.done)}')  
        print(f'Done aircraft IDs: {env.done}')      

        print(episode_name,'ended in', number_steps_until_done, 'runs, with', np.mean(np.array(conf_list)), 'conflicts (rolling av100), reward (rolling av100)=', np.mean(np.array(tot_rew_list)), 'speed dif (rolling av100)', np.mean(np.array(average_speed_dif)))        

        env.close()