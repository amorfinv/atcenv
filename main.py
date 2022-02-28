"""
Example
"""

import numpy as np

if __name__ == "__main__":
    import random
    random.seed(52)
    from jsonargparse import ArgumentParser, ActionConfigFile
    from atcenv import Environment
    import time
    from tqdm import tqdm

    # RL models
    from atcenv.DDPG.DDPG import DDPG
    import atcenv.DDPG.TempConfig as tc
    from atcenv.SAC.sac import SAC
    import copy

    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(Environment, 'env')

    # parse arguments
    args = parser.parse_args()

    # init environment
    env = Environment(**vars(args.env))

    #RL = DDPG()
    RL = SAC()

    load_models = False

    if load_models:
        RL.load_models()
    # increase number of flights
    tot_rew_list = []
    # run episodes
    state_list = []
    for e in tqdm(range(args.episodes)):        
        episode_name = "EPISODE_" + str(e) 

        # reset environment
        # train with an increasing number of aircraft
        number_of_aircraft = 5 #min(int(e/1000)+2,10)
        obs = env.reset(number_of_aircraft)
        for obs_i in obs:
            RL.normalizeState(obs_i, env.max_speed, env.min_speed)
        # set done status to false
        done = False

        # save how many steps it took for this episode to finish
        number_steps_until_done = 0
        # save how many conflics happened in eacj episode
        number_conflicts = 0
        tot_rew = 0
        # execute one episode
        while not done:
            # get actions from RL model
            actions = []
            for obs_i in obs:
                # print(obs_i)
                actions.append(RL.do_step(obs_i,env.max_speed, env.min_speed))

            obs0 = copy.deepcopy(obs)

            # perform step with dummy action
            obs, rew, done_t, done_e, info = env.step(actions)

            for obs_i in obs:
                RL.normalizeState(obs_i, env.max_speed, env.min_speed)

            if done_t or done_e:
                done = True

            for obs_i in obs:
                state_list.append(obs_i)
            tot_rew += rew
            # train the RL model
            for it_obs in range(len(obs)):
                RL.setResult(episode_name, obs0[it_obs], obs[it_obs], rew[it_obs], actions[it_obs], done_e)
                # print('obs0,',obs0[it_obs],'obs,',obs[it_obs],'done_e,', done_e)
            # comment render out for faster processing
            if e%25 == 0:
                env.render()
                time.sleep(0.01)
            number_steps_until_done += 1
            number_conflicts += sum(env.conflicts)

            
                
        if len(tot_rew_list) < 100:
            tot_rew_list.append(sum(tot_rew)/number_of_aircraft)
        else:
            tot_rew_list[e%100 -1] = sum(tot_rew)/number_of_aircraft
        # save information
        RL.learn() # train the model
        if e%100 == 0:
            RL.save_models()
        #RL.episode_end(episode_name)
        np.savetxt('states.csv', state_list)
        tc.dump_pickle(number_steps_until_done, 'results/save/numbersteps_' + episode_name)
        tc.dump_pickle(number_conflicts, 'results/save/numberconflicts_' + episode_name)
        print(episode_name,'ended in', number_steps_until_done, 'runs, with', number_conflicts, 'conflicts, reward (rolling av100)=', np.mean(np.array(tot_rew_list)))        

        # close rendering
        env.close()
