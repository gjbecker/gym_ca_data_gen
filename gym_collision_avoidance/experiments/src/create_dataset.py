import os
import pickle
import time
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ["GYM_CONFIG_CLASS"] = "DataGeneration"
import gym_collision_avoidance.envs.test_cases as tc
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.experiments.src.env_utils import (
    create_env,
    policies,
    run_episode,
    store_stats,
)

def reset_env(
    env,
    test_case_fn,
    test_case_args,
    test_case,
    num_agents,
    policies,
    policy,
    prev_agents,
):
    # env.unwrapped.plot_policy_name = policy
    test_case_args["num_agents"] = num_agents
    test_case_args["prev_agents"] = prev_agents
    agents = test_case_fn(**test_case_args)
    
    env.set_agents(agents)
    init_obs = env.reset()
    print(f'ENV: {env}')
    # env.unwrapped.test_case_index = test_case
    return init_obs, agents


def main():
    np.random.seed(0)

    Config.EVALUATE_MODE = True
    Config.SAVE_EPISODE_PLOTS = True
    Config.SHOW_EPISODE_PLOTS = True
    Config.DT = 0.1
    Config.PLOT_CIRCLES_ALONG_TRAJ = True
    Config.RECORD_PICKLE_FILES = True
    Config.GENERATE_DATASET = True

    num_agents_to_test = range(3,4)
    num_test_cases = 1

    test_case_fn = tc.get_testcase_random
    test_case_args = {
            'policy_to_ensure': 'RVO',
            'policies': ['RVO', 'noncoop', 'static', 'random'],
            'policy_distr': [0.75, 0.10, 0.075, 0.075],
            # 'policy_distr': [1, 0, 0, 0],
            'speed_bnds': [0.5, 2.0],
            'radius_bnds': [0.2, 0.8],
            'side_length': [
                {'num_agents': [0,5], 'side_length': [4,7]}, 
                {'num_agents': [5,9], 'side_length': [7,11]},
                {'num_agents': [9,np.inf], 'side_length': [10,13]}
                ],
            'agents_sensors': ['other_agents_states'],
        }
    policies = ['mixed']

    env = create_env()

    print(
        "Running {test_cases} test cases for {num_agents} for policies:"
        " {policies}".format(
            test_cases=num_test_cases,
            num_agents=num_agents_to_test,
            policies=policies,
        )
    )
    with tqdm(
        total=len(num_agents_to_test)
        * len(policies)
        * num_test_cases
    ) as pbar:
        for num_agents in num_agents_to_test:
            env.set_plot_save_dir(
                os.path.dirname(os.path.realpath(__file__))
                + "/../../../DATA/results/{num_agents}_agents/figs/"
                .format(num_agents=num_agents)
            )
            for policy in policies:
                np.random.seed(0)
                prev_agents = None
                df = pd.DataFrame()
                datasets = []
                for test_case in range(num_test_cases):
                    ##### Actually run the episode ##########
                    _ = reset_env(
                        env,
                        test_case_fn,
                        test_case_args,
                        test_case,
                        num_agents,
                        policies,
                        policy,
                        prev_agents,
                    )
                    episode_stats, prev_agents, dataset = run_episode(env)
                    datasets.append(dataset)
                    
                    # print(episode_stats)
                    df = store_stats(
                        df,
                        {"test_case": test_case, "policy_id": policy},
                        episode_stats,
                    )
                    logging.info(f'EPISODE {test_case}: {episode_stats}')
                    ########################################
                    pbar.update(1)

                if Config.GENERATE_DATASET:
                    file_dir = os.path.dirname(os.path.realpath(__file__))+ "/../../../DATA/datasets/{}_{}_agent_{}".format(policy, num_agents, num_test_cases)
                    with open(file_dir + '.pkl', 'wb') as handle:
                        pickle.dump(datasets, handle, protocol=pickle.HIGHEST_PROTOCOL)       
                    print(f'Generated Dataset Length: {len(datasets)}')

                if Config.RECORD_PICKLE_FILES:
                    file_dir = os.path.dirname(
                        os.path.realpath(__file__)
                    ) + "/../../../DATA/results/"
                    file_dir += "{num_agents}_agents/stats/".format(
                        num_agents=num_agents
                    )
                    os.makedirs(file_dir, exist_ok=True)
                    log_filename = file_dir + "/stats_{}.p".format(policy)
                    df.to_pickle(log_filename)
    return True


if __name__ == "__main__":
    logging.basicConfig(filename=os.path.dirname(os.path.realpath(__file__))
                        + "/../../../DATA/datasets/DATASET.log", filemode='w', level=logging.DEBUG)
    logging.info('started')
    main()
    logging.info('finished')
    print("Experiment over.")