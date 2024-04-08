"""A script for training a DQN agent and storing best policy """

import nasim_with_defender
from nasim_with_defender.agents.dqn_agent import DQNAgent
from nasim_with_defender.agents.dqn_agent_defender import DQNAgent_Defender

if __name__ == "__main__":

    env = nasim_with_defender.make_benchmark('tiny_with_defender',
                                             fully_obs=True,
                                             flat_actions=True,
                                             flat_obs=True)
    dqn_agent = DQNAgent(env,
                         lr=0.001,
                         training_steps=2000000,
                         batch_size=32,
                         replay_size=10000,
                         final_epsilon=0.05,
                         exploration_steps=1000,
                         gamma=0.99,
                         hidden_sizes=[64, 64],
                         target_update_freq=1000,
                         verbose=True,
                         )
    dqn_agent_defender = DQNAgent_Defender(env,
                                           lr=0.001,
                                           training_steps=2000000,
                                           batch_size=32,
                                           replay_size=10000,
                                           final_epsilon=0.05,
                                           exploration_steps=1000,
                                           gamma=0.99,
                                           hidden_sizes=[64, 64],
                                           target_update_freq=1000,
                                           verbose=True,)
    episode = 0
    while episode < 5000:
        episode += 1
        o, _ = env.reset()
        done = False
        env_step_limit_reached = False

        steps = 0
        ak_episode_return = 0
        df_episode_return = 0

        while not done and not env_step_limit_reached and steps < 1000:
            #attacker
            a = dqn_agent.get_egreedy_action(o, dqn_agent.get_epsilon())

            next_o, r, done, env_step_limit_reached, _ = dqn_agent.env.step(a)
            dqn_agent.replay.store(o, a, next_o, r, done)
            dqn_agent.steps_done += 1
            loss, mean_v = dqn_agent.optimize()
            dqn_agent.logger.add_scalar("attacker_loss", loss, dqn_agent.steps_done)
            dqn_agent.logger.add_scalar("attacker_mean_v", mean_v, dqn_agent.steps_done)

            o = next_o
            ak_episode_return += r
            steps += 1
            #defender
            a_d = dqn_agent_defender.get_egreedy_action(o, dqn_agent_defender.get_epsilon())

            next_o_d, r_d, done_d, env_step_limit_reached_d, _ = dqn_agent_defender.env.step(a_d)
            dqn_agent_defender.replay.store(o, a, next_o, r, done)
            dqn_agent_defender.steps_done += 1
            loss_d, mean_v_d = dqn_agent_defender.optimize()
            dqn_agent_defender.logger.add_scalar("defender_loss", loss_d, dqn_agent_defender.steps_done)
            dqn_agent_defender.logger.add_scalar("defender_mean_v", mean_v_d, dqn_agent_defender.steps_done)

            df_episode_return += r_d
            o = next_o_d

        print(f'###########################################################')
        print(f'Attacker \n episode:{episode},reward:{ak_episode_return}, '
              f'steps:{steps}, done:{dqn_agent.env.goal_reached()}')
        print(f'Defender \n episode:{episode},reward:{df_episode_return}, '
              f'steps:{steps}, done:{dqn_agent_defender.env.goal_reached()}')