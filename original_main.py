"""A script for training a DQN agent and storing best policy """

import nasim
import gymnasium
from nasim.agents.dqn_agent import DQNAgent


if __name__ == "__main__":
    env = gymnasium.make("TinyPO-v0", render_mode=None)
    dqn_agent = DQNAgent(env,
                         lr=0.001,
                         training_steps=20000,
                         batch_size=32,
                         replay_size=10000,
                         final_epsilon=0.05,
                         exploration_steps=1000,
                         gamma=0.99,
                         hidden_sizes=[64, 64],
                         target_update_freq=1000,
                         verbose=True,
                         )
    dqn_agent.train()
    dqn_agent.save()
    dqn_agent.run_eval_episode(render=False)