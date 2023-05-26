
import gymnasium as gym
import numpy as np
import joblib

from reinforce import REINFORCE_Agent, REINFORCE_Policy
from ppo import PPO_Agent, PPO_Policy, PPO_Critic
from fittedQ import SklearnFittedModel


class Controller():

    invertedPendulum = "InvertedPendulum-v4"
    doublePendulum = "InvertedDoublePendulum-v4"

    fqi = "FQI"
    reinforce = "REINFORCE"
    ppo = "PPO"

    def __init__(self, envName="InvertedPendulum-v4", algo="REINFORCE", load=False, render_mode=None):

        self.env = gym.make(envName, render_mode=render_mode)
        self.algo = algo.upper()
        self.envName = envName
        self.objects = []
        self.gamma = .95

        if (algo == Controller.reinforce):
            self.objects.append(REINFORCE_Policy(
                inChannel=self.env.observation_space.shape[0], lr=1e-3))
            self.agent = REINFORCE_Agent(
                observation_space=self.env.observation_space, action_space=self.env.action_space, gamma=self.gamma)

        elif (algo == Controller.ppo):
            self.objects.append(PPO_Policy(
                self.env.observation_space.shape[0], lr=1e-4))
            self.objects.append(PPO_Critic(
                self.env.observation_space.shape[0], lr=1e-3))
            self.agent = PPO_Agent(observation_space=self.env.observation_space,
                                   action_space=self.env.action_space, gamma=self.gamma, epsilon=.2)

        if load:
            self.load()

    def load(self):
        for item in self.objects:
            item.load(self.envName)

    def save(self):
        for item in self.objects:
            item.save(self.envName)

    def train(self, N, M, save=True, verbose=True):
        self.agent.update(self.objects, N, M, self.env, verbose)
        if (save):
            self.save()

    def test(self, run=1):
        avg_r = 0
        avg_s = 0
        for i in range(run):
            print("Start Test with "+self.algo+" on "+self.envName)
            r, s = self.agent.test(self.objects, self.env)
            avg_r += r
            avg_s += s
            print("End Test with "+self.algo+" on "+self.envName)
            print(f"Total reward {r} for {s} steps")
        avg_r /= run
        avg_s /= run
        print("Avg ttl reward :", avg_r, "Avg episode duration :", avg_s)

############################################################
# FittedQ vizualization
############################################################


def show_fittedQ(env, model_path):
    # loaded Model
    fitted_model = joblib.load(model_path)

    # Environment specs
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    # Discretization of the action space
    num_actions = 100
    discrete_actions = np.linspace(
        env.action_space.low[0], env.action_space.high[0], num=num_actions)
    x = np.zeros((len(discrete_actions), s_dim+a_dim))
    x[:, s_dim:] = np.array(discrete_actions).reshape(x[:, s_dim:].shape)

    # Initializing the current state
    current_state = env.reset()[0]

    for i in range(1000):
        x[:, :s_dim] = current_state
        action_i = fitted_model.get_argmax_prediction(x)
        action = [discrete_actions[action_i]]
        next_state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
        current_state = next_state


if __name__ == "__main__":
    # Controller for reinforce simple pendulum

    # controller = Controller(envName=Controller.invertedPendulum, load=True, render_mode="human")
    # controller.test()

    # Controller for reinforce double pendulum

    # controller = Controller(envName=Controller.doublePendulum, load=True, render_mode="human")
    # controller.test()

    # Rendering for fittedQ simple pendulum

    # env = gym.make('InvertedPendulum-v4', render_mode="human")
    # model_path = "./weights/fittedQ/simple-extratrees-10-20000nsteps-20iters.joblib"
    # show_fittedQ(env, model_path=model_path)

    # Rendering for fittedQ double pendulum

    # env = gym.make("InvertedDoublePendulum-v4",render_mode="human")
    # model_path = "./weights/fittedQ/double-extratrees-10-20000nsteps-20iters.joblib"
    # show_fittedQ(env, model_path=model_path)

    pass
