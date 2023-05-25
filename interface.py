
import torch
import gymnasium as gym

from reinforce import REINFORCE_Agent, REINFORCE_Policy
from ppo import PPO_Agent, PPO_Policy, PPO_Critic

import torch
import gymnasium as gym



class Controller():

    invertedPendulum = "InvertedPendulum-v4"
    doublePendulum = "InvertedDoublePendulum-v4"

    fqi = "FQI"
    reinforce = "REINFORCE"
    ppo = "PPO"

    def __init__(self, envName="InvertedPendulum-v4", algo="REINFORCE", load=False, render_mode=None) :

        self.env = gym.make(envName, render_mode=render_mode)
        self.algo = algo.upper()
        self.envName = envName
        self.objects = []
        self.gamma = .95

        if (algo == Controller.reinforce):
            self.objects.append(REINFORCE_Policy(inChannel=self.env.observation_space.shape[0], lr=1e-3))
            self.agent = REINFORCE_Agent(observation_space=self.env.observation_space, action_space=self.env.action_space, gamma=self.gamma)

        elif (algo == Controller.ppo):
            self.objects.append(PPO_Policy(self.env.observation_space.shape[0], lr=1e-4))
            self.objects.append(PPO_Critic(self.env.observation_space.shape[0], lr=1e-3))
            self.agent = PPO_Agent(observation_space=self.env.observation_space, action_space=self.env.action_space, gamma=self.gamma, epsilon=.2)
        
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


if __name__ == "__main__":
    controller = Controller(envName=Controller.doublePendulum, load=True, render_mode="human")
    controller.test()
