
import torch
import gymnasium as gym

from reinforce import *

invertedPendulum = "InvertedPendulum-v4"
doublePendulum = "InvertedDoublePendulum-v4"

fqi = "FQI"
reinforce = "REINFORCE"

class Controller():
    def __init__(self, envName="InvertedPendulum-v4", algo="REINFORCE", load=False, render_mode=None) :
        self.env = gym.make(envName, render_mode=render_mode)
        self.algo = algo
        self.envName = envName
        if (algo == reinforce):
            self.gamma = .95
            self.policy = REINFORCE_Policy(self.env.observation_space.shape[0], 1e-3)
            self.agent = REINFORCE_Agent(self.env.observation_space, self.env.action_space, .95)
            if load:
                self.policy.load("weights/"+reinforce.lower()+"/"+self.envName+self.algo+"save.pt")

    def train(self, nEpisode=30000, frequenceUpdate=500,savePath=None, plot=True):
        if(self.algo ==  "REINFORCE"):
            self.agent.update(self.policy, nEpisode, frequenceUpdate, self.env, plot)
            if not (savePath is None):
                torch.save({"model_state_dict":self.policy.regressor.state_dict(), "optimizer":self.policy.optimizer.state_dict()}, savePath+self.envName+self.algo+"save.pt")
    def test(self):
      print("Start Test with "+self.algo+" on "+self.envName)
      if(self.algo ==  reinforce):
        r, s = self.agent.test(self.policy, self.env)
      print("End Test with "+self.algo+" on "+self.envName)
      print(f"Total reward {r} for {s} steps")

if __name__ == "__main__":
    controller = Controller(envName=doublePendulum, load=True, render_mode="human")
    controller.test()
