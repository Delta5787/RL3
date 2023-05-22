import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import gymnasium as gym
import matplotlib.pyplot as plt

class NormalDistribParam(nn.Module):
    def __init__(self, inChannel):
        super().__init__()
        self.inLayer = nn.Sequential(
            nn.Linear(inChannel, 64),
            nn.Tanh()
        )
        self.linLayer1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.linLayer2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.linLayer3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.muLayer = nn.Sequential(
            nn.Linear(64, 1),
        )
        self.sigmaLayer = nn.Sequential(
            nn.Linear(64, 1),
        )
    def forward(self, x):
        x = self.inLayer(x.float())
        x = self.linLayer1(x)
        x = self.linLayer2(x)
        x = self.linLayer3(x)
        return self.muLayer(x),torch.log(1+torch.exp(self.sigmaLayer(x))) 

class REINFORCE_Policy():
    def __init__(self, inChannel, lr):
        self.regressor = NormalDistribParam(inChannel)
        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.regressor.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
    def load(self, model):
        weights = torch.load(model)
        self.regressor.load_state_dict(weights["model_state_dict"])
        self.optimizer.load_state_dict(weights["optimizer"])

class REINFORCE_Agent():
    def __init__(self,observation_space, action_space, gamma):
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.cst = 1e-7
    def take_action(self,policy:REINFORCE_Policy, state):
        mean, stdv = policy.regressor(state)
        distrib = Normal(mean[0]+self.cst, stdv[0]+self.cst)
        action = distrib.sample()
        p = distrib.log_prob(action)

        return action, p
    def update(self,policy:REINFORCE_Policy, nEpisode:int, frequenceUpdate:int, env:gym.Env, plot=True):
        probs = []
        rewards = []
        loss = 0
        lenEpisode = []
        ttlRewardEpisode = []
        policy.regressor.train()
        avg_rewards = []
        for episode in range(nEpisode+1):
            state, _ = env.reset()
            running = True
            reward = []
            prob = []
            t = 0
            while(running):
                state = torch.tensor(np.array([state]))
                action, p = self.take_action(policy=policy, state=state)
                state, r, terminated, truncated, _ = env.step(action)
                prob.append(p)
                reward.append(r)
                avg_rewards.append(r)
                running = not (terminated or truncated)
                
            probs.append(prob)
            rewards.append(reward)
            ttlRewardEpisode.append(sum(reward))
            lenEpisode.append(len(prob))

            if(episode%1000 == 0):

                loss = 0
                policy.optimizer.zero_grad()

                print("AVG nStep : ", np.mean([len(probs[n]) for n in range(len(probs))]))
                for n in range(len(probs)):
                    for t in range(len(probs[n])):
                        loss += -1*probs[n][t]*torch.sum(torch.tensor([rewards[n][t]*self.gamma**(t_prime-t) for t_prime in range(t, len(probs[n]))]))
                
                loss.backward()
                policy.optimizer.step()
                loss = 0
                if(episode%(frequenceUpdate*5)==0):
                  print("lr reduce")
                  policy.scheduler.step()
                print(f"Episode : {episode}/{nEpisode} Average Reward : {np.mean(avg_rewards)}")
                probs = []
                rewards = []
                avg_rewards = []
        if plot:
            plt.figure()
            plt.plot(ttlRewardEpisode)
            plt.grid()
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Total Reward per episode")

            plt.figure()
            plt.plot(lenEpisode)
            plt.grid()
            plt.xlabel("Episode")
            plt.ylabel("Length of an episode")
            plt.title("Length of each episode episode")

    def test(self, policy:REINFORCE_Policy, env:gym.Env):
      running = True
      state, _ = env.reset()
      policy.regressor.eval()
      reward = 0
      step = 0
      while running:
        state = torch.tensor(np.array([state]))
        action, p = self.take_action(policy=policy, state=state)
        state, r, terminated, truncated, _ = env.step(action)
        reward += r
        step += 1

        running = not (terminated or truncated)
      return reward, step

if __name__ == "__main__":
    print("NOT TO RUN")