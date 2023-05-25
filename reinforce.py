# Implementation inspired by : https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/

import numpy as np
import torch
from torch.distributions.normal import Normal
import gymnasium as gym
import matplotlib.pyplot as plt
from utils import NormalDistribParam

class REINFORCE_Policy(): # Use to generate a policy

    def __init__(self, inChannel, lr, load=False):
        
        self.lr = lr # learning rates

        self.regressor = NormalDistribParam(inChannel) # Neural Network use to generate the parameters of the policy (mean, var)
        self.optimizer = torch.optim.AdamW(self.regressor.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    def load(self, envName): # Load weights
        
        weights = torch.load("weights/reinforce/"+envName+"REINFORCEsave.pt")
        
        self.regressor.load_state_dict(weights["model"])
        self.optimizer.load_state_dict(weights["optimizer"])
        self.scheduler.load_state_dict(weights["scheduler"])

    def save(self, envName): # save weights
        
        torch.save({
            "model":self.regressor.state_dict(),
            "optimizer":self.optimizer.state_dict(),
            "scheduler":self.scheduler.state_dict()
        }, "weights/reinforce/"+envName+"REINFORCEsave.pt")

class REINFORCE_Agent():
    def __init__(self,observation_space, action_space, gamma):
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.gamma = gamma # Reward discount factor
        self.cst = 1e-7 # Small number for mathematical stability
    
    def choose_action(self,policy:REINFORCE_Policy, state): # choose an action given the current state
        
        mean, stdv = policy.regressor(state)
        
        distrib = Normal(mean[0]+self.cst, stdv[0]+self.cst)
        
        action = distrib.sample()
        p = distrib.log_prob(action)

        return action, p

    def update(self,objects:list[REINFORCE_Policy],  N:int, M:int, env:gym.Env, verbose=True, scheduler=False):
        # Update the policy
        
        policy = objects[0]
        nEpisode = N # Will be trained over N episodes
        frequenceUpdate = M # frequence of the update

        probs = []
        rewards = []
        loss = 0
        lenEpisode = []
        ttlRewardEpisode = []

        policy.regressor.train()

        avg_rewards = []

        for episode in range(nEpisode+1): # Generate trajectories

            state, _ = env.reset()
            running = True
            reward = []
            prob = []
            t = 0

            while(running):

                state = torch.tensor(np.array([state]))
                action, p = self.choose_action(policy=policy, state=state)
                state, r, terminated, truncated, _ = env.step(action)
                prob.append(p)
                reward.append(r)
                avg_rewards.append(r)
                running = not (terminated or truncated)
                
            probs.append(prob)
            rewards.append(reward)
            ttlRewardEpisode.append(sum(reward))
            lenEpisode.append(len(prob))

            if(episode%frequenceUpdate == 0):

                loss = 0
                policy.optimizer.zero_grad()

                print("AVG nStep : ", np.mean([len(probs[n]) for n in range(len(probs))]))
                for n in range(len(probs)):
                    for t in range(len(probs[n])):
                        loss += -1*probs[n][t]*torch.sum(torch.tensor([rewards[n][t]*self.gamma**(t_prime-t) for t_prime in range(t, len(probs[n]))]))
                
                loss.backward()
                policy.optimizer.step()
                loss = 0
                if(episode%(frequenceUpdate*5)==0)and(scheduler):
                  policy.scheduler.step()
                if(verbose):
                    print(f"Episode : {episode}/{nEpisode} Average Reward : {np.mean(avg_rewards)}")
                probs = []
                rewards = []
                avg_rewards = []
        if verbose:
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
        action, _ = self.choose_action(policy=policy, state=state)
        state, r, terminated, truncated, _ = env.step(action)
        reward += r
        step += 1

        running = not (terminated or truncated)
        
      return reward, step
