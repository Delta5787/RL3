# Implementation inspired by :
# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb
#  https://github.com/ericyangyu/PPO-for-Beginners/blob/master/ppo.py#L55

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

import gymnasium as gym

import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

from utils import NNSingleOut


class PPO_Policy():  # Use to generate a policy
    def __init__(self, inChannel, lr):

        self.lr = lr  # Learning Rate of the Neural Network

        #  Neural Network to generate the mean of the distribution use to generate an action
        self.regressor = NNSingleOut(inChannel)
        self.optimizer = torch.optim.AdamW(
            self.regressor.parameters(), lr=self.lr)  #  Optimizer of the network
        # Scheduler, reduce the lr if the average episode duration stay the same for to long while testing
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max')

    def load(self, envName):  #  load the weights

        weights = torch.load("weights/ppo/"+envName+"PPO_policy.pt")

        self.regressor.load_state_dict(weights["model"])
        self.optimizer.load_state_dict(weights["optimizer"])
        self.scheduler.load_state_dict(weights["scheduler"])

    def save(self, envName):  # save the weights

        torch.save({
            "model": self.regressor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }, "weights/ppo/"+envName+"PPO_policy.pt")


class PPO_Critic():  # Use to approximate the V function
    def __init__(self, inChannel, lr):

        self.lr = lr  #  learning rates of the Neural Network

        # Neural Network that will approximate the V function
        self.NN = NNSingleOut(inChannel)
        self.optimizer = torch.optim.AdamW(self.NN.parameters(), lr=self.lr)

    def load(self, envName):

        weights = torch.load("weights/ppo/"+envName+"PPO_critic.pt")

        self.NN.load_state_dict(weights["model"])
        self.optimizer.load_state_dict(weights["optimizer"])

    def save(self, envName):

        torch.save({
            "model": self.NN.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, "weights/ppo/"+envName+"PPO_critic.pt")


class PPO_Agent():
    def __init__(self, observation_space, action_space, gamma, epsilon):

        self.observation_space = observation_space
        self.action_space = action_space

        self.gamma = gamma  # Reward discount factor
        self.epsilon = epsilon  # Clipping factor

        self.cst = 1e-7  # Small number for mathematical stability

        self.c1 = 0.005  # Coefficient of the MSE
        self.c2 = 0.001  # Coefficient of the entropy factor

        self.mse = nn.MSELoss()

        #  Covariance matrix to generate an action
        self.cov_var = torch.full(size=(1,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def choose_action(self, policy: PPO_Policy, critic: PPO_Critic, state):
        #  Create the distribution
        mean = policy.regressor(state)
        distrib = MultivariateNormal(mean, self.cov_mat)

        # Samle an action
        action = distrib.sample()
        # Old ln(p_action)
        p = distrib.log_prob(action)
        #  Old V(state)
        v = critic.NN(state)

        return action, p, v

    def evaluate(self, policy, critic, batch_state, batch_action):

        # Create a distribution
        mean = (policy.regressor(batch_state))
        distrib = MultivariateNormal(mean, self.cov_mat)

        action = torch.unsqueeze(action, 1)

        #  New ln(p_action)
        p = distrib.log_prob(batch_action)
        # New V(state)
        v = critic.NN(batch_state)
        #  Entropy of the distribution, will serve as entropy factor in the loss
        entr = distrib.entropy()

        return p, v, entr

    def update(self, objects, N: int, M: int, env: gym.Env, verbose=True, scheduler=False):
        # Update a new policy

        # Unpack objects
        # list[PPO_Policy, PPO_Critic]
        policy = objects[0]
        critic = objects[1]

        # For print
        test_reward = []
        test_step = []

        # Set the old policy
        oldPolicy = PPO_Policy(self.observation_space.shape[0], lr=policy.lr)
        oldPolicy.regressor.load_state_dict(policy.regressor.state_dict())

        # train for M iterations
        for m in range(M):

            old_state = []  # Save state for training
            old_actions = []  # Save actions for training
            old_p = []  # Save ln(p_action) for training
            old_adv = []  # Save advantage for training
            # Save True V(state) (discounted sum of reward) for training
            old_discRewards = []

            # Generate trajectories

            for n in range(N):  # for potentiel futur multi process implementation

                state, _ = env.reset()  #  init the environment

                # diverse buffer for the nexts time steps
                t_actions = []
                t_rewards = []
                t_p = []
                t_v = []
                t_state = []
                t_discRewards = []
                t_isTerminante = []

                for t in range(40):  # For t time steps

                    state = torch.tensor(np.array([state]))

                    t_state.append(state)
                    old_state.append(state)

                    # Choose an action
                    action, p, v = self.choose_action(
                        policy=oldPolicy, critic=critic, state=state)
                    action = action.view(1,)

                    # Make a step
                    state, reward, term, trunc, _ = env.step(action)

                    t_actions.append(action)
                    old_actions.append(action)
                    t_rewards.append(reward)
                    t_p.append(p)
                    old_p.append(p)
                    t_v.append(v)

                    t_isTerminante.append(term or trunc)

                    if (term or trunc):
                        state, _ = env.reset()

                # Discounted sum of reward
                t0 = 0

                for t in range(len(t_rewards)):
                    t_discRewards.append(
                        sum([t_rewards[j]*self.gamma**j for j in range(t0, t+1)]))
                    if (t_isTerminante[t]):

                        t0 = t+1
                # Normalisation of the results
                t_discRewards = torch.tensor(t_discRewards)
                t_discRewards = (
                    t_discRewards-t_discRewards.mean())/t_discRewards.std()

                #  Advantage
                for t in range(len(t_discRewards)):
                    old_discRewards.append(t_discRewards[t])
                    old_adv.append(t_discRewards[t]-t_v[t])

            # To tensor

            old_state = torch.squeeze(torch.stack(old_state, dim=0)).detach()
            old_actions = torch.squeeze(
                torch.stack(old_actions, dim=0)).detach()
            old_p = torch.squeeze(torch.stack(old_p, dim=0)).detach()
            old_discRewards = torch.squeeze(
                torch.stack(old_discRewards, dim=0)).detach()
            old_adv = torch.squeeze(torch.stack(old_adv, dim=0)).detach()

            #  Train
            avg_loss = []

            for k in range(40):  #  For k epochs

                policy.regressor.train()

                newP, newV, newEntr = self.evaluate(
                    policy=policy, critic=critic, state=old_state, action=old_actions)

                ratio = torch.exp(newP-old_p)

                # Surrogate loss
                loss1 = ratio * old_adv
                loss2 = torch.clamp(ratio, 1-self.epsilon,
                                    1+self.epsilon)*old_adv

                #               To maximize                 To minimize
                loss = -(torch.min(loss1, loss2) - self.c1 *
                         self.mse(torch.squeeze(newV), old_discRewards) + self.c2*newEntr)
                avg_loss.append(loss.detach().mean())

                # loss step
                policy.optimizer.zero_grad()
                critic.optimizer.zero_grad()
                loss.mean().backward()
                policy.optimizer.step()
                critic.optimizer.step()

            # Test of the new policy
            list_r = []
            list_s = []
            for _ in range(20):
                r, s = self.test(objects, env)
                list_r.append(r)
                list_s.append(s)

            test_reward.append(np.mean(list_r))
            test_step.append(np.mean(list_s))

            if (scheduler):
                policy.scheduler.step(test_step[-1])
            if (verbose):
                print(
                    f"Iter [{m}/{M}] TTL reward {test_reward[-1]} Episode Duration {test_step[-1]} Loss avg {np.mean(avg_loss)}")

            # update old policy for next iteration
            oldPolicy.regressor.load_state_dict(policy.regressor.state_dict())

        if (verbose):
            plt.figure()
            plt.grid()
            plt.plot(test_reward)
            plt.xlabel("Epochs")
            plt.ylabel("Average Sum of Reward Per Episode")
            plt.figure()
            plt.grid()
            plt.plot(test_step)
            plt.xlabel("Epochs")
            plt.ylabel("Average Duration of Episodes")

    def test(self, objects, env: gym.Env):
        # Unpack
        # list[PPO_Policy, PPO_Critic]
        policy = objects[0]
        critic = objects[1]

        # Freeze model weights
        policy.regressor.eval()
        state, _ = env.reset()

        r = 0
        s = 0
        running = True

        while (running):
            state = torch.tensor(np.array([state]))
            action, p, v = self.choose_action(
                policy=policy, critic=critic, state=state)
            action = action.view(1,)
            state, reward, term, trunc, _ = env.step(action)
            s += 1
            r += reward
            running = not (term or trunc)

        # release the policy
        policy.regressor.train()

        return r, s
