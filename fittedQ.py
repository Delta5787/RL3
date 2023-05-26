import joblib
import gymnasium as gym
from tqdm.notebook import tqdm
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

############################################################
# FittedQ
############################################################


class discrete_random_agent:
    def __init__(self, actions):
        self.actions = actions

    def choose_action(self, state):
        return np.random.choice(self.actions)


def generate_historical_steps(env, agent, n_steps):
    steps = [None] * n_steps
    current_state = env.reset()[0]
    for i in range(n_steps):
        action = agent.choose_action(current_state)
        next_state, reward, terminated, truncated, info = env.step([action])

        if terminated or truncated:
            reward = -1.0
            steps[i] = (current_state, action, next_state, reward)
            current_state = np.array(env.reset()[0])
        else:
            steps[i] = (current_state, action, next_state, reward)
            current_state = next_state
    return steps

# Fit a model, provided with parameters


def fitted_Q_function(s_dim, a_dim, steps, model, gamma, n_iter, discrete_actions):
    flattened_steps = np.array([[*state, action, *state_p, reward]
                               for (state, action, state_p, reward) in steps])
    x = flattened_steps[:, 0:s_dim+a_dim]
    next_states = flattened_steps[:, x.shape[1]:x.shape[1]+s_dim]
    rewards = flattened_steps[:, -1]
    y = rewards.copy()
    next_x = np.zeros((len(discrete_actions), x.shape[1]))
    next_x[:, s_dim:] = np.array(
        discrete_actions).reshape(next_x[:, s_dim:].shape)

    for i in tqdm(range(n_iter)):
        model.fit(x, y)
        for i, next_state in enumerate(next_states):
            next_x[:, :s_dim] = next_state
            max_prediction = model.get_max_prediction(next_x)
            y[i] = rewards[i] + gamma * max_prediction
    return model

# Interface for facilitating the training and the use


class SklearnFittedModel():
    def __init__(self, model):
        self.model = model

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def get_max_prediction(self, x):
        return self.model.predict(x).max()

    def get_argmax_prediction(self, x):
        return self.model.predict(x).argmax()

############################################################
# FittedQ models
############################################################


def fq1():
    env = gym.make('InvertedPendulum-v4')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    num_actions = 100
    n_iter = 100

    model = ExtraTreesRegressor(n_estimators=10)
    model = SklearnFittedModel(model)

    actions = np.linspace(
        env.action_space.low[0], env.action_space.high[0], num=num_actions)
    agent = discrete_random_agent(actions)

    steps = generate_historical_steps(env, agent=agent, n_steps=10000)
    fitted_model = fitted_Q_function(s_dim=s_dim, a_dim=a_dim, steps=steps,
                                     model=model, gamma=0.95, n_iter=n_iter, discrete_actions=actions)
    joblib.dump(model, "simple-extratrees-10-10000nsteps-100iters.joblib")


def fq2():
    env = gym.make("InvertedDoublePendulum-v4")
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    num_actions = 100
    n_iter = 100

    model = ExtraTreesRegressor(n_estimators=10)
    model = SklearnFittedModel(model)

    actions = np.linspace(
        env.action_space.low[0], env.action_space.high[0], num=num_actions)
    agent = discrete_random_agent(actions)

    steps = generate_historical_steps(env, agent=agent, n_steps=10000)
    fitted_model = fitted_Q_function(s_dim=s_dim, a_dim=a_dim, steps=steps,
                                     model=model, gamma=0.95, n_iter=n_iter, discrete_actions=actions)
    joblib.dump(model, "double-extratrees-10-10000nsteps-10iters.joblib")