## Usage

interface.py contains a main block of code with commented lines you can simply choose which model to visualize by uncommenting those lines.

## Presentations

The class Controller allows interaction with the gymnasium environments "InvertedPendulum-v4", "InvertedDoublePendulum-v4" with policies learned with FQI, REINFORCE or PPO (clipped version).
Only REINFORCE and fittedQ were "successfully" implemented. Provided they're not optimal as training could go much longer, they serve as proof of concepts for the project.

## Controller details

### Parameters

> envName : "InvertedPendulum-v4" or "InvertedDoublePendulum-v4" <br>
> algo : "FQI", "REINFORCE", or "PPO" algorithme to use to learn the policy <br>
> load : True load pretrained weights for the models, default False <br>
> render_mode : render_mode of the gym Env, default None (no feedback), "human" for a graphic feedback <br>

### Methods

> load : load the weights of the pretrained models <br>
> save : save the weights of the pretrained models <br>
> train : train a new policy from scratch or from the pretrained models <br>
    >> M and N training parameters see the algorithm for more details <br>
    >> save : if True, save the weights of the models, default True <br>
    >> verbose : if True, print feedback while training (i.g. average episode duration ...) and plot graphs, defaul True <br>
> test : test the policy over n run <br>

## Algorithms Details 

### FQI

> fitted_Q_function trains the model on a given set of element
> fq1 and fq2 are examples of execution <br>

### REINFORCE

> REINFORCE_Policy contains elements to train a neural network that will serve as a policy (Neural Network, optimize, learning rates, scheduler) <br>

> REINFORCE_Agent contains the training loops (update) and choose_action that will return an action given a policy
    >> N : number of episode to train on
    >> M : how often to update the weights of the policy

(the networks are pretrained for N = 29000 and M = 250)

### PPO

> PPO_Policy contains elements to train a neural network that will serve as a policy (Neural Network, optimize, learning rates, scheduler) <br>

> PPO_Critic contains element to train a critic (Neural Network, optimize, learning rates) <br>

> PPO_Agent contains the training loops (update) and choose_action that will return an action given a policy
    >> N : number of time we must sample trajectories (could be use for multiprocessing in futur implementations)
    >> M : total number of iteration (one iteration contains batch sampling and training over 40 epochs)