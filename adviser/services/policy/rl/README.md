# Purpose:
The `rl` folder contains code related to creating/training a reinforcement learning (RL) agent for dialog policy learning

# File Descriptions:
* `policy_rl.py`: Base class for creating RL-based policies. Includes a lot of utilities (e.g. automatic beliefstate to state-vector conversions, entity
* `experience_buffer`: Interface for off-policy experience buffers. Contains concrete implementations for random uniform and prioritized buffers. querying, action space, ...). Inherit from this class to create a concrete RL-based policy (for an example, have a look at `dqnpolicy.py`)
* `models`: A folder for trained RL policy models

## Example-Implementations
* `dqn.py`: Different Deep-Q Network architectures: DQN and Dueling DQN
* `dqnpolicy.py`: Concrete DQN-based policy (implementation of the RLPolicy`-interface) with options to configure as DQN, Dueling DQN, Double DQN or an arbitrary combination of those.
* `train_dqnpolicy.py`: script for training an DQN-policy (from `dqnpolicy.py`)
* `reinforce.py`: Different Deep-Q Network architectures: REINFORCE
* `reinforcepolicy.py`: Concrete REINFORCE-based policy (implementation of the RLPolicy`-interface) with options to configure.
* `train_reinforce_policy.py`: script for training an REINFORCE-policy (from `reinforcepolicy.py`)

## Implementation-Tests
* `reinforce-cartpole` implemented gym-cartpole with tensorflow and REINFORCE-algorithm