# Alpha NL Holdem - Configuration Guide

This document explains the parameters configured in `confs/nl_holdem.py` (and similar config files). These parameters are structured to be directly fed into Ray RLlib's `ImpalaConfig` and the custom Texas Holdem environment.

## Ray RLlib Parameters

These top-level parameters dictate how the distributed reinforcement learning algorithm (IMPALA) collects and trains on data.

- **`env`**: The name of the registered custom reinforcement learning environment. (`'NlHoldemEnvWithOpponent'`)
- **`rollout_fragment_length`**: The number of environment steps (actions) that each rollout worker collects in one chunk before sending it to the central learner.
- **`train_batch_size`**: The total number of steps combined into a single training batch before a gradient update is performed on the neural network.
- **`num_workers`**: The number of parallel rollout workers (CPU processes) allocated to gather game data. Scale this up based on available CPU cores.
- **`num_envs_per_env_runner`**: The number of independent environment instances simulated simultaneously within a single rollout worker. 
- **`num_gpus`**: The number of GPUs allocated to the central learner for backpropagation. Set to `0` to train strictly on CPU.
- **`gamma`**: The discount factor for future rewards. A value of `1` means future rewards are not discounted (undiscounted returns), which is standard for finite episodic games like poker.
- **`entropy_coeff`**: The coefficient for the entropy regularization term in the loss function. A higher value (like `1e-1`) strongly encourages exploration by penalizing the agent for becoming overly deterministic too quickly.
- **`lr`**: The learning rate for the optimizer (e.g., Adam) during training (e.g., `3e-4`).

## Model Configuration (`model`)

Settings for the neural network architecture.

- **`custom_model`**: The string name of the custom neural network class registered via `ModelCatalog` (e.g., `'NlHoldemNet'`).
- **`max_seq_len`**: The maximum sequence length used for memory/recurrent mechanisms. Ensures that batch padding aligns properly for sequential state inputs.
- **`custom_model_config`**: A dictionary for any additional, model-specific hyperparameters passed directly to the custom model's `__init__` function.

## Environment Configuration (`env_config.custom_options`)

These parameters are passed directly to `NlHoldemEnvWithOpponent` to dictate the rules and state representation of the Texas Holdem simulation.

- **`weight`**: Defines the initial model weight loading scheme. `'default'` starts fresh.
- **`cut`**: A list of array bounds that define subsets or partitioning of features. This usually maps to partitioning the 54-card deck (e.g., four suits of 13 cards: 0-12, 13-25, 26-38, 39-51, plus special board tokens).
- **`epsilon`**: Used for epsilon-greedy action sampling or bounds in custom logic (e.g., `0.15`).
- **`tracker_n`**: The size of the rolling window/buffer used to track internal environment metrics during a session (e.g., `1000`).
- **`conut_bb_rather_than_winrate`**: Controls reward interpretation. Instead of evaluating purely by win/loss rates, a value like `2` indicates that the environment calculates rewards in terms of big blinds (mbb/h) won or lost.
- **`use_history`**: A boolean flag (`True`/`False`) indicating whether the observation state should include the sequential history of betting actions from the current round.
- **`use_cardnum`**: A boolean flag indicating whether numeric representations of card counts or specific card features should be explicitly included in the observations.
- **`history_len`**: The maximum length of the action history matrix passed into the neural network (e.g., the last `20` actions).
- **`num_players`**: The number of players seated at the poker table (`6`). Note that this setup primarily focuses on 1v1 (heads-up) scenarios even if the table size is allowed to be larger.
