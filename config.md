# Current configuration

Configurations are Python dictionary literals loaded with `ast.literal_eval`. See `readme.md` for setup and commands.

- `num_env_runners`: rollout workers, default 2; override with `--workers`.
- `num_envs_per_env_runner`: environments per rollout worker.
- `num_gpus`: learner CUDA resources; set automatically by `--device`. `--gpus 0/1` remains an explicit compatibility override.
- `rollout_fragment_length`: sampled transitions per fragment.
- `train_batch_size`: learner batch size; override with `--batch-size`.
- `model.custom_model`: `NlHoldemNet` or `NlHoldemLgNet`.
- `model.custom_model_config.stack_scale`: chip-feature divisor, default 200.
- `env_config.custom_options.num_players`: 2 through 6, default 6.
- `env_config.custom_options.history_len`: latest actions retained per street; default 20 at six players, 6 heads-up.
- `env_config.custom_options.reward_scale`: learner chip-reward divisor, default 200.
- `env_config.custom_options.rwd_ratio`: additional learner reward multiplier, default 1.

Remaining older `custom_options` keys are retained for historical configs but do not control the current environment. The four provided configs use the tested old RLlib policy API stack; the training entry point explicitly disables the new RLModule/EnvRunner stacks. A different model or observation shape requires a matching checkpoint and fresh training. CLI `--sp` is the chance of using current-policy self-play when replacing the opponent; it does not assign separate policies to all five opponent seats.

`--device auto|cpu|cuda|mps` selects the learner backend. Automatic choice is CUDA, then MPS, then CPU. Rollout workers reserve zero GPUs. MPS forces the single-optimizer learner path and is experimental until verified on Mac hardware. CUDA uses native RLlib learning.
