# Comprehensive Plan: Fixing Environment Bugs and Upgrading to 6 Players

This document outlines a detailed step-by-step plan to fix the known environment bugs identified in `readme.md` (50bb pot, wrong pot sizes, incorrect action order after flop) and to upgrade the reinforcement learning environment and neural network to support 6-player No-Limit Texas Hold'em.

## 1. Fix Existing Environment Bugs
Since `rlcard` is used as the underlying environment, fixing these core rules requires either patching the `rlcard` local installation, monkey-patching it at runtime, or creating a custom fork of the environment inside the project (recommended).

### A. 50bb Pot Issue (Stack Size)
- **Bug**: The default `rlcard` No-Limit Hold'em environment initializes players with 50 big blinds. Standard deep RL environments (like ACPC) use 100bb.
- **Files Impacted**: Custom `rlcard` environment fork (e.g., `rlcard/games/nolimitholdem/game.py`) or initialization arguments.
- **Change Required**: Update the initial chips setting.
- **Pseudo-code**:
  ```python
  # In the game initialization
  def __init__(self, allow_step_back=False, num_players=6):
      self.init_chips = 100  # assuming 1bb = 1
  ```

### B. Wrong Pot/Bet Sizes
- **Bug**: Incorrect calculation of minimum and maximum allowed raises.
- **Files Impacted**: Custom `rlcard` fork (e.g., `rlcard/games/nolimitholdem/judger.py` or `round.py`).
- **Change Required**: Ensure that the legal actions accurately compute the minimum raise (usually `max(2 * last_raise, big_blind)`) and the maximum raise (all-in / remaining stack).
- **Pseudo-code**:
  ```python
  min_raise = max(2 * last_raise_amount, big_blind)
  max_raise = current_player.remained_chips
  valid_actions = [fold, check/call]
  if min_raise <= max_raise:
      valid_actions.extend(range(min_raise, max_raise + 1))
  ```

### C. Wrong Action Order After Flop
- **Bug**: Pre-flop starts with the player left of the Big Blind (UTG). Post-flop (Flop, Turn, River), the action should start with the Small Blind, or the closest active player left of the Dealer button. `rlcard` likely fails to shift this correctly.
- **Files Impacted**: Custom `rlcard` fork (e.g., `rlcard/games/nolimitholdem/round.py`).
- **Change Required**: Add logic to properly reset the `current_player` index at the beginning of each post-flop stage.
- **Pseudo-code**:
  ```python
  def start_new_stage(self):
      if self.stage in [FLOP, TURN, RIVER]:
          # Find the first active player starting from small blind
          self.current_player = self._get_next_active_player(self.button_index)
  ```

---

## 2. Upgrade to 6-Player Environment

### A. Modify Environment Wrapper (`agi/nl_holdem_env.py`)
- **Impact**: Expand observation and state tracking from 2 players to 6 players. Increase the tracked history length per round, as 6 players will take more actions per round than 2 players.
- **Changes Required**:
  1. Initialize `rlcard` with 6 players.
  2. Increase the first dimension of `action_info` from `4` to `8` (6 players + 2 indicator matrices for current action and legal actions).
  3. Expand `extra_info` from `(2,)` to `(6,)` to track the stakes of all 6 players.
  4. Expand history truncation from `[:6]` (max 6 actions per round) to `[:20]` to accommodate larger multi-way pots.
  5. Update `NlHoldemEnvWithOpponent` to randomize `our_pid` across 6 seats.

- **Pseudo-code**:
  ```python
  # 1. Init
  self.env = rlcard.make('no-limit-holdem', config={'seed': seed, 'allow_num_players': 6, 'num_players': 6})
  
  # 2. Update Space shapes
  space = {
      'card_info': spaces.Box(low=-1024, high=1024, shape=(4,13,6)),
      'action_info': spaces.Box(low=-256, high=256, shape=(8, self.action_num, 4 * 20 + 1)), # 6 players + 2 indicators, 20 actions/round
      'extra_info': spaces.Box(low=-256, high=256, shape=(6,)), # 6 players
      'legal_moves': spaces.Box(...)
  }
  
  # 3. Observation Array Initialization
  action_info = np.zeros([8, self.action_num, 4 * 20 + 1], np.uint8)
  extra_info = np.zeros([6], np.uint8)
  
  # 4. History loop adjustment
  for ind_round, one_history in enumerate(self.history):
      for ind_h, (player_id, action_id, legal_actions) in enumerate(one_history[:20]): # Increased from 6 to 20
          action_info[player_id, action_id, ind_round * 20 + ind_h] = 1
          action_info[6, action_id, ind_round * 20 + ind_h] = 1  # 6 is current action
          for la_ind in legal_actions:
              action_info[7, la_ind, ind_round * 20 + ind_h] = 1 # 7 is legal actions
              
  # 5. Extract 6 players' stakes
  for i in range(6):
      extra_info[i] = obs[0]["raw_obs"]["stakes"][i]
  
  # 6. PID randomization in reset()
  self.our_pid = random.randint(0, 5) # 6-player seat assignment
  ```

### B. Modify Neural Network Architecture (`agi/nl_holdem_net.py`)
- **Impact**: The network's input layers must match the new observation space sizes. Additionally, there is a hidden bug in the current implementation where `input_extra_info` is ignored.
- **Changes Required**:
  1. Change `input_action_info` shape to `(8, 5, 81)` (where 81 is `4 * 20 + 1`).
  2. Change `input_extra_info` shape to `(6,)`.
  3. **Fix Existing Bug**: `last_layer_extra` is currently incorrectly passing `last_layer_card` into its dense layer instead of `input_extra_info`. This must be corrected so the network actually observes the stakes.

- **Pseudo-code**:
  ```python
  input_action_info = tf.keras.layers.Input(shape=(8, 5, 81), name="action_info")
  input_extra_info = tf.keras.layers.Input(shape=(6,), name="extra_info")
  
  # ... existing conv blocks ...
  
  # Fix the bug by passing `input_extra_info` instead of `last_layer_card`
  last_layer_extra = tf.keras.layers.Dense(
      16,
      name="extra_fc",
      activation=tf.nn.relu,
      kernel_initializer=normc_initializer(0.01))(input_extra_info) # FIXED
  ```

### C. Update Training Configurations (`confs/*.py`)
- **Impact**: Any hardcoded lengths or configurations must be updated to align with the new 6-player structure.
- **Changes Required**: Ensure that dictionaries specifying history length or network configurations align with the new shapes.
- **Pseudo-code (e.g., `confs/nl_holdem.py`)**:
  ```python
  "env_config": {
      'custom_options': {
          'num_players': 6,
          'history_len': 20, # Reflects the increased action history length
          # ...
      },
  }
  ```

## Summary of Impact
- **Accuracy**: Fixing the `rlcard` mechanics ensures that the agent learns actual Texas Hold'em rules, particularly post-flop positional advantages and realistic bet sizing.
- **Complexity**: Extending to 6 players exponentially increases the state space. The agent must now process actions and stakes for 5 opponents instead of 1.
- **Model Parameters**: The `action_info` CNN input grows from 25 to 81 channels, and from 4 to 8 rows, slightly increasing the parameter count of the first convolutional layer and thus memory footprint.
- **Training Time**: Simulating 6 players is slower. Expect a drop in environment steps-per-second, requiring longer wall-clock time for the model to converge.
