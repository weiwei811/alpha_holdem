# Alpha NL Holdem — six-player PyTorch training

This checkout uses 64-bit Python 3.11, PyTorch 2.8.0 (2.2.2 for Intel macOS), and Ray RLlib 2.49.2. The bundled RLCard engine is used directly; do not install another `rlcard` over it.

## Setup: Windows CUDA, Windows CPU, and macOS

Use **64-bit Python 3.11**. The setup and training launchers locate the repository from their own path and can be called from another directory. Direct Python commands below assume the repository root. The scripts create a local `.venv`; activation is optional because the examples call its Python directly.

### Windows with NVIDIA CUDA

```powershell
.\setup_env.ps1 -Device cuda
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`setup_env.ps1` without `-Device` detects a working NVIDIA driver and selects CUDA; otherwise it selects CPU. CUDA uses the official **PyTorch 2.8.0 + CUDA 12.6** wheel. You need a compatible NVIDIA driver; a separate CUDA toolkit installation is unnecessary. A matching CUDA wheel is not reinstalled.

For an exact installation of the verified Windows CUDA dependency set:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows-cuda.txt
```

CUDA inference, backward execution, and a training iteration were verified on an **RTX 2060 Max-Q with 6 GB VRAM**. A GPU-trained checkpoint also loaded successfully on CPU.

### Windows CPU

```powershell
.\setup_env.ps1 -Device cpu
```

Use `--device cpu` for training even if a CUDA wheel is installed.

### macOS: Apple Silicon or Intel

Install Python 3.11 and run:

```sh
sh setup_env.sh
.venv/bin/python -c 'import torch; print(torch.__version__); print("MPS available:", torch.backends.mps.is_available())'
```

The script installs the shared `requirements.txt`; platform markers select the matching PyTorch version for the Python architecture. Ray's wheels target macOS 12 or newer; Metal also requires a compatible macOS/PyTorch/hardware combination. Run an arm64 Python on Apple Silicon to use the current native PyTorch build.

Apple Silicon uses the standard **PyTorch 2.8.0 macOS wheel**, which includes MPS support. Intel Mac uses **PyTorch 2.2.2**, since PyTorch 2.8 does not publish an Intel macOS wheel. Ray 2.49.2 publishes wheels for both architectures. Both Mac dependency sets were resolved with binary wheels only; **Mac runtime execution has not been verified on hardware**. Do not install the Windows CUDA lockfile on a Mac.

### Which requirements file should I use?

There is one portable **`requirements.txt`**, with platform markers selecting the appropriate PyTorch version. It does not need separate manually maintained Windows and Mac dependency lists:

```text
torch==2.8.0; platform_system != "Darwin" or platform_machine == "arm64"
torch==2.2.2; platform_system == "Darwin" and platform_machine == "x86_64"
```

Windows and Apple Silicon share the PyTorch 2.8 API version. Windows CUDA additionally needs the CUDA wheel from the official CUDA index; installing only `requirements.txt` does not guarantee a CUDA-enabled build. Mac MPS is supplied by the normal Mac wheel, with no CUDA installation.

| File | Purpose |
|---|---|
| `requirements.txt` | Portable direct dependencies; platform markers choose PyTorch 2.8 or Intel Mac 2.2.2. Use when resolving dependencies afresh. |
| `requirements-windows-cuda.txt` | Exact verified Windows CUDA packages; includes the CUDA wheel index and `torch==2.8.0+cu126`. |

The optional Windows CUDA lock pins transitive dependencies for repeatable setup on the tested machine. Regenerate it when changing direct dependencies; use `requirements.txt` on other platforms. The bundled RLCard engine is used from this repository and must not be replaced with a separate pip installation.

Run the suite with the matching environment:

```powershell
# Windows
.\.venv\Scripts\python.exe -m pytest tests -q
```

```sh
# Mac
.venv/bin/python -m pytest tests -q
```

Windows CPU/CUDA verification passed **38 tests**, with **2 MPS hardware tests skipped**.

## Train six players

There are two training launchers: `run_training.ps1` for Windows and `run_training.sh` for Mac/Linux. Both use the repository's `.venv`, forward all CLI arguments, and share the Python entry point's defaults:

```powershell
.\run_training.ps1 --device cuda --workers 2 --checkpoint-steps 10000
```

```sh
sh run_training.sh --device cpu --workers 2 --checkpoint-steps 10000
# Apple Silicon GPU (experimental): use --device mps
```


```powershell
.\.venv\Scripts\python.exe train_league.py --conf confs/nl_holdem.py --workers 2 --checkpoint-steps 10000
```

Without `--iterations`, training continues until stopped. A bounded verification run is:

```powershell
.\.venv\Scripts\python.exe train_league.py --workers 1 --batch-size 100 --iterations 1 --checkpoint-steps 200 --output_dir league/quick_check
```

Defaults use six seats, 200 chips (100 big blinds), automatic learner-device selection, and two CPU rollout workers. The learner controls a randomly selected seat and the other five seats use a shared historical policy. All seats become learner seats across episodes. Opponents are selected before the first action. League profit includes hands ending before a learner decision (such as blind walks); those hands produce no learner training transition. This is a six-player self-play environment, with one learner trajectory per hand; it is not a six-policy multi-agent RLlib environment.

The four street histories keep separate seat IDs and include folds. The most recent `history_len` actions per street are retained; persistent seat status remains visible after history truncation. Observations also contain remaining stacks, contributions, alive/all-in status, button, and acting seat. Card and history tensors are binary floats; chip counts do not wrap at 255. The models derive dimensions from the observation space. Hidden and value layers use unit-scale initialization; the policy head uses scale 0.01 to begin with mild action preferences. Existing checkpoint tensors load unchanged; the new initialization applies to freshly trained models.

Learner rewards are net chips divided by `reward_scale` (default 200), multiplied by `rwd_ratio` (default 1). League statistics remain raw mean chip profit per hand, despite the historical `winrate` field name. Big blind is 2 chips, so mbb/hand is mean chip profit multiplied by 500. `--upwin` is a mean-chip-profit threshold, not a probability of winning.

`--training-seconds 3600` stops after an hour of training and saves full state at the end of the current iteration. SIGINT/SIGTERM also request a graceful save. Initialization and final saving add a little wall time.

Training progress is measured in **trained transitions**, not reporting iterations. An iteration can perform zero, one, or several optimizer updates. Checkpoints default to every 10,000 trained transitions and evaluations to every 50,000. Batch and minibatch sizes must be positive multiples of the configured rollout fragment (50 by default). Metrics collection waits at most one second by default (`--metrics-timeout`).

Train to an absolute transition target (the last batch may overshoot):

```powershell
.\.venv\Scripts\python.exe train_league.py --device cuda --workers 2 --trained-steps 1000000 --checkpoint-steps 10000 --eval-steps 50000 --eval-hands 600 --output_dir league/run_6p
```

Resume optimizer state, trained-step counters, and the historical league pool/statistics from the latest full checkpoint:

```powershell
.\.venv\Scripts\python.exe train_league.py --restore-state league/run_6p --output_dir league/run_6p --device cuda --workers 2 --trained-steps 2000000
```

Use the same model, observation and batch configuration when resuming. On Mac/Linux, use `.venv/bin/python` and `--device cpu` or experimental `--device mps`. The transition target is absolute, including restored progress; an already reached target does no further training. Full checkpoints are indexed by `OUTPUT_DIR/latest_checkpoint.json`. In-flight hands, asynchronous queues and random-number streams are not restored for exact replay.

For a weight-only restart with fresh optimizer/counters:

```powershell
.\.venv\Scripts\python.exe train_league.py --restore league/run_6p --output_dir league/restarted --workers 2
```

Fresh runs save the actual starting learner as `OUTPUT_DIR/initial_weight.pkl`. Historical opponents are under `OUTPUT_DIR/weights/c_N.pkl`; `output_weight.pkl` contains the latest learner. Full checkpoints also preserve league match counts and smoothed rewards. Older weight-only directories restore opponent weights/rewards but reset match counts. Bounded runs save `training_config.json`; logs are inside `work/ray_results/`.

Periodic evaluations append `OUTPUT_DIR/evaluation.jsonl`, with trained transitions, optimizer updates, chips per hand and approximate 95% confidence intervals against random legal actions, check/call, and the frozen initial policy. Each evaluation reuses the seed bank and rotates hero seats. Evaluation runs on CPU and pauses the training driver; 600 hands per opponent is a quick diagnostic, not proof of strength. Use thousands of hands and paired comparisons for conclusions. `--eval-steps 0` disables evaluations; enabled evaluations require the original initial checkpoint.

Use `--freeze-opponents --sp 0` to keep the historical opponent pool fixed during a controlled comparison. `--opponent-sampling ranked` (default) emphasizes difficult historical policies; `--opponent-sampling uniform` samples all policies in that pool equally. The CLI sampling choice applies after restoring state, and full checkpoints preserve the chosen strategy. Otherwise progress checkpoints add the current learner to the pool. `--checkpoint-steps 0` disables periodic checkpoints; bounded runs still save at exit. The legacy `--gap N` maps to `N * train_batch_size` trained transitions; `--upwin` is retained but no longer controls snapshots.

## Select a training device

### Windows CUDA

```powershell
.\.venv\Scripts\python.exe train_league.py --device cuda --workers 2 --checkpoint-steps 10000
```

### macOS CPU or experimental Apple Silicon GPU

```sh
# CPU
.venv/bin/python train_league.py --device cpu --workers 2 --checkpoint-steps 10000
# Metal/MPS, experimental
.venv/bin/python train_league.py --device mps --workers 2 --checkpoint-steps 10000
```

`--device auto` prioritizes CUDA, then MPS, then CPU. Use `--device cpu` to force CPU. An explicitly requested unavailable device fails clearly. The earlier `--gpus 1` remains a CUDA alias; `--gpus 0` forces CPU when device is auto. This entry point supports **one learner GPU**. Rollout workers and opponent policies remain on CPU.

MPS training uses a single-device adapter around the pinned old RLlib policy stack. It moves the learner and batches to Metal and bypasses RLlib's CUDA-only gradient context. The adapter's update path was tested against real IMPALA/V-trace loss on CPU and CUDA, but **actual Metal training remains experimental and unverified on Mac hardware**. Use CPU if an MPS operation is unsupported. Intel CPU is the conservative path; AMD-backed MPS availability depends on the installed PyTorch/macOS/hardware combination.

For a bounded GPU check on either platform, add `--batch-size 200 --iterations 1 --checkpoint-steps 200 --output_dir league/gpu_check`. Without `--iterations`, training runs continuously. On a 6 GB GPU, start with the default model and batch size before increasing memory use. Model-update speed improved in the local CUDA benchmark, but CPU game simulation can limit total training throughput.

Weight snapshots contain NumPy arrays. With matching model and observation configuration, device selection does not change parameter shapes or the checkpoint format. CPU/CUDA portability was verified; loading and running on MPS still requires a Mac hardware check.

## Play or evaluate a newly trained checkpoint

Windows:

```powershell
.\.venv\Scripts\python.exe gui/play_against_ai_in_ui.py --weights league/quick_check/output_weight.pkl --deterministic --device cpu
.\.venv\Scripts\python.exe gui/play_self.py --weights league/quick_check/output_weight.pkl --games 1000 --device cpu
```

Mac:

```sh
.venv/bin/python gui/play_against_ai_in_ui.py --weights league/quick_check/output_weight.pkl --deterministic --device cpu
.venv/bin/python gui/play_self.py --weights league/quick_check/output_weight.pkl --games 1000 --device cpu
```

Replace the weight path with your trained checkpoint. To accelerate inference, replace `--device cpu` with `--device cuda` on Windows or `--device mps` on a compatible Mac. GUI/evaluation default to CPU because transfer overhead can outweigh GPU benefits for single decisions. `--device auto` is also available.

Open http://127.0.0.1:8000 after starting the GUI. It is a local, single-client tool. Use `--conf` in either GUI command for a nondefault model configuration.

**The bundled `weights/c_1048.pkl` is a historical heads-up TensorFlow checkpoint. It is incompatible with the six-player PyTorch policy. Loading it fails clearly instead of padding layers or silently using random weights. Train a fresh six-player model.** A successful smoke checkpoint demonstrates execution, not strong poker play or convergence.

## Deterministic decisions and sampling

```python
from agi.evaluation_tools import NNAgent
agent = NNAgent(env.observation_space, env.action_space, conf, weights_path)
action = agent.make_action(obs, deterministic=True)
# Default samples the masked policy. Higher temperature makes it less concentrated.
action = agent.make_action(obs, temperature=2.0)
model_deci = agent.model_deci
action = model_deci(history, board, deterministic=True)
```

`history` is four street lists; every entry is `(seat_id, action_id, legal_action_ids)`. Preserve every opponent's fold and seat ID. `board` is the complete decision state:

```python
board = {
    'hand': ['SA', 'HK'],          # Only acting player's private cards
    'public_cards': [],           # Flop/turn/river cards, suit then rank
    'stakes': [200, 199, 198, 200, 200, 200],
    'contributions': [0, 1, 2, 0, 0, 0],
    'statuses': ['alive'] * 6,     # alive, folded, or allin
    'current_player': 3,
    'dealer_id': 0,
    'legal_moves': [0, 1, 2, 3, 4], # Supply legal IDs from the engine
}
```

Actions: 0 fold, 1 check/call, 2 raise half pot, 3 raise pot, 4 all-in. Pot-size raises now call first and raise by the chosen fraction of the pot after calling. Short calls contribute only available chips. Short all-ins reopen a previous actor only when the cumulative increase reaches a full raise. All-in/folded seats are skipped, and the board runs out when no more betting is possible. Each side pot ranks only its eligible hands; odd chips go clockwise from the button.

Large absolute logits alone do not imply near-deterministic play: differences between legal logits determine probabilities. Sampling uses a stable masked softmax; `deterministic=True` selects the highest-scoring legal action. Stack and reward normalization improve scale for new training, but no inference option can teach an old heads-up checkpoint multiway strategy or guarantee convergence.

## Experimental structured six-player model

`confs/nl_holdem_structured.py` selects `NlHoldemStructuredNet`. It rotates public seat/history rows into acting-player order, adds public pot, street call, short-call cost, raise increment, effective stack, pot odds, SPR and active-opponent features, and uses independent policy and value encoders. This reduces seat-learning redundancy and lets the critic learn without directly changing the policy encoder. Both branches retain the existing convolutional design; the algorithm, learning rate and entropy coefficient are unchanged.

Start a **fresh run** (choose a new output directory):

```sh
python train_league.py --conf confs/nl_holdem_structured.py --device auto --workers 2 --trained-steps 1000000 --output_dir work/structured --experiment_name structured
```

Resume its full optimizer, league and counters with the same model configuration; the step target is cumulative:

```sh
python train_league.py --conf confs/nl_holdem_structured.py --device auto --workers 2 --restore-state work/structured --trained-steps 2000000 --output_dir work/structured_resumed --experiment_name structured_resumed
```

Use `--device cuda`, `--device cpu`, or experimental `--device mps` to choose explicitly. CPU and CUDA inference/backward checks pass; the portable IMPALA update path is also exercised on both. MPS needs verification on actual Apple hardware. Legacy models/configs remain available, but their checkpoints cannot initialize this architecture. Restore rejects changed model, seat semantics, observation shape or reward scaling before starting Ray.

For controlled ablations, set `custom_model_config.relative_seats=False` or `separate_critic=False` in a copied config and start a separate fresh run. `--opponent-sampling uniform` is an optional pool-diversity experiment; ranked sampling remains the default. The earlier pilots trained unequal transition counts, and uniform sampling's pooled confidence interval included zero, so they do not establish an improvement.

For `NNAgent.model_deci` with betting features, also provide `board['street_contributions']` (one public street commitment per seat) and `board['last_raise_amount']` (the engine's last full raise increment). Total hand contributions cannot substitute for street commitments.

Reward-scale, terminal/bootstrap boundaries, actor-relative histories, private-card isolation and separate-critic gradients have regression coverage. Learner logs now include reward mean/maximum, terminal fraction and sampled-value spread. `tools/modal_validate_structured.py` runs a small fresh/resume/evaluation integration check on Modal CPU; choose a unique RUN_ID before repeating it. The completed Modal validation passed 12 model tests, trained from scratch to 600 transitions, resumed to 1,000 cumulative transitions, and kept learner statistics finite. Its six-hand evaluations check execution only. The bounded continuation below improved held-out benchmark profit. An architecture advantage remains unproven: compare fresh legacy and structured runs at equal trained-transition budgets, ideally across multiple training seeds, to isolate that effect.

## Verification and historical material

Run `python -m pytest tests -q` from the repository root. Coverage includes six-player poker rules, side pots, short raises/calls, seed reproducibility, observation/model shapes, legal deterministic inference, incompatible checkpoints, GUI behavior, and 300 seeded unequal-stack hands. Original notebooks and TensorFlow source are historical examples and have not been migrated to the new API. Use the scripts above for current training and evaluation.

# Released data

1. Weights of all checkpoints in the process of a week of training, ~ 1 billion of selfplay games:
    Google Drive:   https://drive.google.com/file/d/1G_GwTaVe4syCwW43DauwSQi6FqjRS3nj/view?usp=sharing
    Baidu  Drive:   https://pan.baidu.com/s/1PYNLKN2CExRntVvkvYyKkA?pwd=7jmn
2. Evaluation metrics and part of results: see ```nl-evaluation.ipynb```
    

# Remaining limitations

Long-run convergence and playing strength require fresh training and evaluation against stronger opponents. The historical TensorFlow checkpoints and notebooks are not six-player models. This uses five discrete actions, fixed per-hand stacks, and a shared opponent policy; it does not model rake, tournament dynamics, arbitrary bet sizing, or independent opponent styles.

# License

[GNU AGPL v3](https://www.gnu.org/licenses/agpl-3.0.en.html)

# Warning

It's illegal to use this code in any way to commercial purpose, including researching project inside a commercial entity.  

Especially for Chinese company JJ world(竞技世界). You'd better look elseware.


## One-hour Modal T4 training

Install the optional Modal CLI (`pip install modal`) and authenticate (`modal token new`). From the repository root:

```sh
modal run --detach tools/modal_t4_one_hour.py
```

The script uploads only training source/configs and bundled game data. It requests one T4, 16 CPU cores, 16 GiB RAM (24 GiB cap), 14 rollout workers and 4,000-transition batches. Runtime math threads are limited to one per worker to prevent Modal's container defaults from oversubscribing CPUs. It trains for one hour, saves full state, logs resource usage and persists results in the `alpha-holdem-budget-results` volume. Held-out evaluation is disabled during this timed run; evaluate the saved initial/final weights afterward. The estimated compute cost is about $1.55–$1.67 per run at the October 2026 published rates, excluding other usage/storage; verify current pricing before launching.

The launcher generates a unique run ID by default. Select a config and explicit unique run ID when needed:

```sh
modal run --detach tools/modal_t4_one_hour.py --conf confs/nl_holdem_structured.py --run-id my-structured-run --checkpoint-steps 50000
```

This starts a fresh model. An automatic Modal preemption restart restores the committed full checkpoint with the same source/config and only the remaining cumulative time; it reserves time for uncommitted progress and restart overhead. Every checkpoint is committed promptly. An unrelated launch cannot overwrite the directory.

If an older launcher failed after preemption, first confirm that its function has stopped, then recover the same run:

```sh
modal run --detach tools/modal_t4_one_hour.py --conf confs/nl_holdem_structured.py --run-id YOUR_RUN_ID --checkpoint-steps 10000 --recover
```

Recovery preserves the original training source and requires a committed full checkpoint. It never restarts a full hour. For an additional one-hour continuation in a new directory, use the previous run's full state and saved configuration:

```sh
modal run --detach tools/modal_t4_one_hour.py --conf confs/nl_holdem_structured.py --run-id NEW_RUN_ID --resume-from PREVIOUS_RUN_ID --checkpoint-steps 10000
```

The learning configuration is loaded from the previous checkpoint directory. Optimizer, counters and league restore together; the new run gets its own one-hour budget and preemption recovery. The launch record is saved to `work/t4_one_hour_launch.json`. Training continues after the local command exits. Retrieve the exact run directory with `modal volume get alpha-holdem-budget-results RUN_ID work/modal_results`. The wrapper has a shutdown watchdog and no retries; logs/checkpoints remain available if the process fails.

## Controlled historical-opponent experiment on Modal

`tools/modal_pool_pilot.py` is a budget-capped experiment configured for the saved first- and second-hour run IDs. Update its run IDs for your own checkpoints before using it. `modal run --detach tools/modal_pool_pilot.py` compares both models against fixed and mixed historical tables, then runs two pilots from the same selected full checkpoint: ranked versus uniform historical sampling. Both keep the pool frozen, self-play probability zero, architecture and learning settings unchanged. Each pilot targets 160,000 additional transitions and stops after at most 300 training seconds; final batches may overshoot and asynchronous runs are not bit-for-bit identical. A fresh seed bank evaluates both pilots against the same pool. These short pilots are diagnostic, not proof of long-term improvement.

Evaluation and report/plot generation run on Modal CPU through `tools/evaluate_policy_pool.py`. Artifacts are stored under `RUN_ID/diagnostic/` and `RUN_ID/pilot_evaluation/` in the results volume; per-pilot logs/checkpoints are under `ranked/` and `uniform/`. Reports include paired confidence intervals and an equal-weight benchmark average. All stages have timeouts and no retries; estimated additional compute is capped around $0.55 at the stated rates. Change RUN_ID before repeating the experiment. For a full second-hour resume followed by evaluation, `tools/modal_resume_hour.py` also supports `--evaluate-only` to recover evaluation independently of training.

## Structured-model plateau diagnostics

`tools/modal_structured_experiment.py` performs a bounded ranked-versus-uniform historical-pool experiment from the saved structured hour-one full checkpoint. Both pilots use a frozen pool, self-play probability zero, unchanged learning settings and a target of 120,000 new transitions. Evaluation selects the largest shared checkpoint count between 80,000 and 120,000 additional transitions, and refuses an unequal-step comparison. It evaluates paired held-out seeds against fixed and mixed tables on Modal CPU, then runs `tools/diagnose_critic.py` to measure value calibration by street, active opponents and pot size. The critic compares predictions to realized terminal returns in training reward units; it does not treat correlated decisions as independent evidence.

Choose new RUN/BASE/SECOND identifiers before repeating a new experiment. Each GPU pilot has a hard 600-second function timeout and subprocess watchdog; retries do not start another paid pilot in an existing directory. Total compute is estimated below $0.60, excluding unrelated account usage. The historical one-seed results are diagnostic and do not establish long-run convergence. Native RLlib IMPALA value loss sums errors over the batch, so use per-decision diagnostics before inferring a reward-scaling problem from large logged losses.

The 2026-10-06 structured-model experiment matched both pilots at 1,404,000 cumulative transitions (+120,000 each). On the six-table benchmark, mean profits were baseline +11.76, ranked +6.75, uniform +11.47 chips/hand. Uniform minus ranked was +4.72 with paired approximate 95% interval [-0.93, +10.37]; uniform minus baseline was -0.29 [-3.05, +2.46]. These results do not justify changing the default or claiming improvement. Critic calibration against one held-out legacy table showed explained variance 0.007 (hour one) and -0.009 (hour two); around 95% of recorded decisions were preflop, so postflop conclusions are not supported. An identical-policy six-seat evaluator sanity check was consistent with zero profit. Opponent refresh probability remains 0.01 (about 100 hands between refreshes on average), and all five opponents share the table's selected checkpoint. Those are diversity constraints to test, not established root causes.

## Learning settings diagnostics

`tools/modal_learning_lab.py --stage probe` tests critic optimization on a fixed on-policy return dataset, splitting by complete hand to avoid target leakage. It fits a tiny subset to test capacity, then checks generalization to held-out hands. `--stage screen` compares an unchanged control, entropy coefficient 0.02, and faster uniform opponent refresh, starting from the structured hour-one full state and selecting equal-step checkpoints. `--stage select` recovers CPU selection independently of paid training. Set a new RUN for a new experiment; completed candidate metadata prevents retraining.

`train_league.py --entropy-coeff 0.02` explicitly resets the learner's entropy schedule after checkpoint restore. This is necessary for intentional experiments because RLlib restores saved policy configuration; the override preserves weights and optimizer state. `--exg_oppo_prob 0.25` refreshes the historical table policy more often, and `--opponent-sampling uniform` changes selection. These are experimental settings, not established new defaults. Carry the intended runtime flags into subsequent runs. Selection seeds are validation data; final performance must be checked on a separate seed bank.


## Verified bounded continuation (2026-10-06)

### Use the included trained model

The verified inference package is checked into [models/six_player_structured_v1](models/six_player_structured_v1). The newest best verified checkpoint is [six_player_best_2026-10-06.pkl](models/six_player_structured_v1/six_player_best_2026-10-06.pkl); `weights.pkl` remains an identical compatibility copy. After installing the environment described above, activate it and run from the repository root:

```bash
python gui/play_against_ai_in_ui.py --conf confs/nl_holdem_structured.py --weights models/six_player_structured_v1/six_player_best_2026-10-06.pkl --device cpu
```

Open `http://127.0.0.1:8000` to play at a six-player table. Windows can use `.venv\Scripts\python.exe`; macOS can use `.venv/bin/python`. CPU is sufficient for individual decisions. Use `--device cuda` for NVIDIA CUDA, or `--device mps` for experimental Apple Silicon inference. Add `--deterministic` to always choose the highest-probability legal action. The reported benchmarks used sampling, so deterministic play can perform differently.

For Python integration:

```python
import json
from pathlib import Path
from agi.nl_holdem_env import NlHoldemEnvWrapper
from agi.evaluation_tools import NNAgent

folder = Path("models/six_player_structured_v1")
conf = json.loads((folder / "training_config.json").read_text())
env = NlHoldemEnvWrapper(conf)
agent = NNAgent(env.observation_space, env.action_space, conf,
                folder / "six_player_best_2026-10-06.pkl", device="cpu", seed=42)
obs, _ = env.reset(seed=42)
action = agent.make_action(obs)  # Decision for env.my_agent(), the current seat
obs, payoffs, done, _, info = env.step(action)
env.close()
```

Action IDs: `0` fold, `1` check/call, `2` raise half-pot, `3` raise pot, `4` all-in. Use the wrapper's observation and legal-action mask; do not merge opponents or omit their folds. The config must select `NlHoldemStructuredNet` with six-player betting features. For an external game, `agent.model_deci(history, board, deterministic=True)` accepts four street histories and the full public seat state described in [agi/evaluation_tools.py](agi/evaluation_tools.py), including street contributions and the last raise amount. It needs the acting player's hole cards and public cards, not opponents' hole cards.

The structured candidate continued with entropy 0.02, ranked historical sampling, refresh 0.01 and a frozen pool. Two independent evaluation banks (108,000 total hands) improved pool profit from 11.35 to 18.14 chips/hand; paired gain +6.79 with approximate 95% interval [+3.84, +9.73]. Profit intervals were positive on all five fixed tables, including trained legacy and mixed historical opponents. This measures one training seed and a complete continuation recipe; it does not isolate an architecture or entropy advantage, or establish optimal poker play. See [the concise change/test log](docs/model_improvement_log.md).

Final run: `six-player-20261006-lowentropy-final-1h`. Its `final_model/` folder in Modal volume `alpha-holdem-budget-results` contains inference weights, matching config and the benchmark report; full optimizer/league state stays under `league/`.

To continue this checkpoint for another bounded hour (incurs new Modal charges), use a unique run ID and preserve its runtime settings:

```bash
modal run --detach tools/modal_t4_one_hour.py --conf confs/nl_holdem_structured.py --run-id YOUR_NEW_RUN_ID --resume-from six-player-20261006-lowentropy-final-1h --opponent-sampling ranked --opponent-refresh 0.01 --entropy-coeff 0.02 --freeze-opponents --checkpoint-steps 100000
```

For a fresh evaluation bank on Modal (choose a new seed):

```bash
modal run --detach tools/modal_evaluate_structured.py --run-id YOUR_NEW_RUN_ID --previous-run six-player-20261005-structured-t4-1h --hands 3600 --seed 31000000 --historical-mix
```
