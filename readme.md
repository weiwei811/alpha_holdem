# Alpha NL Holdem — six-player PyTorch training

This checkout uses 64-bit Python 3.11, PyTorch 2.8.0 (2.2.2 for Intel macOS), and Ray RLlib 2.49.2. The bundled RLCard engine is used directly; do not install another `rlcard` over it.

## Setup: Windows CUDA, Windows CPU, and macOS

Run setup commands from the repository root. Use **64-bit Python 3.11**. The scripts create a local `.venv`; activation is optional because the examples call its Python directly.

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

Use `--device cpu` for training even if a CUDA wheel is installed. `requirements-lock.txt` is the earlier Windows CPU dependency snapshot; it does not select a CUDA wheel.

### macOS: Apple Silicon or Intel

Install Python 3.11 and run:

```sh
sh setup_env.sh
.venv/bin/python -c 'import torch; print(torch.__version__); print("MPS available:", torch.backends.mps.is_available())'
```

The script detects `arm64` or `x86_64` and selects the matching Mac dependency lock. Ray's wheels target macOS 12 or newer; Metal also requires a compatible macOS/PyTorch/hardware combination. Run an arm64 Python on Apple Silicon to use the current native PyTorch build.

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
| `requirements-lock.txt` | Earlier Windows CPU package snapshot. |
| `requirements-mac-arm64.txt` | Resolved Apple Silicon package versions; selected by Mac setup. |
| `requirements-mac-intel.txt` | Resolved Intel Mac package versions; selected by Mac setup. |

The platform locks pin transitive dependencies for repeatable setup. When changing direct dependencies, regenerate the appropriate locks; do not copy the Windows CUDA lock onto another platform. The bundled RLCard engine is used from this repository and must not be replaced with a separate pip installation.

Run the suite with the matching environment:

```powershell
# Windows
.\.venv\Scripts\python.exe -m pytest tests -q
```

```sh
# Mac
.venv/bin/python -m pytest tests -q
```

Windows CPU/CUDA verification passed **32 tests**, with **2 MPS hardware tests skipped**.

## Train six players

```powershell
.\.venv\Scripts\python.exe train_league.py --conf confs/nl_holdem.py --workers 2 --gap 500
```

Without `--iterations`, training continues until stopped. A bounded verification run is:

```powershell
.\.venv\Scripts\python.exe train_league.py --workers 1 --batch-size 100 --iterations 1 --gap 1 --output_dir league/quick_check
```

Defaults use six seats, 200 chips (100 big blinds), automatic learner-device selection, and two CPU rollout workers. The learner controls a randomly selected seat and the other five seats use a shared historical policy. All seats become learner seats across episodes. Opponents are selected before the first action. League profit includes hands ending before a learner decision (such as blind walks); those hands produce no learner training transition. This is a six-player self-play environment, with one learner trajectory per hand; it is not a six-policy multi-agent RLlib environment.

The four street histories keep separate seat IDs and include folds. The most recent `history_len` actions per street are retained; persistent seat status remains visible after history truncation. Observations also contain remaining stacks, contributions, alive/all-in status, button, and acting seat. Card and history tensors are binary floats; chip counts do not wrap at 255. The models derive dimensions from the observation space.

Learner rewards are net chips divided by `reward_scale` (default 200), multiplied by `rwd_ratio` (default 1). League statistics remain raw mean chip profit per hand, despite the historical `winrate` field name. Big blind is 2 chips, so mbb/hand is mean chip profit multiplied by 500. `--upwin` is a mean-chip-profit threshold, not a probability of winning.

Resume from a matching newly trained league:

```powershell
.\.venv\Scripts\python.exe train_league.py --restore league/history_agents --workers 2
```

Snapshots are under `OUTPUT_DIR/weights/c_N.pkl`. Every completed iteration atomically saves the latest learner to `OUTPUT_DIR/output_weight.pkl`; bounded runs also save `training_config.json`. Logs are inside `work/ray_results/`. Restore prefers `OUTPUT_DIR/output_weight.pkl` for the learner and falls back to the latest `c_N.pkl` for older runs. Historical `c_N.pkl` files remain the opponent pool. These weight snapshots resume policies and league rewards; they do not restore optimizer state, RNG state, or the exact training iteration.

## Select a training device

### Windows CUDA

```powershell
.\.venv\Scripts\python.exe train_league.py --device cuda --workers 2 --gap 500
```

### macOS CPU or experimental Apple Silicon GPU

```sh
# CPU
.venv/bin/python train_league.py --device cpu --workers 2 --gap 500
# Metal/MPS, experimental
.venv/bin/python train_league.py --device mps --workers 2 --gap 500
```

`--device auto` prioritizes CUDA, then MPS, then CPU. Use `--device cpu` to force CPU. An explicitly requested unavailable device fails clearly. The earlier `--gpus 1` remains a CUDA alias; `--gpus 0` forces CPU when device is auto. This entry point supports **one learner GPU**. Rollout workers and opponent policies remain on CPU.

MPS training uses a single-device adapter around the pinned old RLlib policy stack. It moves the learner and batches to Metal and bypasses RLlib's CUDA-only gradient context. The adapter's update path was tested against real IMPALA/V-trace loss on CPU and CUDA, but **actual Metal training remains experimental and unverified on Mac hardware**. Use CPU if an MPS operation is unsupported. Intel CPU is the conservative path; AMD-backed MPS availability depends on the installed PyTorch/macOS/hardware combination.

For a bounded GPU check on either platform, add `--batch-size 200 --iterations 1 --gap 1 --output_dir league/gpu_check`. Without `--iterations`, training runs continuously. On a 6 GB GPU, start with the default model and batch size before increasing memory use. Model-update speed improved in the local CUDA benchmark, but CPU game simulation can limit total training throughput.

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

