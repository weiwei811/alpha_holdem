import ray
import logging
import argparse
import ast
from utils import get_winrate_and_weight, save_learner_weights
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import numpy as np
import pickle
import os
import json
import uuid
import signal
import time
from pathlib import Path
from agi.training_progress import StepInterval, validate_batches, evaluate_weights

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env

from agi.nl_holdem_env import NlHoldemEnvWithOpponent
from agi.nl_holdem_net import NlHoldemNet
from agi.nl_holdem_lg_net import NlHoldemLgNet

ModelCatalog.register_custom_model('NlHoldemNet', NlHoldemNet)
ModelCatalog.register_custom_model('NlHoldemLgNet', NlHoldemLgNet)

from agi.league import League

def main():
    parser = argparse.ArgumentParser(description="Train AlphaNLHoldem via League Training.")
    parser.add_argument('--conf',  type=str, default='confs/nl_holdem.py', help="Path to the training config file (e.g., confs/nl_holdem.py).")
    parser.add_argument('--gap', type=int, default=None, help='Legacy interval: converted to gap * train_batch_size trained transitions.')
    parser.add_argument('--sp',  type=float, default=0.0, help="Probability of playing against the agent's own current policy (self-play).")
    parser.add_argument('--exg_oppo_prob',  type=float, default=0.01, help="Probability of exchanging/sampling a new opponent after an episode.")
    parser.add_argument('--upwin',  type=float, default=1.0, help="Legacy argument retained for compatibility; progress-based saves do not use a winrate threshold.")
    parser.add_argument('--kbest',  type=int, default=5, help="Number of best historical agents to consider in the league evaluations.")
    parser.add_argument('--league_tracker_n',  type=int, default=10000, help="Number of games to track in the league statistics.")
    parser.add_argument('--last_num',  type=int, default=100000, help="The number of latest matches to consider when evaluating win rates.")
    parser.add_argument('--rwd_update_ratio',  type=float, default=1.0, help="Ratio for updating results (rewards) to the league tracker.")
    parser.add_argument('--restore',  type=str, default=None, help="Directory to restore training from (e.g., league/history_agents).")
    parser.add_argument('--output_dir',  type=str, default="league/history_agents", help="Directory where historical agents will be saved.")
    parser.add_argument('--mode', type=str, default="local", help="Ray cluster mode to use, usually 'local'.")
    parser.add_argument('--experiment_name', default='run_trial_1', type=str, help="Name of the experiment run.")
    parser.add_argument('--iterations', type=int, default=0, help='Stop after N iterations; 0 trains continuously.')
    parser.add_argument('--workers', type=int, default=None, help='Override number of rollout workers.')
    parser.add_argument('--gpus', type=int, default=None, help='Legacy CUDA count: 0 or 1.')
    parser.add_argument('--device', choices=['auto','cpu','cuda','mps'], default='auto')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--checkpoint-steps', type=int, default=10000, help='Trained transitions between league and full-state checkpoints; 0 disables periodic saves.')
    parser.add_argument('--eval-steps', type=int, default=50000, help='Trained transitions between held-out evaluations; 0 disables.')
    parser.add_argument('--eval-hands', type=int, default=600)
    parser.add_argument('--trained-steps', type=int, default=0, help='Absolute trained-transition target; 0 has no target.')
    parser.add_argument('--metrics-timeout', type=float, default=1.0)
    parser.add_argument('--freeze-opponents', action='store_true', help='Keep the initial historical pool fixed for controlled comparisons.')
    parser.add_argument('--restore-state', type=str, help='Run directory containing latest_checkpoint.json; restores RLlib optimizer and counters.')
    parser.add_argument('--training-seconds', type=float, default=0, help='Stop gracefully after this many training seconds; 0 is unlimited.')
    args = parser.parse_args()
    if not np.isfinite(args.training_seconds) or args.training_seconds < 0:
        parser.error('training-seconds must be finite and nonnegative')
    if min(args.checkpoint_steps, args.eval_steps, args.trained_steps) < 0 or args.eval_hands < 2 or args.metrics_timeout < 0:
        parser.error('Step intervals/target and timeout must be nonnegative; eval-hands must be at least 2')
    if args.restore and args.restore_state:
        parser.error('Choose --restore (weights only) or --restore-state')

    if (args.gap is not None and args.gap < 1) or args.iterations < 0 or args.league_tracker_n < 1 or args.kbest < 1:
        parser.error('gap, league_tracker_n and kbest must be positive; iterations must be nonnegative')
    for probability in (args.sp, args.exg_oppo_prob, args.rwd_update_ratio):
        if not 0 <= probability <= 1:
            parser.error('Probabilities must be between 0 and 1')
    conf = ast.literal_eval(open(args.conf).read())
    if args.workers is not None:
        conf['num_env_runners'] = args.workers
    if args.batch_size is not None:
        conf['train_batch_size'] = args.batch_size
        conf['minibatch_size'] = min(args.batch_size, conf.get('minibatch_size', args.batch_size))

    if args.gap is not None:
        args.checkpoint_steps = args.gap * conf.get('train_batch_size', 4000)
    conf['metrics_episode_collection_timeout_s'] = args.metrics_timeout
    try:
        validate_batches(conf)
    except ValueError as error:
        parser.error(str(error))

    from agi.devices import configure_training_device
    try:
        device = configure_training_device(conf, args.device, args.gpus)
    except ValueError as error:
        parser.error(str(error))
    print('Learner device: {}'.format(device), flush=True)
    if device.type == 'mps':
        print('MPS training is experimental; use --device cpu if an operation is unsupported.', flush=True)
    if args.mode == 'local':
        ray.init(include_dashboard=False)
    else:
        parser.error('Only local mode is supported')

    register_env("NlHoldemEnvWithOpponent", lambda config: NlHoldemEnvWithOpponent(
            conf
    ))

    league = League.remote(
        n=args.league_tracker_n,
        last_num=args.last_num,
        kbest=args.kbest,
        output_dir=args.output_dir,
    )

    class LeagueCallbacks(DefaultCallbacks):
        def __init__(self):
            super().__init__()
            self.count = 0

        def on_sub_environment_created(self, *, worker, sub_environment, **kwargs):
            def prepare_opponent(env):
                if env.oppo_name is not None:
                    return
                # Invoked by reset after worker policies exist, before any actions.
                if not ray.get(league.initized.remote()):
                    weight = worker.get_policy("default_policy").get_weights()
                    ray.get(league.initize_if_possible.remote(weight))
                pid, weight = ray.get(league.select_opponent.remote())
                env.oppo_policy.set_weights(weight)
                env.oppo_name = pid

            def record_hand(env, reward):
                if np.random.random() < args.rwd_update_ratio:
                    ray.get(league.update_result.remote(
                        None if env.oppo_name == "self" else env.oppo_name,
                        reward, selfplay=env.oppo_name == "self"))

            sub_environment.prepare_opponent = prepare_opponent
            sub_environment.on_hand_end = record_hand

        def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
            pass

        def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
            default_policy = policies["default_policy"]
            for env in [base_env.get_sub_environments()[env_index]]:
                if hasattr(env, "is_done") and env.is_done:
                    # 2. 更新对手权重
                    if np.random.random() < args.exg_oppo_prob:
                        if np.random.random() < args.sp:
                            p_weights = default_policy.get_weights()
                            weight = {}
                            for k,v in p_weights.items():
                                k = k.replace("default_policy","oppo_policy")
                                weight[k] = v
                            env.oppo_name = "self"
                            if hasattr(env, "oppo_policy"):
                                env.oppo_policy.set_weights(weight)
                        else:
                            pid, weight = ray.get(league.select_opponent.remote())
                            env.oppo_name = pid
                            if hasattr(env, "oppo_policy"):
                                env.oppo_policy.set_weights(weight)

        def on_train_result(self, *, algorithm, result, **kwargs):
            save_learner_weights(args.output_dir, algorithm.get_policy().get_weights())
            winrates_pd = ray.get(league.get_statics_table.remote())
            winrates_pd.to_csv("winrates.csv", header=False, index=False)

            table_t = winrates_pd.T
            table_t["mbb/h"] = np.asarray(table_t["winrate"] / 2.0 * 1000.0, dtype=int)

            # Print the table fully to the terminal to bypass Ray Tune's truncation
            print("\n" + "="*50)
            print("CURRENT LEAGUE WINRATES:")
            print(table_t.T.to_string())
            print("="*50 + "\n")

            result['league_mean_chip_profit'] = {str(k): float(v) for k,v in zip(table_t['oppo'], table_t['winrate'])}
            self.count += 1

    from ray.rllib.algorithms.impala import ImpalaConfig
    from agi.portable_impala import PortableImpala
    config = (ImpalaConfig(algo_class=PortableImpala).update_from_dict(conf)
              .api_stack(enable_rl_module_and_learner=False,
                         enable_env_runner_and_connector_v2=False)
              .callbacks(LeagueCallbacks))
    restore_weights = None
    restore_directory = args.restore_state or args.restore
    manifest = None
    if args.restore_state:
        manifest = json.loads((Path(args.restore_state) / 'latest_checkpoint.json').read_text())
        league_state = Path(args.restore_state) / manifest['path'] / 'league_state.pkl'
        if league_state.exists():
            with league_state.open('rb') as stream:
                ray.get(league.restore_state.remote(pickle.load(stream)))
        else:
            restore_weights = get_winrate_and_weight(restore_directory, league)
    elif args.restore:
        restore_weights = get_winrate_and_weight(restore_directory, league)
    agent = None
    try:
        logdir = os.path.abspath(os.path.join('work', 'ray_results', args.experiment_name))
        os.makedirs(logdir, exist_ok=True)
        agent = config.build_algo(logger_creator=lambda config: UnifiedLogger(config, logdir, loggers=None))
        actual_device = next(agent.get_policy().model.parameters()).device
        if actual_device.type != device.type:
            raise RuntimeError('Requested {} but learner is on {}'.format(device, actual_device))
        print('Verified learner model on {}'.format(actual_device), flush=True)
        if args.restore_state:
            agent.restore(str((Path(args.restore_state) / manifest['path']).resolve()))
            agent.env_runner_group.sync_weights()
            print('Restored full RLlib training state', flush=True)
        elif restore_weights is not None:
            agent.get_policy().set_weights(restore_weights)
            agent.env_runner_group.sync_weights()
            print("Restored learner weights from " + args.restore, flush=True)
        # Capture the actual learner baseline, not an independently initialized worker.
        if restore_directory is None:
            initial_weights = agent.get_policy().get_weights()
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, 'initial_weight.pkl'), 'wb') as stream:
                pickle.dump(initial_weights, stream)
            ray.get(league.initize_if_possible.remote(initial_weights))
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        baseline = output / 'initial_weight.pkl'
        if restore_directory and not baseline.exists():
            import shutil
            source = Path(restore_directory) / 'initial_weight.pkl'
            if source.exists():
                shutil.copy2(source, baseline)
        if args.eval_steps and not baseline.exists():
            raise ValueError('Periodic evaluation needs initial_weight.pkl from the original run')
        steps = int(manifest['trained_steps']) if args.restore_state else 0
        checkpoint_interval = StepInterval(args.checkpoint_steps, steps)
        evaluation_interval = StepInterval(args.eval_steps, steps)

        def save_state():
            relative = 'checkpoints/step_{}_{}'.format(steps, uuid.uuid4().hex[:8])
            destination = output / relative
            destination.mkdir(parents=True, exist_ok=True)
            # The local callback closes over this run's Ray actor. Persist a
            # plain callback class, so checkpoints never deserialize a dead actor.
            original_config = agent.config
            policy = agent.get_policy()
            original_policy_config = policy.config
            agent.config = original_config.copy(copy_frozen=False)
            agent.config.callbacks_class = DefaultCallbacks
            policy.config = dict(original_policy_config)
            policy.config['callbacks'] = DefaultCallbacks
            policy.config['callbacks_class'] = DefaultCallbacks
            try:
                agent.save_checkpoint(str(destination.resolve()))
            finally:
                agent.config = original_config
                policy.config = original_policy_config
            with (destination / 'league_state.pkl').open('wb') as stream:
                pickle.dump(ray.get(league.export_state.remote()), stream)
            temporary = output / 'latest_checkpoint.json.tmp'
            temporary.write_text(json.dumps({'path': relative, 'trained_steps': steps}))
            temporary.replace(output / 'latest_checkpoint.json')

        stop_requested = False
        def request_stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True
            print('Stop requested; finishing current iteration and saving full state', flush=True)
        previous_term = signal.signal(signal.SIGTERM, request_stop)
        previous_int = signal.signal(signal.SIGINT, request_stop)
        training_started = time.monotonic()
        iteration = 0
        while not stop_requested and (not args.training_seconds or time.monotonic() - training_started < args.training_seconds) and (args.iterations == 0 or iteration < args.iterations) and (not args.trained_steps or steps < args.trained_steps):
            result = agent.train()
            iteration += 1
            steps = int(result.get('num_env_steps_trained', 0))
            updates = result.get('info', {}).get('learner', {}).get('default_policy', {}).get('num_grad_updates_lifetime', 0)
            print('Trained transitions: {}; optimizer updates: {}'.format(steps, updates), flush=True)
            if checkpoint_interval.due(steps):
                if not args.freeze_opponents:
                    ray.get(league.add_weight.remote(agent.get_policy().get_weights()))
                save_state()
            if evaluation_interval.due(steps):
                evaluation = evaluate_weights(conf, output / 'output_weight.pkl', baseline, args.eval_hands)
                with (output / 'evaluation.jsonl').open('a') as stream:
                    stream.write(json.dumps({'trained_steps': steps, 'optimizer_updates': updates, 'results': evaluation}) + '\n')
                print('Held-out evaluation: ' + json.dumps(evaluation), flush=True)
            if args.trained_steps and steps >= args.trained_steps:
                break
            print('Training iteration {} completed'.format(iteration), flush=True)
        os.makedirs(args.output_dir, exist_ok=True)
        save_learner_weights(args.output_dir, agent.get_policy().get_weights())
        save_state()
        with open(os.path.join(args.output_dir, 'training_config.json'), 'w') as stream:
            json.dump(conf, stream, indent=2)
        ray.get(league.get_statics_table.remote())
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        print('Checkpoint saved to ' + os.path.join(args.output_dir, 'output_weight.pkl'))
    finally:
        if agent is not None:
            agent.stop()
        ray.shutdown()


if __name__ == '__main__':
    main()
