"""Paired six-seat evaluation and report, runnable inside a Modal CPU container."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def evaluate_opponent(task):
    opponent, run_dir, previous_dir, hands = task
    import numpy as np
    import torch
    from agi.nl_holdem_env import NlHoldemEnvWrapper
    from agi.evaluation_tools import NNAgent
    torch.set_num_threads(1)
    config = json.loads((Path(run_dir)/'league/training_config.json').read_text())
    env = NlHoldemEnvWrapper(config)
    paths = {'initial': Path(run_dir)/'league/initial_weight.pkl',
             'previous': Path(previous_dir)/'league/output_weight.pkl',
             'final': Path(run_dir)/'league/output_weight.pkl'}
    agents = {name: NNAgent(env.observation_space, env.action_space, config, path, device='cpu')
              for name,path in paths.items()}
    # Separate opponent instance ensures hero/opponent streams never interfere.
    frozen = NNAgent(env.observation_space,env.action_space,config,paths['initial'],device='cpu')
    results=[]
    for name in paths:
        rewards=[]; seat_rewards=[[] for _ in range(6)]; actions=np.zeros(5,dtype=int)
        for hand in range(hands):
            hero=hand%6
            obs,_=env.reset(seed=2100000+hand)
            rng=np.random.default_rng(2200000+hand)
            agents[name].rng=np.random.default_rng(2300000+hand)
            frozen.rng=np.random.default_rng(2400000+hand)
            done=False
            while not done:
                if env.my_agent()==hero:
                    action=agents[name].make_action(obs);actions[action]+=1
                elif opponent=='initial': action=frozen.make_action(obs)
                elif opponent=='check_call':
                    action=1 if obs['legal_moves'][1] else int(np.flatnonzero(obs['legal_moves'])[0])
                else: action=int(rng.choice(np.flatnonzero(obs['legal_moves'])))
                assert obs['legal_moves'][action]>0
                obs,payoff,done,_,_=env.step(action)
            assert abs(sum(payoff))<1e-6
            rewards.append(float(payoff[hero]));seat_rewards[hero].append(float(payoff[hero]))
        mean=float(np.mean(rewards));radius=1.96*float(np.std(rewards,ddof=1))/np.sqrt(hands)
        result={'model':name,'opponent':opponent,'hands':hands,'chips_per_hand':mean,
                'approx_95pct_ci':[mean-radius,mean+radius],
                'seat_mean_chips':[float(np.mean(x)) for x in seat_rewards],
                'action_counts_fold_call_halfpot_pot_allin':actions.tolist(),'rewards':rewards}
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='rewards'}),flush=True)
    env.close()
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--previous-dir',type=Path,required=True)
    parser.add_argument('--hands',type=int,default=2400)
    parser.add_argument('--jobs',type=int,default=3)
    args=parser.parse_args()
    if args.hands<6 or args.hands%6 or args.jobs<1: parser.error('hands must be a positive multiple of six; jobs positive')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[key]='1'
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    import numpy as np
    tasks=[(o,str(args.run_dir),str(args.previous_dir),args.hands) for o in ('random','check_call','initial')]
    with ProcessPoolExecutor(max_workers=min(args.jobs,3),mp_context=multiprocessing.get_context('spawn')) as pool:
        evaluations=[r for batch in pool.map(evaluate_opponent,tasks) for r in batch]
    comparisons=[]
    for opponent in ('random','check_call','initial'):
        selected={r['model']:r for r in evaluations if r['opponent']==opponent}
        for baseline in ('initial','previous'):
            delta=np.asarray(selected['final']['rewards'])-np.asarray(selected[baseline]['rewards'])
            mean=float(delta.mean());radius=1.96*float(delta.std(ddof=1))/np.sqrt(len(delta))
            comparisons.append({'opponent':opponent,'baseline':baseline,'paired_change_chips_per_hand':mean,
                                'approx_95pct_ci':[mean-radius,mean+radius]})
    rows=[json.loads(x) for x in (args.run_dir/'ray_results/t4_one_hour/result.json').read_text().splitlines()]
    import pickle
    with (args.run_dir/'league/output_weight.pkl').open('rb') as f:weights=pickle.load(f)
    health={'parameters':sum(v.size for v in weights.values()),'all_weights_finite':all(bool(np.isfinite(v).all()) for v in weights.values())}
    summary={'evaluation':evaluations,'paired_comparisons':comparisons,'health':health,
             'metadata':json.loads((args.run_dir/'run_metadata.json').read_text()),
             'final_trained_transitions':rows[-1]['num_env_steps_trained'],
             'final_optimizer_updates':rows[-1]['info']['learner']['default_policy']['num_grad_updates_lifetime']}
    output=args.run_dir/'analysis';output.mkdir(exist_ok=True)
    (output/'evaluation.json').write_text(json.dumps(summary,indent=2))
    report='# Second-hour six-player performance\n\n'
    report+=f"Evaluated on Modal CPU: {len(evaluations)*args.hands:,} held-out hands. Equal hero-seat rotation, identical fresh seeds and action streams across models. Opponents are weak fixed baselines; results do not measure exploitability or strong poker.\n\n"
    report+='| Opponent | Initial | First-hour model | Second-hour model |\n|---|---:|---:|---:|\n'
    for opponent in ('random','check_call','initial'):
        selected={r['model']:r for r in evaluations if r['opponent']==opponent}
        values=[selected[n]['chips_per_hand'] for n in ('initial','previous','final')]
        report+=f"| {opponent} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} |\n"
    report+='\nValues are chips/hand. Approximate paired normal 95% confidence intervals:\n\n'
    for c in comparisons:
        lo,hi=c['approx_95pct_ci']
        report+=f"- Final minus {c['baseline']} vs {c['opponent']}: {c['paired_change_chips_per_hand']:+.2f} [{lo:+.2f}, {hi:+.2f}] chips/hand.\n"
    laststats=rows[-1]['info']['learner']['default_policy']['learner_stats']
    report+=f"\nFinal counters: {summary['final_trained_transitions']:,} cumulative transitions and {summary['final_optimizer_updates']:.0f} cumulative optimizer updates. Final value explained variance: {laststats.get('vf_explained_var',float('nan')):.4f}. All weight tensors finite: {health['all_weights_finite']}.\n"
    gains=[c for c in comparisons if c['baseline']=='previous']
    if all(c['approx_95pct_ci'][0]>0 for c in gains): verdict='Second-hour improvement is supported against all three tested baselines. Further bounded training is reasonable.'
    elif any(c['approx_95pct_ci'][0]>0 for c in gains): verdict='Second-hour improvement is supported against some baselines, but is mixed or uncertain elsewhere. Inspect individual confidence intervals before extending training.'
    elif any(c['approx_95pct_ci'][1]<0 for c in gains): verdict='At least one baseline shows regression. Investigate before extending training.'
    else: verdict='No statistically clear additional gain is established. Larger evaluation or a controlled training change is preferable to assuming continued improvement.'
    report+='\n'+verdict+'\n'
    (output/'report.md').write_text(report,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    steps=[r['num_env_steps_trained']/1e6 for r in rows]
    axes[0].plot(steps,[r['info']['learner']['default_policy']['learner_stats'].get('vf_explained_var',np.nan) for r in rows])
    axes[0].set(title='Resumed value learning',xlabel='Cumulative transitions (millions)',ylabel='Explained variance')
    means=[c['paired_change_chips_per_hand'] for c in gains]
    error=[[means[i]-gains[i]['approx_95pct_ci'][0] for i in range(3)],[gains[i]['approx_95pct_ci'][1]-means[i] for i in range(3)]]
    axes[1].errorbar(range(3),means,yerr=error,fmt='o',capsize=5);axes[1].axhline(0,color='grey',linestyle='--')
    axes[1].set(xticks=range(3),xticklabels=['Random','Check/call','Initial'],title='Second hour minus first hour (95% CI)',ylabel='Chips / hand')
    for a in axes:a.grid(alpha=.2)
    fig.savefig(output/'results.png',dpi=160)
    print(report,flush=True)

if __name__=='__main__':main()
