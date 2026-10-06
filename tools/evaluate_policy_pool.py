"""Fixed six-player policy-pool benchmark with paired hero seed streams."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def evaluate_case(task):
    case,spec=task
    import numpy as np, torch
    from agi.evaluation_tools import NNAgent
    from agi.nl_holdem_env import NlHoldemEnvWrapper
    torch.set_num_threads(1)
    env=NlHoldemEnvWrapper(spec['config'])
    heroes={n:NNAgent(env.observation_space,env.action_space,spec.get('model_configs',{}).get(n,spec['config']),p,device='cpu') for n,p in spec['models'].items()}
    opponents={p:NNAgent(env.observation_space,env.action_space,case.get('config',spec['config']),p,device='cpu') for p in case.get('weights',[])}
    results=[]
    for name,hero_agent in heroes.items():
        rewards=[];actions=np.zeros(5,dtype=int)
        for hand in range(spec['hands']):
            hero=hand%6;seed=spec['seed']+hand
            obs,_=env.reset(seed=seed)
            hero_agent.rng=np.random.default_rng(seed+100000)
            rng=np.random.default_rng(seed+200000)
            seat_agents={}
            for seat in range(6):
                if opponents:
                    path=case['weights'][int(rng.integers(len(case['weights'])))]
                    # Restore the dedicated per-seat stream before each decision.
                    agent=opponents[path]
                    seat_agents[seat]=(agent,np.random.default_rng(seed+300000+seat*10000))
            done=False
            while not done:
                actor=env.my_agent()
                if actor==hero:action=hero_agent.make_action(obs);actions[action]+=1
                elif opponents:
                    agent,stream=seat_agents[actor];agent.rng=stream;action=agent.make_action(obs)
                elif case['name']=='check_call': action=1 if obs['legal_moves'][1] else int(np.flatnonzero(obs['legal_moves'])[0])
                else:action=int(rng.choice(np.flatnonzero(obs['legal_moves'])))
                assert obs['legal_moves'][action]>0
                obs,payoff,done,_,_=env.step(action)
            assert abs(sum(payoff))<1e-6
            rewards.append(float(payoff[hero]))
        mean=float(np.mean(rewards));radius=1.96*float(np.std(rewards,ddof=1))/np.sqrt(spec['hands'])
        result={'model':name,'opponent':case['name'],'chips_per_hand':mean,'approx_95pct_ci':[mean-radius,mean+radius],
                'rewards':rewards,'action_counts':actions.tolist()}
        results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='rewards'}),flush=True)
    env.close();return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);args=parser.parse_args()
    spec=json.loads(args.spec.read_text())
    if spec['hands']<6 or spec['hands']%6:parser.error('hands must be a positive multiple of six')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[key]='1'
    import numpy as np,multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=[row for group in pool.map(evaluate_case,[(case,spec) for case in spec['opponents']]) for row in group]
    reference=spec['reference'];comparisons=[]
    names=list(spec['models'])
    for case in spec['opponents']:
        r={row['model']:row for row in results if row['opponent']==case['name']}
        for name in names:
            if name==reference:continue
            delta=np.array(r[name]['rewards'])-np.array(r[reference]['rewards'])
            mean=float(delta.mean());radius=1.96*float(delta.std(ddof=1))/np.sqrt(len(delta))
            comparisons.append({'model':name,'reference':reference,'opponent':case['name'],'paired_change':mean,'approx_95pct_ci':[mean-radius,mean+radius]})
    aggregate={}
    for name in names:
        by_case=[np.array(r['rewards']) for r in results if r['model']==name]
        per_seed=np.mean(by_case,axis=0);aggregate[name]={'mean_chips_per_hand':float(per_seed.mean())}
        if name!=reference:
            ref=np.mean([r['rewards'] for r in results if r['model']==reference],axis=0)
            delta=per_seed-ref;mean=float(delta.mean());radius=1.96*float(delta.std(ddof=1))/np.sqrt(len(delta))
            aggregate[name].update(paired_change=mean,approx_95pct_ci=[mean-radius,mean+radius])
    summary={'spec':spec,'results':results,'paired_comparisons':comparisons,'aggregate':aggregate}
    output=Path(spec['output']);output.mkdir(parents=True,exist_ok=True)
    (output/'evaluation.json').write_text(json.dumps(summary,indent=2))
    report='# Fixed opponent-pool evaluation\n\n'
    report+=f"Computed on Modal CPU. {spec['hands']:,} hands per model/opponent pairing; six-seat rotation; fresh paired deal/action streams. Equal-weight pool average treats shared seed outcomes together when calculating its interval. These are benchmark profits, not exploitability estimates.\n\n"
    report+='| Opponent | '+' | '.join(names)+' |\n|---|'+ '|'.join(['---:']*len(names))+'|\n'
    for case in spec['opponents']:
        values={r['model']:r['chips_per_hand'] for r in results if r['opponent']==case['name']}
        report+='| '+case['name']+' | '+' | '.join(f"{values[n]:+.2f}" for n in names)+' |\n'
    report+='\nPaired changes relative to '+reference+':\n\n'
    for c in comparisons:
        lo,hi=c['approx_95pct_ci'];report+=f"- {c['model']} vs {c['opponent']}: {c['paired_change']:+.2f} [{lo:+.2f}, {hi:+.2f}] chips/hand.\n"
    report+='\nEqual-weight benchmark pool average:\n\n'
    for name,a in aggregate.items():
        report+=f"- {name}: {a['mean_chips_per_hand']:+.2f} chips/hand"
        if name!=reference:
            lo,hi=a['approx_95pct_ci'];report+=f"; paired change {a['paired_change']:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        report+='\n'
    report+='\nSingle-seed-bank results and multiple comparisons need replication. Short pilot training runs do not establish long-term convergence.\n'
    (output/'report.md').write_text(report,encoding='utf-8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for i,name in enumerate(names):
        selected=[r for r in results if r['model']==name];x=np.arange(len(selected))+(i-(len(names)-1)/2)*.18
        means=[r['chips_per_hand'] for r in selected];errors=[[means[j]-selected[j]['approx_95pct_ci'][0] for j in range(len(selected))],[selected[j]['approx_95pct_ci'][1]-means[j] for j in range(len(selected))]]
        ax.errorbar(x,means,yerr=errors,fmt='o',capsize=3,label=name)
    ax.set(xticks=range(len(spec['opponents'])),xticklabels=[c['name'] for c in spec['opponents']],ylabel='Chips / hand',title='Fixed pool performance (approx. 95% CI)');ax.axhline(0,color='grey',linestyle='--');ax.legend();ax.grid(alpha=.2)
    fig.savefig(output/'results.png',dpi=160);print(report,flush=True)

if __name__=='__main__':main()
