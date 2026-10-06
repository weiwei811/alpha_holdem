"""Held-out Monte Carlo value calibration; intended to run on Modal."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def summarize(rows):
    import numpy as np
    if not rows:return {'decisions':0}
    v=np.array([r['value'] for r in rows]);y=np.array([r['return'] for r in rows])
    variance=float(y.var())
    return {'decisions':len(rows),'mean_prediction':float(v.mean()),'mean_return':float(y.mean()),
            'prediction_std':float(v.std()),'return_std':float(y.std()),'rmse':float(np.sqrt(np.mean((v-y)**2))),
            'explained_variance':float(1-(y-v).var()/variance) if variance>0 else None,
            'correlation':float(np.corrcoef(v,y)[0,1]) if v.std()>1e-10 and y.std()>1e-10 else None}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);args=parser.parse_args()
    import numpy as np,torch
    from agi.nl_holdem_env import NlHoldemEnvWrapper
    from agi.evaluation_tools import NNAgent
    torch.set_num_threads(1)
    spec=json.loads(args.spec.read_text());conf=spec['config'];env=NlHoldemEnvWrapper(conf)
    scale=conf['env_config']['custom_options'].get('reward_scale',200)
    output=Path(spec['output']);output.mkdir(exist_ok=True,parents=True)
    models={n:NNAgent(env.observation_space,env.action_space,conf,p,device='cpu') for n,p in spec['models'].items()}
    table=NNAgent(env.observation_space,env.action_space,spec['opponent_config'],spec['opponent'],device='cpu')
    report={}
    for name,agent in models.items():
        records=[]
        for h in range(spec.get('hands',600)):
            hero=h%6;seed=spec.get('seed',8100000)+h
            obs,_=env.reset(seed=seed);agent.rng=np.random.default_rng(seed+100000);table.rng=np.random.default_rng(seed+200000)
            current=[];done=False
            while not done:
                if env.my_agent()==hero:
                    tensors={k:torch.as_tensor(v).unsqueeze(0) for k,v in obs.items()}
                    with torch.no_grad():
                        agent.model({'obs':tensors},[],None);value=float(agent.model.value_function()[0])
                    b=obs['betting_info'];current.append({'value':value,'street':int(b[9]),'opponents':int(b[8]),'pot_bucket':'small' if b[0]<40 else 'medium' if b[0]<200 else 'large'})
                    action=agent.make_action(obs)
                else:action=table.make_action(obs)
                obs,payoff,done,_,_=env.step(action)
            for r in current:r['return']=float(payoff[hero])/scale
            records+=current
        groups={}
        for key in ('street','opponents','pot_bucket'):
            groups[key]={str(v):summarize([r for r in records if r[key]==v]) for v in sorted(set(r[key] for r in records))}
        report[name]={'overall':summarize(records),'groups':groups}
        (output/(name+'_decisions.json')).write_text(json.dumps(records))
    report['notes']=['Returns are realized terminal chips divided by reward_scale; no future rewards leak into model inputs.',
      'Many decisions share one terminal payoff; grouped diagnostics are descriptive, not independent confidence intervals.',
      'Poker return variance can be dominated by hidden cards; low explained variance alone does not prove a critic bug.',
      'Native RLlib IMPALA vf_loss is a sum over batch decisions, not per-decision MSE. Do not interpret thousands as unscaled chips.']
    (output/'critic.json').write_text(json.dumps(report,indent=2))
    text='# Critic calibration on held-out trained-opponent tables\n\n'
    for name in models:text+=name+': '+json.dumps(report[name]['overall'])+'\n\n'
    for n in report['notes']:text+='- '+n+'\n'
    (output/'critic.md').write_text(text);print(text,flush=True)
if __name__=='__main__':main()
