"""Fit the existing critic to fixed on-policy returns with a hand-level split."""
import argparse,json,sys,copy,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--legacy',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    import numpy as np,torch
    from agi.nl_holdem_env import NlHoldemEnvWrapper
    from agi.evaluation_tools import NNAgent
    torch.set_num_threads(1);torch.manual_seed(42)
    base=Path(a.base);legacy=Path(a.legacy);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    conf=json.loads((base/'training_config.json').read_text());lc=json.loads((legacy/'training_config.json').read_text());env=NlHoldemEnvWrapper(conf)
    agent=NNAgent(env.observation_space,env.action_space,conf,base/'output_weight.pkl',device='cpu')
    foe=NNAgent(env.observation_space,env.action_space,lc,legacy/'output_weight.pkl',device='cpu')
    rng=np.random.default_rng(42);obses=[];targets=[];hands=[]
    for hand in range(1200):
        obs,_=env.reset(seed=14100000+hand);hero=hand%6;agent.rng=np.random.default_rng(hand+1000);foe.rng=np.random.default_rng(hand+2000);pending=[];done=False
        while not done:
            if env.my_agent()==hero:
                pending.append({k:v.copy() for k,v in obs.items()});action=agent.make_action(obs)
            elif hand%3==0:action=foe.make_action(obs)
            elif hand%3==1:action=1 if obs['legal_moves'][1] else int(np.flatnonzero(obs['legal_moves'])[0])
            else:action=int(rng.choice(np.flatnonzero(obs['legal_moves'])))
            obs,reward,done,_,_=env.step(action)
        obses+=pending;targets += [float(reward[hero])/200]*len(pending);hands += [hand]*len(pending)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    x={k:torch.as_tensor(np.stack([o[k] for o in obses]),device=device) for k in obses[0]};y=torch.tensor(targets,device=device)
    train=torch.tensor([i for i,h in enumerate(hands) if h%5],device=device);valid=torch.tensor([i for i,h in enumerate(hands) if not h%5],device=device)
    tiny=train[:128]
    model=agent.model.to(device)
    def prediction(m,indices):
        m({'obs':{k:v[indices] for k,v in x.items()}},[],None);return m.value_function()
    def metrics(m,indices):
        with torch.no_grad():
            values=torch.cat([prediction(m,b) for b in indices.split(128)]);target=y[indices]
            mse=float(((values-target)**2).mean());variance=float(target.var(unbiased=False));return {'mse':mse,'ev':1-float((values-target).var(unbiased=False))/variance,'prediction_std':float(values.std(unbiased=False))}
    result={'device':device,'decisions':len(y),'train_decisions':len(train),'validation_decisions':len(valid),'initial_train':metrics(model,train),'initial_validation':metrics(model,valid),'constant_validation_mse':float(((y[valid]-y[train].mean())**2).mean())}
    original=copy.deepcopy(model.state_dict())
    params=list(model.value_encoder.parameters())+list(model.value_head.parameters())
    for name,param in model.named_parameters():param.requires_grad_(name.startswith('value_'))
    opt=torch.optim.Adam(params,lr=0.001)
    for _ in range(500):
        opt.zero_grad();loss=((prediction(model,tiny)-y[tiny])**2).mean();loss.backward();opt.step()
    result['tiny_initial_mse']=float(((y[tiny]-y[tiny].mean())**2).mean());result['tiny_fitted']=metrics(model,tiny)
    model.load_state_dict(original);opt=torch.optim.Adam(params,lr=0.0003)
    best=None;bestmse=float('inf')
    for step in range(1200):
        index=train[torch.randint(len(train),(128,),device=device)];opt.zero_grad();loss=((prediction(model,index)-y[index])**2).mean();loss.backward();torch.nn.utils.clip_grad_norm_(params,10);opt.step()
        if (step+1)%100==0:
            check=metrics(model,valid)
            if check['mse']<bestmse:bestmse=check['mse'];best=copy.deepcopy(model.state_dict());result['best_validation_step']=step+1
    result['final_train']=metrics(model,train);model.load_state_dict(best);result['best_validation']=metrics(model,valid)
    result['notes']=['Held-out hands share no terminal targets with training hands. Validation selection is diagnostic, not a final test set.', 'This fits realized noisy returns, not true conditional expected values. Overfitting the tiny set tests capacity/optimization, not playing strength.']
    (out/'probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
