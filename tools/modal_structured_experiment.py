"""Equal-transition opponent-pool experiment and critic diagnostics on Modal."""
import modal
from pathlib import Path
app=modal.App('alpha-holdem-structured-pool-experiment')
volume=modal.Volume.from_name('alpha-holdem-budget-results')
RUN='six-player-20261006-structured-pool-experiment'
BASE='six-player-20261005-structured-t4-1h'
SECOND='six-player-20261005-structured-t4-hour2'
LEGACY='six-player-20261005-t4-1h-16cpu-v4'
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("ray[rllib]==2.49.2", "numpy==1.26.4", "h5py", "pandas", "pandocfilters", "requests", "scipy", "SuperSuit", "tabulate", "tqdm", "Flask", "Flask-SocketIO", "matplotlib", "pytest", "termcolor")
    .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}))

image = image.pip_install("psutil==7.0.0")


def setup(source):
    import os,io,zipfile,sys
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','ORT_INTER_OP_NUM_THREADS','ORT_INTRA_OP_NUM_THREADS'):os.environ[k]='1'
    root=Path('/workspace/experiment');root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(source)) as z:z.extractall(root)
    os.chdir(root)
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    volume.reload();return root

@app.function(image=image,gpu='T4',cpu=(16,16),memory=(16384,24576),timeout=600,startup_timeout=120,retries=0,max_containers=1,scaledown_window=2,volumes={'/runs':volume})
def pilot(source,sampling):
    import json,subprocess,sys,time,signal,os
    root=setup(source);out=Path('/runs')/RUN/sampling
    if (out/'metadata.json').exists() or (out.exists() and any(out.iterdir())):
        # A preempted pilot must not reset its paid budget or enter a crash loop.
        p=out/'metadata.json';return json.loads(p.read_text()) if p.exists() else {'incomplete':True}
    out.mkdir(parents=True,exist_ok=True)
    work=root/'work'
    if work.is_symlink():work.unlink()
    elif work.exists():raise ValueError('Work path is not the expected symlink')
    work.symlink_to(out,target_is_directory=True)
    previous=Path('/runs')/BASE/'league';conf=json.loads((previous/'training_config.json').read_text())
    start=json.loads((previous/'latest_checkpoint.json').read_text())['trained_steps']
    conf.update(min_time_s_per_iteration=0,min_sample_timesteps_per_iteration=0,min_train_timesteps_per_iteration=4000)
    (out/'config.py').write_text(repr(conf))
    command=[sys.executable,'train_league.py','--conf','work/config.py','--device','cuda','--restore-state',str(previous),
             '--output_dir','work/league','--experiment_name','pilot','--trained-steps',str(start+120000),
             '--training-seconds','480','--checkpoint-steps','4000','--eval-steps','0','--freeze-opponents','--sp','0','--opponent-sampling',sampling]
    meta={'sampling':sampling,'start_steps':start,'target_steps':start+120000,'status':'running','started_at_unix':time.time()}
    (out/'metadata.json').write_text(json.dumps(meta));volume.commit();started=time.monotonic()
    with (out/'training.log').open('w') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:proc.wait(timeout=535)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=25)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    meta.update(returncode=proc.returncode,elapsed_seconds=time.monotonic()-started,status='finished')
    if (out/'league/latest_checkpoint.json').exists():meta['final_steps']=json.loads((out/'league/latest_checkpoint.json').read_text())['trained_steps']
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));volume.commit();print(meta,flush=True);return meta

@app.function(image=image,cpu=(4,4),memory=(8192,8192),timeout=660,retries=0,max_containers=1,scaledown_window=2,volumes={'/runs':volume})
def analyze(source):
    import json,pickle,subprocess,sys,numpy as np
    root=setup(source);out=Path('/runs')/RUN;analysis=out/'analysis';analysis.mkdir(exist_ok=True)
    if (analysis/'findings.md').exists():return {'run_id':RUN,'report':str(analysis/'findings.md')}
    base=Path('/runs')/BASE/'league';second=Path('/runs')/SECOND/'league';legacy=Path('/runs')/LEGACY/'league'
    config=json.loads((base/'training_config.json').read_text());legacy_conf=json.loads((legacy/'training_config.json').read_text())
    start=json.loads((base/'latest_checkpoint.json').read_text())['trained_steps']
    checkpoints={}
    for name in ('ranked','uniform'):
        d=out/name/'league/checkpoints'
        checkpoints[name]={int(p.name.split('_')[1]):p for p in d.glob('step_*') if (p/'policies/default_policy/policy_state.pkl').exists()}
    shared=set(checkpoints['ranked']) & set(checkpoints['uniform'])
    shared={n for n in shared if start+80000<=n<=start+120000}
    if not shared:raise ValueError('No equal-step checkpoints with at least 80k new transitions; refuse an unequal comparison')
    matched=max(shared)
    models={'baseline':str(base/'output_weight.pkl')}
    for name in ('ranked','uniform'):
        with (checkpoints[name][matched]/'policies/default_policy/policy_state.pkl').open('rb') as f:state=pickle.load(f)
        path=out/name/'matched_weight.pkl'
        with path.open('wb') as f:pickle.dump(state['weights'],f)
        models[name]=str(path)
    paths=sorted((base/'weights').glob('c_*.pkl'),key=lambda p:int(p.stem.split('_')[1]))
    mix=[str(paths[int(q*(len(paths)-1))]) for q in (.2,.5,.8)]
    cases=[{'name':'random'},{'name':'check_call'},{'name':'initial','weights':[str(base/'initial_weight.pkl')]},
           {'name':'hour2_table','weights':[str(second/'output_weight.pkl')]}, {'name':'historical_mix','weights':mix},
           {'name':'legacy_table','weights':[str(legacy/'output_weight.pkl')],'config':legacy_conf}]
    spec={'config':config,'models':models,'reference':'baseline','opponents':cases,'hands':1200,'seed':9100000,'output':str(analysis)}
    (analysis/'spec.json').write_text(json.dumps(spec,indent=2))
    with (analysis/'evaluation.log').open('w') as log:r=subprocess.run([sys.executable,'tools/evaluate_policy_pool.py','--spec',str(analysis/'spec.json')],stdout=log,stderr=subprocess.STDOUT,timeout=400)
    volume.commit()
    if r.returncode:raise RuntimeError('Evaluation failed; inspect saved log')
    summary=json.loads((analysis/'evaluation.json').read_text())
    def paired(name,reference):
        left=np.mean([r['rewards'] for r in summary['results'] if r['model']==name],axis=0)
        right=np.mean([r['rewards'] for r in summary['results'] if r['model']==reference],axis=0)
        delta=left-right;m=float(delta.mean());rad=1.96*float(delta.std(ddof=1))/np.sqrt(len(delta));return {'mean':m,'ci':[m-rad,m+rad]}
    contrast=paired('uniform','ranked');summary['uniform_minus_ranked']=contrast;summary['matched_steps']=matched;summary['additional_steps']=matched-start
    (analysis/'evaluation.json').write_text(json.dumps(summary,indent=2))
    critic={'config':config,'models':{'hour1':str(base/'output_weight.pkl'),'hour2':str(second/'output_weight.pkl')},'opponent':str(legacy/'output_weight.pkl'),'opponent_config':legacy_conf,'hands':600,'seed':10100000,'output':str(analysis/'critic')}
    (analysis/'critic_spec.json').write_text(json.dumps(critic))
    with (analysis/'critic.log').open('w') as log:r=subprocess.run([sys.executable,'tools/diagnose_critic.py','--spec',str(analysis/'critic_spec.json')],stdout=log,stderr=subprocess.STDOUT,timeout=160)
    if r.returncode:raise RuntimeError('Critic diagnostics failed')
    calibration=json.loads((analysis/'critic/critic.json').read_text())
    report='# Structured model plateau experiment\n\n'
    report+=f'Both pilots restored the same optimizer/league checkpoint at {start:,} transitions. Compared at exactly {matched:,} transitions (+{matched-start:,}); the pool remained frozen and self-play probability zero. Learning settings are unchanged. Runs share a training seed but asynchronous rollouts are stochastic.\n\n'
    report+=(analysis/'report.md').read_text()+'\nUniform minus ranked benchmark average: '+json.dumps(contrast)+'\n\n'
    if contrast['ci'][0]>0:report+='Uniform sampling outperformed ranked sampling in this pilot; replication is needed before changing the default.\n'
    elif contrast['ci'][1]<0:report+='Ranked sampling outperformed uniform sampling in this pilot.\n'
    else:report+='Opponent sampling alone has not shown a statistically clear benefit in this pilot. This does not prove it cannot help with more training.\n'
    for name in ('hour1','hour2'):report+='\n'+name+' critic: '+json.dumps(calibration[name]['overall'])+'\n'
    report+='\nNative IMPALA value loss is batch-summed; large logged values alone do not indicate a scaling bug. Monte Carlo calibration includes irreducible hidden-card variance. One training seed and a short continuation cannot establish the root cause of a plateau.\n'
    (analysis/'findings.md').write_text(report);volume.commit();print(report,flush=True)
    return {'run_id':RUN,'matched_steps':matched,'contrast':contrast,'aggregate':summary['aggregate'],'critic':{n:calibration[n]['overall'] for n in ('hour1','hour2')}}

@app.function(image=modal.Image.debian_slim(python_version='3.11'),cpu=(.125,.125),memory=(256,256),timeout=1900,retries=0,max_containers=1)
def workflow(source):
    for name in ('ranked','uniform'):
        result=pilot.remote(source,name)
        if result.get('returncode')!=0:raise RuntimeError('Pilot incomplete; stopping experiment without another training attempt')
    return analyze.remote(source)

@app.function(image=image,cpu=(2,2),memory=(4096,4096),timeout=180,retries=0,max_containers=1,scaledown_window=2,volumes={'/runs':volume})
def sanity(source):
    import json,subprocess,sys
    root=setup(source);legacy=Path('/runs')/LEGACY/'league';out=Path('/runs')/RUN/'sanity';out.mkdir(parents=True,exist_ok=True)
    conf=json.loads((legacy/'training_config.json').read_text());weight=str(legacy/'output_weight.pkl')
    spec={'config':conf,'models':{'identical':weight},'reference':'identical','opponents':[{'name':'identical_table','weights':[weight]}],'hands':2400,'seed':12100000,'output':str(out)}
    (out/'spec.json').write_text(json.dumps(spec))
    with (out/'evaluation.log').open('w') as log:r=subprocess.run([sys.executable,'tools/evaluate_policy_pool.py','--spec',str(out/'spec.json')],stdout=log,stderr=subprocess.STDOUT,timeout=150)
    volume.commit()
    if r.returncode:raise RuntimeError('Evaluator sanity check failed')
    result=json.loads((out/'evaluation.json').read_text())['results'][0]
    verdict={'chips_per_hand':result['chips_per_hand'],'ci':result['approx_95pct_ci'],'zero_in_interval':result['approx_95pct_ci'][0]<=0<=result['approx_95pct_ci'][1]}
    (out/'sanity.json').write_text(json.dumps(verdict,indent=2));volume.commit();print(verdict,flush=True);return verdict

@app.local_entrypoint()
def main(sanity_only: bool = False, analyze_only: bool = False):
    import io,zipfile,json
    root=Path(__file__).resolve().parents[1];b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('agi','rlcard','confs'):
            for p in (root/folder).rglob('*'):
                if p.is_file() and p.suffix in ('.py','.json','.txt','.csv','.npy','.npz') and '__pycache__' not in p.parts:z.write(p,p.relative_to(root))
        for name in ('train_league.py','utils.py','tools/evaluate_policy_pool.py','tools/diagnose_critic.py'):z.write(root/name,name)
    call=(sanity if sanity_only else analyze if analyze_only else workflow).spawn(b.getvalue());record={'call_id':call.object_id,'run_id':RUN,'max_compute_estimate_usd':.60}
    (root/('work/structured_sanity_launch.json' if sanity_only else 'work/structured_experiment_launch.json')).write_text(json.dumps(record,indent=2));print(record,flush=True)
