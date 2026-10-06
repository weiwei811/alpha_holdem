import modal
from pathlib import Path
app=modal.App('alpha-holdem-learning-lab')
volume=modal.Volume.from_name('alpha-holdem-budget-results')
RUN='six-player-20261006-learning-lab'
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("ray[rllib]==2.49.2", "numpy==1.26.4", "h5py", "pandas", "pandocfilters", "requests", "scipy", "SuperSuit", "tabulate", "tqdm", "Flask", "Flask-SocketIO", "matplotlib", "pytest", "termcolor")
    .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}))

image = image.pip_install("psutil==7.0.0")


@app.function(image=image,gpu='T4',cpu=(4,4),memory=(8192,8192),timeout=600,retries=0,max_containers=1,scaledown_window=2,volumes={'/runs':volume})
def probe(source):
    import io,zipfile,os,subprocess,sys,json
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
    volume.reload();root=Path('/workspace/lab');root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(source)) as z:z.extractall(root)
    os.chdir(root);out=Path('/runs')/RUN/'critic_probe';out.mkdir(parents=True,exist_ok=True)
    if (out/'probe.json').exists():return json.loads((out/'probe.json').read_text())
    with (out/'probe.log').open('w') as log:r=subprocess.run([sys.executable,'tools/fit_critic_probe.py','--base','/runs/six-player-20261005-structured-t4-1h/league','--legacy','/runs/six-player-20261005-t4-1h-16cpu-v4/league','--output',str(out)],stdout=log,stderr=subprocess.STDOUT,timeout=540)
    volume.commit()
    if r.returncode:raise RuntimeError('Probe failed; inspect saved log')
    return json.loads((out/'probe.json').read_text())
BASE='six-player-20261005-structured-t4-1h'
LEGACY='six-player-20261005-t4-1h-16cpu-v4'
VARIANTS={'control':{'entropy':.1,'sampling':'ranked','refresh':.01},'low_entropy':{'entropy':.02,'sampling':'ranked','refresh':.01},'refresh_uniform':{'entropy':.1,'sampling':'uniform','refresh':.25}}

def setup(source):
    import os,sys,io,zipfile
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','ORT_INTER_OP_NUM_THREADS','ORT_INTRA_OP_NUM_THREADS'):os.environ[k]='1'
    volume.reload();root=Path('/workspace/lab');root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(source)) as z:z.extractall(root)
    os.chdir(root)
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    return root

@app.function(image=image,gpu='T4',cpu=(16,16),memory=(16384,24576),timeout=900,retries=0,max_containers=1,scaledown_window=2,volumes={'/runs':volume})
def candidate(source,name):
    import json,subprocess,sys,os,signal,time
    root=setup(source);out=Path('/runs')/RUN/name
    if (out/'metadata.json').exists():return json.loads((out/'metadata.json').read_text())
    out.mkdir(parents=True,exist_ok=True);work=root/'work'
    if work.is_symlink():work.unlink()
    elif work.exists():raise ValueError('Unexpected work directory')
    work.symlink_to(out,target_is_directory=True)
    base=Path('/runs')/BASE/'league';conf=json.loads((base/'training_config.json').read_text());start=json.loads((base/'latest_checkpoint.json').read_text())['trained_steps']
    conf.update(min_time_s_per_iteration=0,min_sample_timesteps_per_iteration=0,min_train_timesteps_per_iteration=4000)
    (out/'config.py').write_text(repr(conf));v=VARIANTS[name]
    command=[sys.executable,'train_league.py','--conf','work/config.py','--device','cuda','--restore-state',str(base),'--output_dir','work/league','--experiment_name','candidate','--trained-steps',str(start+240000),'--training-seconds','760','--checkpoint-steps','4000','--eval-steps','0','--freeze-opponents','--entropy-coeff',str(v['entropy']),'--opponent-sampling',v['sampling'],'--exg_oppo_prob',str(v['refresh'])]
    meta={'variant':v,'name':name,'status':'running','start_steps':start,'target_steps':start+240000,'started_at_unix':time.time()}
    (out/'metadata.json').write_text(json.dumps(meta));volume.commit();started=time.monotonic()
    with (out/'training.log').open('w') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:proc.wait(timeout=830)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=25)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    meta.update(status='finished',returncode=proc.returncode,elapsed_seconds=time.monotonic()-started)
    if (out/'league/latest_checkpoint.json').exists():meta['final_steps']=json.loads((out/'league/latest_checkpoint.json').read_text())['trained_steps']
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));volume.commit();print(meta,flush=True);return meta

@app.function(image=image,cpu=(4,4),memory=(8192,8192),timeout=660,retries=0,max_containers=1,scaledown_window=2,volumes={'/runs':volume})
def select(source):
    import json,pickle,numpy as np,subprocess,sys
    root=setup(source);out=Path('/runs')/RUN;analysis=out/'selection';analysis.mkdir(exist_ok=True)
    if (analysis/'selection.json').exists():return json.loads((analysis/'selection.json').read_text())
    base=Path('/runs')/BASE/'league';legacy=Path('/runs')/LEGACY/'league';conf=json.loads((base/'training_config.json').read_text());lc=json.loads((legacy/'training_config.json').read_text());start=json.loads((base/'latest_checkpoint.json').read_text())['trained_steps']
    checkpoints={n:{int(p.name.split('_')[1]):p for p in (out/n/'league/checkpoints').glob('step_*') if (p/'policies/default_policy/policy_state.pkl').exists()} for n in VARIANTS}
    common=set.intersection(*(set(c) for c in checkpoints.values()));common={v for v in common if start+160000<=v<=start+240000}
    if not common:raise ValueError('No suitable equal-step comparison; refusing unequal pilots')
    matched=max(common);models={'baseline':str(base/'output_weight.pkl')}
    for name in VARIANTS:
        with (checkpoints[name][matched]/'policies/default_policy/policy_state.pkl').open('rb') as f:state=pickle.load(f)
        path=out/name/'matched_weight.pkl'
        with path.open('wb') as f:pickle.dump(state['weights'],f)
        models[name]=str(path)
    paths=sorted((base/'weights').glob('c_*.pkl'),key=lambda p:int(p.stem.split('_')[1]));mix=[str(paths[int(q*(len(paths)-1))]) for q in (.2,.5,.8)]
    cases=[{'name':'random'},{'name':'check_call'},{'name':'initial','weights':[str(base/'initial_weight.pkl')]},{'name':'historical_mix','weights':mix},{'name':'legacy_table','weights':[str(legacy/'output_weight.pkl')],'config':lc}]
    spec={'config':conf,'models':models,'reference':'baseline','opponents':cases,'hands':1200,'seed':15100000,'output':str(analysis)}
    (analysis/'spec.json').write_text(json.dumps(spec,indent=2))
    if not (analysis/'evaluation.json').exists():
        with (analysis/'evaluation.log').open('w') as log:r=subprocess.run([sys.executable,'tools/evaluate_policy_pool.py','--spec',str(analysis/'spec.json')],stdout=log,stderr=subprocess.STDOUT,timeout=560)
        volume.commit()
        if r.returncode:raise RuntimeError('Selection evaluation failed')
    elif json.loads((analysis/'evaluation.json').read_text())['spec'] != spec:
        raise ValueError('Saved selection evaluation uses different settings')
    summary=json.loads((analysis/'evaluation.json').read_text());contrasts={}
    def paired(name,reference):
        means={m:np.mean([r['rewards'] for r in summary['results'] if r['model']==m],axis=0) for m in (name,reference)}
        d=means[name]-means[reference];mean=float(d.mean());rad=1.96*float(d.std(ddof=1))/np.sqrt(len(d));return {'mean':mean,'ci':[mean-rad,mean+rad]}
    for n in ('low_entropy','refresh_uniform'):contrasts[n]=paired(n,'control')
    winner=max(('low_entropy','refresh_uniform'),key=lambda n:summary['aggregate'][n]['mean_chips_per_hand'])
    result={'matched_steps':matched,'additional_steps':matched-start,'aggregate':summary['aggregate'],'contrasts_vs_control':contrasts,'candidate':winner,'candidate_settings':VARIANTS[winner], 'supported_vs_control':bool(contrasts[winner]['ci'][0]>0)}
    (analysis/'selection.json').write_text(json.dumps(result,indent=2));report=(analysis/'report.md').read_text()+'\nMatched-step settings comparison:\n\n'+json.dumps(result,indent=2)+'\n\nSelection uses a validation deal bank. The selected candidate must pass a separate final test bank before claiming final improvement.\n'
    (analysis/'findings.md').write_text(report);volume.commit();print(report,flush=True);return result

@app.function(image=modal.Image.debian_slim(python_version='3.11'),cpu=(.125,.125),memory=(256,256),timeout=3500,retries=0,max_containers=1)
def screening(source):
    for name in VARIANTS:
        result=candidate.remote(source,name)
        if result.get('returncode')!=0:raise RuntimeError('Candidate incomplete; no paid retry')
    return select.remote(source)

@app.function(image=image,cpu=(1,1),memory=(4096,4096),timeout=120,retries=0,max_containers=1,scaledown_window=2,volumes={'/runs':volume})
def export_selected(source):
    import json,shutil,pickle,numpy as np
    setup(source)
    volume.reload();out=Path('/runs')/RUN
    result=json.loads((out/'selection/selection.json').read_text());winner=result['candidate'];matched=result['matched_steps']
    paths=list((out/winner/'league/checkpoints').glob('step_'+str(matched)+'_*'))
    # Interval and end-of-run saves can share a step. Require the exact
    # evaluated weights rather than assuming one directory per step.
    with (out/winner/'matched_weight.pkl').open('rb') as f:expected=pickle.load(f)
    matching=[]
    for path in sorted(paths):
        with (path/'policies/default_policy/policy_state.pkl').open('rb') as f:weights=pickle.load(f)['weights']
        if weights.keys()==expected.keys() and all(np.array_equal(weights[k],expected[k]) for k in expected):matching.append(path)
    if not matching:raise ValueError('No full checkpoint matches the evaluated weights')
    checkpoint=matching[-1]
    if not (checkpoint/'league_state.pkl').exists():raise ValueError('Selected optimizer/league state is incomplete')
    resume=out/'selected_resume';resume.mkdir(exist_ok=True)
    config=json.loads((out/winner/'league/training_config.json').read_text())
    (resume/'training_config.json').write_text(json.dumps(config,indent=2))
    (resume/'latest_checkpoint.json').write_text(json.dumps({'path':'../'+winner+'/league/checkpoints/'+checkpoint.name,'trained_steps':matched}))
    shutil.copy2(out/winner/'matched_weight.pkl',resume/'output_weight.pkl')
    shutil.copy2(out/winner/'league/initial_weight.pkl',resume/'initial_weight.pkl')
    (resume/'selection.json').write_text(json.dumps(result,indent=2));volume.commit();return {'resume_run':RUN,'resume_subdir':'selected_resume','steps':matched,'candidate':winner,'settings':result['candidate_settings']}

@app.local_entrypoint()
def main(stage: str = "probe"):
    import io,zipfile,json
    root=Path(__file__).resolve().parents[1];b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('agi','rlcard','confs'):
            for p in (root/folder).rglob('*'):
                if p.is_file() and p.suffix in ('.py','.json','.txt','.csv','.npy','.npz') and '__pycache__' not in p.parts:z.write(p,p.relative_to(root))
        for name in ('tools/fit_critic_probe.py','tools/evaluate_policy_pool.py','train_league.py','utils.py'):z.write(root/name,name)
    if stage not in ('probe','screen','select','export'):
        raise ValueError('stage must be probe, screen, select or export')
    call={'probe':probe,'screen':screening,'select':select,'export':export_selected}[stage].spawn(b.getvalue());record={'call_id':call.object_id,'run_id':RUN,'stage':stage,'max_compute_usd':.15 if stage=='probe' else 1.26}
    (root/'work/learning_lab_launch.json').write_text(json.dumps(record));print(record)
