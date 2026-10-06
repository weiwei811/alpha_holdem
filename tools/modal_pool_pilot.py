import modal
from pathlib import Path
app=modal.App('alpha-holdem-pool-diagnostic-pilot')
volume=modal.Volume.from_name('alpha-holdem-budget-results',create_if_missing=True)
RUN_ID='six-player-20261005-pool-pilot'
HOUR1='six-player-20261005-t4-1h-16cpu-v4'
HOUR2='six-player-20261005-t4-hour2'
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("ray[rllib]==2.49.2", "numpy==1.26.4", "h5py", "pandas", "pandocfilters", "requests", "scipy", "SuperSuit", "tabulate", "tqdm", "Flask", "Flask-SocketIO", "matplotlib", "pytest", "termcolor")
    .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}))

image = image.pip_install("psutil==7.0.0")

def setup_source(source, directory="/workspace/alpha_holdem"):
    import io,os,zipfile
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(source)) as archive:archive.extractall(root)
    os.chdir(root)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','ORT_INTER_OP_NUM_THREADS','ORT_INTRA_OP_NUM_THREADS'):
        os.environ[key]='1'
    return root


def pool_cases():
    old=Path('/runs')/HOUR1/'league';new=Path('/runs')/HOUR2/'league'
    return [{'name':'random'},{'name':'check_call'},
            {'name':'initial','weights':[str(old/'initial_weight.pkl')]},
            {'name':'hour1_table','weights':[str(old/'output_weight.pkl')]},
            {'name':'hour2_table','weights':[str(new/'output_weight.pkl')]},
            {'name':'older_mixed','weights':[str(old/'weights'/f'c_{i}.pkl') for i in (10,19,30)]},
            {'name':'recent_mixed','weights':[str(new/'weights'/f'c_{i}.pkl') for i in (39,59,79)]}]


@app.function(image=image,cpu=(4,4),memory=(8192,8192),timeout=900,retries=0,
              volumes={'/runs':volume},max_containers=1,scaledown_window=2)
def benchmark(source, pilots=None):
    import json,subprocess,sys
    volume.reload();root=setup_source(source)
    output=Path('/runs')/RUN_ID/('diagnostic' if pilots is None else 'pilot_evaluation')
    output.mkdir(parents=True,exist_ok=True)
    old=Path('/runs')/HOUR1/'league';new=Path('/runs')/HOUR2/'league'
    config=json.loads((old/'training_config.json').read_text())
    if pilots is None:
        models={'hour1':str(old/'output_weight.pkl'),'hour2':str(new/'output_weight.pkl')};reference='hour1';hands=2400
    else:
        baseline=Path('/runs')/pilots['baseline']/'league'
        models={'baseline':str(baseline/'output_weight.pkl'),
                'ranked':str(Path('/runs')/RUN_ID/'ranked/league/output_weight.pkl'),
                'uniform':str(Path('/runs')/RUN_ID/'uniform/league/output_weight.pkl')}
        reference='baseline';hands=1200
    spec={'config':config,'models':models,'reference':reference,'opponents':pool_cases(),
          'hands':hands,'seed':3100000 if pilots is None else 4100000,'output':str(output)}
    (output/'spec.json').write_text(json.dumps(spec,indent=2))
    for case in spec['opponents']:
        for path in case.get('weights',[]):
            if not Path(path).exists():raise ValueError('Missing benchmark checkpoint: '+path)
    with (output/'evaluation.log').open('w') as log:
        result=subprocess.run([sys.executable,str(root/'tools/evaluate_policy_pool.py'),'--spec',str(output/'spec.json')],stdout=log,stderr=subprocess.STDOUT,timeout=810)
    volume.commit()
    if result.returncode:raise RuntimeError('Benchmark failed: '+str(output/'evaluation.log'))
    summary=json.loads((output/'evaluation.json').read_text());print((output/'report.md').read_text(),flush=True)
    # Prefer the earlier reference unless the broader average gives clear evidence for hour two.
    if pilots is None:
        improvement=summary['aggregate']['hour2']['approx_95pct_ci'][0]>0
        return {'baseline':HOUR2 if improvement else HOUR1,'aggregate':summary['aggregate']}
    return {'run_id':RUN_ID,'report':str(output/'report.md'),'aggregate':summary['aggregate']}


@app.function(image=image,gpu='T4',cpu=(16,16),memory=(16384,24576),timeout=420,
              startup_timeout=120,retries=0,volumes={'/runs':volume},max_containers=1,scaledown_window=2)
def pilot(source, baseline, sampling):
    import json,os,signal,subprocess,sys,time,torch
    volume.reload();root=setup_source(source,"/workspace/pilot_"+sampling)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    output=Path('/runs')/RUN_ID/sampling
    if (output/'metadata.json').exists():raise ValueError('Pilot already exists; choose new RUN_ID')
    output.mkdir(parents=True,exist_ok=True)
    (root/'work').symlink_to(output,target_is_directory=True)
    previous=Path('/runs')/baseline/'league'
    config=json.loads((previous/'training_config.json').read_text())
    # Keep restored configuration and all learning settings unchanged.
    config['metrics_episode_collection_timeout_s']=1
    (output/'config.py').write_text(repr(config))
    manifest=json.loads((previous/'latest_checkpoint.json').read_text())
    target=manifest['trained_steps']+160000
    command=[sys.executable,'train_league.py','--conf','work/config.py','--device','cuda',
             '--restore-state',str(previous),'--output_dir','work/league',
             '--experiment_name','pilot','--trained-steps',str(target),
             '--training-seconds','300','--checkpoint-steps','0','--eval-steps','0',
             '--freeze-opponents','--sp','0','--opponent-sampling',sampling]
    metadata={'baseline':baseline,'sampling':sampling,'start_steps':manifest['trained_steps'],
              'target_steps':target,'training_seconds_cap':300,'command':command,'started_at_unix':time.time()}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2));volume.commit()
    started=time.monotonic()
    with (output/'training.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:process.wait(timeout=365)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=25)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            metadata['watchdog_stop']=True
    metadata.update(returncode=process.returncode,elapsed_seconds=time.monotonic()-started)
    if (output/'league/latest_checkpoint.json').exists():metadata['final_checkpoint']=json.loads((output/'league/latest_checkpoint.json').read_text())
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2));volume.commit()
    print(json.dumps(metadata),flush=True)
    if process.returncode:raise RuntimeError('Pilot failed: '+str(output/'training.log'))
    return metadata


@app.function(image=modal.Image.debian_slim(python_version='3.11'),cpu=(.125,.125),memory=(256,256),timeout=2800,retries=0,max_containers=1)
def workflow(source,digest):
    result=benchmark.remote(source)
    for sampling in ('ranked','uniform'):pilot.remote(source,result['baseline'],sampling)
    final=benchmark.remote(source,result)
    return {'source_sha256':digest,'diagnostic':result,'pilot':final}


@app.local_entrypoint()
def main():
    import hashlib,io,json,zipfile
    root=Path(__file__).resolve().parents[1];buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ('agi','rlcard','confs'):
            for path in (root/folder).rglob('*'):
                if path.is_file() and path.suffix in ('.py','.json','.txt','.csv','.npy','.npz') and '__pycache__' not in path.parts:
                    archive.write(path,path.relative_to(root))
        for name in ('train_league.py','utils.py','tools/evaluate_policy_pool.py'):archive.write(root/name,name)
    source=buffer.getvalue();digest=hashlib.sha256(source).hexdigest()
    call=workflow.spawn(source,digest)
    launch={'run_id':RUN_ID,'call_id':call.object_id,'source_sha256':digest,
            'volume':'alpha-holdem-budget-results','extra_compute_max_estimate_usd':.55}
    (root/'work/pool_pilot_launch.json').write_text(json.dumps(launch,indent=2));print(json.dumps(launch),flush=True)
