import modal
from pathlib import Path
app=modal.App('alpha-holdem-structured-validation')
volume=modal.Volume.from_name('alpha-holdem-budget-results',create_if_missing=True)
RUN_ID='six-player-20261005-structured-validation-v2'
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("ray[rllib]==2.49.2", "numpy==1.26.4", "h5py", "pandas", "pandocfilters", "requests", "scipy", "SuperSuit", "tabulate", "tqdm", "Flask", "Flask-SocketIO", "matplotlib", "pytest", "termcolor")
    .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}))

image = image.pip_install("psutil==7.0.0")

@app.function(image=image,cpu=(4,4),memory=(8192,8192),timeout=240,retries=0,
              volumes={'/runs':volume},max_containers=1,scaledown_window=2)
def validate(source):
    import io,json,os,subprocess,sys,time,zipfile
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS'):
        os.environ[key]='1'
    root=Path('/workspace/alpha_holdem');root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(source)) as archive:archive.extractall(root)
    os.chdir(root)
    output=Path('/runs')/RUN_ID
    if output.exists():raise ValueError('Choose a fresh validation RUN_ID')
    output.mkdir(parents=True)
    (root/'work').symlink_to(output,target_is_directory=True)
    metadata={'started_at_unix':time.time(),'compute':'Modal CPU','commands':[]}
    def run(command,name,timeout):
        metadata['commands'].append(command)
        with (output/name).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        volume.commit()
        if result.returncode:raise RuntimeError('Validation failed: '+name)
    run([sys.executable,'-m','pytest','tests/test_structured_model.py','-q'],'tests.log',60)
    common=[sys.executable,'train_league.py','--conf','confs/nl_holdem_structured.py',
            '--device','cpu','--workers','1','--batch-size','200','--training-seconds','40',
            '--checkpoint-steps','200','--eval-steps','200','--eval-hands','6']
    run(common+['--trained-steps','400','--output_dir','work/fresh','--experiment_name','structured_fresh'],
        'fresh.log',90)
    manifest=json.loads((output/'fresh/latest_checkpoint.json').read_text())
    run(common+['--restore-state','work/fresh','--trained-steps',str(manifest['trained_steps']+200),
                '--output_dir','work/resumed','--experiment_name','structured_resumed'],'resumed.log',80)
    rows=[json.loads(x) for x in (output/'ray_results/structured_resumed/result.json').read_text().splitlines()]
    last=rows[-1];policy=last['info']['learner']['default_policy'];stats=policy['learner_stats']
    metadata.update(finished_at_unix=time.time(),fresh_steps=manifest['trained_steps'],
                    resumed_steps=last['num_env_steps_trained'],updates=policy['num_grad_updates_lifetime'],
                    learner_stats=stats,returncode=0)
    if metadata['resumed_steps']<=metadata['fresh_steps']:raise AssertionError('Resume did not train')
    import math
    if not all(math.isfinite(float(v)) for v in stats.values()):raise AssertionError('Nonfinite stats')
    report='# Structured model integration validation\n\n'
    report+='All model tests, fresh RLlib training, full-state resume and periodic evaluation ran on Modal CPU.\n\n'
    report+=json.dumps(metadata,indent=2)+'\n\nSix-hand evaluations verify execution only; these results do not establish improved playing strength. Architecture requires fresh training; existing checkpoints retain their original model/config.\n'
    (output/'report.md').write_text(report,encoding='utf-8')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2));volume.commit()
    print(report,flush=True)
    return metadata

@app.local_entrypoint()
def main():
    import io,json,zipfile
    root=Path(__file__).resolve().parents[1];buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ('agi','rlcard','confs'):
            for p in (root/folder).rglob('*'):
                if p.is_file() and p.suffix in ('.py','.json','.txt','.csv','.npy','.npz') and '__pycache__' not in p.parts:
                    archive.write(p,p.relative_to(root))
        for name in ('train_league.py','utils.py','tests/test_structured_model.py'):archive.write(root/name,name)
    call=validate.spawn(buffer.getvalue())
    record={'call_id':call.object_id,'run_id':RUN_ID,'volume':'alpha-holdem-budget-results'}
    (root/'work/structured_validation_launch.json').write_text(json.dumps(record,indent=2));print(record,flush=True)
