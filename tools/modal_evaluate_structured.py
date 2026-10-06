"""Bounded post-training comparison on Modal CPU."""
import sys
from pathlib import Path
import modal
volume=modal.Volume.from_name('alpha-holdem-budget-results')
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("ray[rllib]==2.49.2", "numpy==1.26.4", "h5py", "pandas", "pandocfilters", "requests", "scipy", "SuperSuit", "tabulate", "tqdm", "Flask", "Flask-SocketIO", "matplotlib", "pytest", "termcolor")
    .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}))

image = image.pip_install("psutil==7.0.0")


app=modal.App('alpha-holdem-structured-evaluation')
RUN='six-player-20261005-structured-t4-1h'
OLD='six-player-20261005-t4-1h-16cpu-v4'
@app.function(image=image,cpu=(4,4),memory=(8192,8192),timeout=1500,retries=0,volumes={'/runs':volume},scaledown_window=2,max_containers=1)
def evaluate(source, run_id=RUN, previous_run=OLD, hands=1200, seed=6100000, historical_mix=False):
    import io,zipfile,json,os,subprocess,time,pickle
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
    volume.reload()
    root=Path('/workspace/eval');root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(source)) as z:z.extractall(root)
    os.chdir(root)
    run=Path('/runs')/run_id;old=Path('/runs')/previous_run
    out=run/('comparison_'+previous_run+'_seed_'+str(seed)+'_hands_'+str(hands)+('_mixed' if historical_mix else ''));out.mkdir(exist_ok=True)
    if (out/'evaluation.json').exists():
        summary=json.loads((out/'evaluation.json').read_text())
        # Older reports repeated comparisons under an incorrect section label.
        report=(out/'report.md').read_text().split('\nStructured final minus previous model:')[0].split('\nStructured final minus legacy hour-one model:')[0]
        report+='\n'+json.dumps(summary.get('health',{}))+'\n\nThis compares checkpoints from one training seed against fixed opponent tables; it does not establish exploitability or an architecture advantage.\n'
        (out/'report.md').write_text(report)
        unique={ (c['model'],c['reference'],c['opponent']):c for c in summary['paired_comparisons'] }
        summary['paired_comparisons']=list(unique.values())
        (out/'evaluation.json').write_text(json.dumps(summary,indent=2))
        volume.commit()
        return {'aggregate':summary['aggregate'],'health':summary.get('health',{})}
    conf=json.loads((run/'league/training_config.json').read_text());legacy=json.loads((old/'league/training_config.json').read_text())
    initial=str(run/'league/initial_weight.pkl');final=str(run/'league/output_weight.pkl');previous=str(old/'league/output_weight.pkl')
    spec={'config':conf,'model_configs':{'previous':legacy},'models':{'initial':initial,'structured_final':final,'previous':previous},'reference':'previous',
          'opponents':[{'name':'random'},{'name':'check_call'},{'name':'structured_initial','weights':[initial]}, {'name':'legacy_table','weights':[str(Path('/runs')/OLD/'league/output_weight.pkl')],'config':json.loads((Path('/runs')/OLD/'league/training_config.json').read_text())}],
          'hands':hands,'seed':seed,'output':str(out)}
    if historical_mix:
        paths=sorted((old/'league/weights').glob('c_*.pkl'),key=lambda p:int(p.stem.split('_')[1]))
        if not paths:raise ValueError('Reference historical pool is missing')
        mix=[str(paths[int(q*(len(paths)-1))]) for q in (.2,.5,.8)]
        spec['opponents'].append({'name':'historical_mix','weights':mix,'config':legacy})
    (out/'spec.json').write_text(json.dumps(spec,indent=2))
    with (out/'evaluation.log').open('w') as log:
        result=subprocess.run([sys.executable,'tools/evaluate_policy_pool.py','--spec',str(out/'spec.json')],stdout=log,stderr=subprocess.STDOUT,timeout=1440)
    volume.commit()
    if result.returncode:raise RuntimeError('Evaluation failed; inspect evaluation.log')
    import numpy as np
    rows=[json.loads(x) for x in (run/'ray_results/t4_one_hour/result.json').read_text().splitlines()]
    with (run/'league/output_weight.pkl').open('rb') as f:weights=pickle.load(f)
    summary=json.loads((out/'evaluation.json').read_text())
    summary['health']={'parameters':sum(v.size for v in weights.values()),'all_weights_finite':all(bool(np.isfinite(v).all()) for v in weights.values()),'trained_steps':rows[-1]['num_env_steps_trained'],'updates':rows[-1]['info']['learner']['default_policy']['num_grad_updates_lifetime']}
    (out/'evaluation.json').write_text(json.dumps(summary,indent=2))
    report=(out/'report.md').read_text()
    report+='\n'+json.dumps(summary['health'])+'\n\nThis compares checkpoints from one training seed against fixed opponent tables; it does not establish exploitability or an architecture advantage.\n'
    (out/'report.md').write_text(report)
    volume.commit();print(report,flush=True)
    return {'aggregate':summary['aggregate'],'health':summary['health']}
@app.local_entrypoint()
def main(run_id: str = RUN, previous_run: str = OLD, hands: int = 1200, seed: int = 6100000, historical_mix: bool = False):
    import io,zipfile,json
    root=Path(__file__).resolve().parents[1];b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('agi','rlcard','confs'):
            for p in (root/folder).rglob('*'):
                if p.is_file() and p.suffix in ('.py','.json','.txt','.csv','.npy','.npz') and '__pycache__' not in p.parts:z.write(p,p.relative_to(root))
        z.write(root/'tools/evaluate_policy_pool.py','tools/evaluate_policy_pool.py')
    if hands < 6 or hands % 6:
        raise ValueError('hands must be a positive multiple of six')
    call=evaluate.spawn(b.getvalue(),run_id,previous_run,hands,seed,historical_mix);record={'call_id':call.object_id,'run_id':run_id,'previous_run':previous_run,'hands':hands,'seed':seed,'historical_mix':historical_mix,'max_compute_estimate_usd':.13}
    (root/'work/structured_evaluation_launch.json').write_text(json.dumps(record));print(record)
