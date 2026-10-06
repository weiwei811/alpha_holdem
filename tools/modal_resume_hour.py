import modal
from pathlib import Path
app = modal.App("alpha-holdem-six-player-t4-resume-and-evaluate")
volume = modal.Volume.from_name("alpha-holdem-budget-results", create_if_missing=True)
RUN_ID = "six-player-20261005-t4-hour2"
PREVIOUS_RUN = "six-player-20261005-t4-1h-16cpu-v4"
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("ray[rllib]==2.49.2", "numpy==1.26.4", "h5py", "pandas", "pandocfilters", "requests", "scipy", "SuperSuit", "tabulate", "tqdm", "Flask", "Flask-SocketIO", "matplotlib", "pytest", "termcolor")
    .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}))

image = image.pip_install("psutil==7.0.0")

@app.function(image=image, gpu="T4", cpu=(16,16), memory=(16384,24576),
              timeout=3900, startup_timeout=180, retries=0,
              volumes={"/runs": volume}, max_containers=1, scaledown_window=2)
def train(source: bytes, source_sha256: str):
    import ast, io, json, os, signal, subprocess, sys, time, zipfile
    # Modal sets math thread counts from the container CPU allocation.
    # Override at runtime before torch import and before spawning Ray workers.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ORT_INTER_OP_NUM_THREADS",
                "ORT_INTRA_OP_NUM_THREADS"):
        os.environ[key] = "1"
    import psutil, torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path("/workspace/alpha_holdem")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        archive.extractall(root)
    os.chdir(root)
    persistent = Path("/runs") / RUN_ID
    if (persistent / 'run_metadata.json').exists():
        raise ValueError('RUN_ID already exists; choose a unique run ID')
    persistent.mkdir(parents=True, exist_ok=True)
    resume_directory = Path('/runs') / PREVIOUS_RUN / 'league'
    if not (resume_directory / 'latest_checkpoint.json').exists():
        raise ValueError('Previous full checkpoint is missing')
    (root / "work").symlink_to(persistent, target_is_directory=True)
    config = ast.literal_eval(Path("confs/nl_holdem.py").read_text())
    config.update(seed=42, rollout_fragment_length=200, train_batch_size=4000,
                  minibatch_size=4000, num_env_runners=14,
                  num_envs_per_env_runner=1, num_cpus_for_main_process=1,
                  min_time_s_per_iteration=10,
                  metrics_episode_collection_timeout_s=1,
                  learner_queue_size=3, num_gpu_loader_threads=2, learner_queue_timeout=600)
    Path("work/modal_config.py").write_text(repr(config))
    command = [sys.executable, "train_league.py", "--conf", "work/modal_config.py",
               "--device", "cuda", "--workers", "14", "--batch-size", "4000",
               "--training-seconds", "3600", "--checkpoint-steps", "50000",
               "--eval-steps", "0", "--league_tracker_n", "10000",
               "--output_dir", "work/league", "--experiment_name", "t4_one_hour"]
    command += ['--restore-state', str(resume_directory)]
    metadata = {'resume_from': PREVIOUS_RUN,"run_id": RUN_ID, "gpu": torch.cuda.get_device_name(0),
                "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "training_seconds": 3600, "cpu_cores":16,
                "memory_request_mib":16384, "memory_limit_mib":24576,
                "source_sha256":source_sha256, "command":command,
                "started_at_unix":time.time(),
                "max_compute_estimate_usd":3900*(.000164+16*.0000131+24*.00000222)}
    Path("work/source.zip").write_bytes(source)
    Path("work/run_metadata.json").write_text(json.dumps(metadata,indent=2))
    volume.commit()
    print(json.dumps(metadata), flush=True)
    started=time.monotonic()
    with Path("work/training.log").open("w") as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        previous_cpu=0.; previous_time=started; last_commit=started
        while process.poll() is None:
            time.sleep(30)
            now=time.monotonic()
            row={"elapsed_seconds":now-started,"time_unix":time.time()}
            try:
                procs=[psutil.Process(process.pid)]
                procs+=procs[0].children(recursive=True)
                cpu=0.; rss=0
                for p in procs:
                    try:
                        t=p.cpu_times();cpu+=t.user+t.system;rss+=p.memory_info().rss
                    except psutil.Error: pass
                row.update(process_cpu_cores=max(0,(cpu-previous_cpu)/(now-previous_time)),
                           summed_process_rss_gib=rss/1024**3)
                previous_cpu=cpu; previous_time=now
            except psutil.Error: pass
            try:
                row['container_memory_gib']=int(Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text())/1024**3
            except OSError: pass
            probe=subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True)
            row["gpu_util_memoryutil_usedMiB_totalMiB_watts"]=probe.stdout.strip()
            result=Path("work/ray_results/t4_one_hour/result.json")
            if result.exists():
                try:
                    last=json.loads(result.read_text().splitlines()[-1])
                    row.update(iteration=last.get('training_iteration'),
                        trained_transitions=last.get('num_env_steps_trained'),
                        sampled_transitions=last.get('num_env_steps_sampled'),
                        updates=last.get('info',{}).get('learner',{}).get('default_policy',{}).get('num_grad_updates_lifetime'))
                except (IndexError,ValueError): pass
            with Path("work/resources.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
            if now-last_commit>=120:
                volume.commit();last_commit=now
            if now-started>=3720 and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                metadata['watchdog_stop']=True
                break
        metadata.update(returncode=process.returncode,finished_at_unix=time.time(),
                        elapsed_seconds=time.monotonic()-started)
    Path("work/run_metadata.json").write_text(json.dumps(metadata,indent=2))
    volume.commit()
    print("FINISHED "+json.dumps(metadata),flush=True)
    return metadata

@app.function(image=image, cpu=(4,4), memory=(8192,8192), timeout=900,
              retries=0, volumes={"/runs":volume}, max_containers=1, scaledown_window=2)
def evaluate(source: bytes):
    import io, os, subprocess, sys, zipfile, json
    volume.reload()
    root = Path('/workspace/alpha_holdem')
    root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(source)) as archive: archive.extractall(root)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS'):
        os.environ[key]='1'
    output=Path('/runs')/RUN_ID/'analysis'
    output.mkdir(exist_ok=True)
    command=[sys.executable,str(root/'tools/evaluate_training_run.py'),
             '--run-dir',str(Path('/runs')/RUN_ID),
             '--previous-dir',str(Path('/runs')/PREVIOUS_RUN),
             '--hands','2400','--jobs','3']
    with (output/'evaluation.log').open('w') as log:
        process=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=840)
    volume.commit()
    if process.returncode: raise RuntimeError('Modal evaluation failed; inspect analysis/evaluation.log')
    print((output/'report.md').read_text(),flush=True)
    return {'run_id':RUN_ID,'report':str(output/'report.md')}

@app.function(image=modal.Image.debian_slim(python_version='3.11'),
              cpu=(0.125,0.125),memory=(256,256),timeout=4900,retries=0,max_containers=1)
def workflow(source: bytes, digest: str):
    metadata=train.remote(source,digest)
    if metadata['returncode']!=0:
        raise RuntimeError('Training failed; evaluation not started')
    return evaluate.remote(source)

@app.local_entrypoint()
def main(evaluate_only: bool = False):
    import hashlib, io, json, zipfile
    root=Path(__file__).resolve().parents[1]
    (root/"work").mkdir(exist_ok=True)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as archive:
        for folder in ["agi","rlcard","confs"]:
            for path in (root/folder).rglob("*"):
                if path.is_file() and path.suffix in (".py", ".json", ".txt", ".csv", ".npy", ".npz") and "__pycache__" not in path.parts:
                    archive.write(path,path.relative_to(root))
        archive.write(root/"tools/evaluate_training_run.py","tools/evaluate_training_run.py")
        for name in ["train_league.py","utils.py"]:archive.write(root/name,name)
    source=buffer.getvalue();digest=hashlib.sha256(source).hexdigest()
    call=evaluate.spawn(source) if evaluate_only else workflow.spawn(source,digest)
    launch={"run_id":RUN_ID,"call_id":call.object_id,"source_sha256":digest,
            "volume":"alpha-holdem-budget-results"}
    (root/"work/t4_second_hour_launch.json").write_text(json.dumps(launch,indent=2))
    print("LAUNCHED "+json.dumps(launch),flush=True)
