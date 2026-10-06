import modal
from pathlib import Path
app = modal.App("alpha-holdem-six-player-t4-one-hour")
volume = modal.Volume.from_name("alpha-holdem-budget-results", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("ray[rllib]==2.49.2", "numpy==1.26.4", "h5py", "pandas", "pandocfilters", "requests", "scipy", "SuperSuit", "tabulate", "tqdm", "Flask", "Flask-SocketIO", "matplotlib", "pytest", "termcolor")
    .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}))

image = image.pip_install("psutil==7.0.0")

@app.function(image=image, gpu="T4", cpu=(16,16), memory=(16384,24576),
              timeout=3900, startup_timeout=180, retries=0,
              volumes={"/runs": volume}, max_containers=1, scaledown_window=2)
def train(source: bytes, source_sha256: str, run_id: str, conf_path: str, checkpoint_steps: int, launch_token: str = "", recover: bool = False, resume_from: str = "", resume_subdir: str = "league", opponent_sampling: str = "", opponent_refresh: float = -1, entropy_coeff: float = -1, freeze_opponents: bool = False):
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
    persistent = Path("/runs") / run_id
    existing = persistent.exists()
    budget_used = 0.0
    previous = {}
    if existing:
        metadata_path = persistent / 'run_metadata.json'
        if not metadata_path.exists():
            raise ValueError('Existing run has no recovery metadata')
        previous = json.loads(metadata_path.read_text())
        if not recover and previous.get('launch_token', '') != launch_token:
            raise ValueError('RUN_ID belongs to another launch; use --recover after checking it has stopped')
        settings={'freeze_opponents':freeze_opponents,'resume_subdir':resume_subdir,'opponent_sampling':opponent_sampling,'opponent_refresh':opponent_refresh,'entropy_coeff_override':entropy_coeff}
        for key,value in settings.items():
            if previous.get(key, False if key=='freeze_opponents' else 'league' if key=='resume_subdir' else '' if key=='opponent_sampling' else -1)!=value:
                raise ValueError('Recovery runtime settings must match the original launch')
        if previous.get('resume_from', '') != resume_from:
            raise ValueError('Recovery must preserve the original resume source')
        if previous.get('config') != conf_path:
            raise ValueError('Recovery must use the original config')
        if previous.get('returncode') == 0:
            return previous
        if not recover and previous.get('source_sha256') != source_sha256:
            raise ValueError('Automatic restart source does not match saved source')
        source = (persistent / 'source.zip').read_bytes()
        source_sha256 = previous['source_sha256']
        budget_used = float(previous.get('budget_used_seconds', 0))
        resource_path = persistent / 'resources.jsonl'
        if resource_path.exists():
            for line in resource_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    budget_used = max(budget_used, float(row.get('cumulative_elapsed_seconds', row.get('elapsed_seconds', 0))))
                except (ValueError, TypeError):
                    pass  # A preempted write may leave the last line incomplete.
        # Charge the uncommitted tail, initialization and recovery conservatively.
        budget_used += 150
        if not (persistent / 'league/latest_checkpoint.json').exists():
            raise ValueError('No committed full checkpoint to recover; refusing to start over')
    remaining_seconds = max(0.0, 3600 - budget_used)
    if not remaining_seconds:
        previous.update(budget_exhausted=True, budget_used_seconds=budget_used)
        (persistent / 'run_metadata.json').write_text(json.dumps(previous, indent=2))
        volume.commit()
        return previous
    root = Path("/workspace/alpha_holdem")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        archive.extractall(root)
    os.chdir(root)
    persistent.mkdir(parents=True, exist_ok=True)
    (root / "work").symlink_to(persistent, target_is_directory=True)
    resume_directory = Path('/runs') / resume_from / resume_subdir if resume_from else None
    if resume_directory:
        if not (resume_directory / 'latest_checkpoint.json').exists():
            raise ValueError('Resume source lacks a full checkpoint')
        config = json.loads((resume_directory / 'training_config.json').read_text())
    else:
        config = ast.literal_eval(Path(conf_path).read_text())
    config.update(seed=42, rollout_fragment_length=200, train_batch_size=4000,
                  minibatch_size=4000, num_env_runners=14,
                  num_envs_per_env_runner=1, num_cpus_for_main_process=1,
                  min_time_s_per_iteration=10,
                  metrics_episode_collection_timeout_s=1,
                  learner_queue_size=3, num_gpu_loader_threads=2, learner_queue_timeout=600)
    Path("work/modal_config.py").write_text(repr(config))
    command = [sys.executable, "train_league.py", "--conf", "work/modal_config.py",
               "--device", "cuda", "--workers", "14", "--batch-size", "4000",
               "--training-seconds", str(remaining_seconds), "--checkpoint-steps", str(checkpoint_steps),
               "--eval-steps", "0", "--league_tracker_n", "10000",
               "--output_dir", "work/league", "--experiment_name", "t4_one_hour"]
    if freeze_opponents:
        command += ['--freeze-opponents']
    if opponent_sampling:
        command += ['--opponent-sampling',opponent_sampling]
    if opponent_refresh>=0:
        command += ['--exg_oppo_prob',str(opponent_refresh)]
    if entropy_coeff>=0:
        config['entropy_coeff']=entropy_coeff
        config['entropy_coeff_schedule']=None
        Path('work/modal_config.py').write_text(repr(config))
        command += ['--entropy-coeff',str(entropy_coeff)]
    if existing:
        command += ['--restore-state', 'work/league']
    elif resume_directory:
        command += ['--restore-state', str(resume_directory)]
    # Needed even when preemption occurs before train_league's normal exit.
    league = persistent / 'league'
    league.mkdir(exist_ok=True)
    saved_config = league / 'training_config.json'
    if not saved_config.exists():
        saved_config.write_text(json.dumps(config, indent=2))
    metadata = {"run_id": run_id, "resume_from":resume_from,"resume_subdir":resume_subdir,"freeze_opponents":freeze_opponents,"opponent_sampling":opponent_sampling,"opponent_refresh":opponent_refresh,"entropy_coeff_override":entropy_coeff, "config": conf_path, "model": config["model"], "gpu": torch.cuda.get_device_name(0),
                "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "training_seconds": 3600, "cpu_cores":16,
                "memory_request_mib":16384, "memory_limit_mib":24576,
                "source_sha256":source_sha256, "command":command,
                "started_at_unix":previous.get('started_at_unix', time.time()),
                "attempt_started_at_unix":time.time(), "launch_token":launch_token,
                "budget_used_seconds":budget_used, "remaining_training_seconds":remaining_seconds,
                "recovery_count":previous.get('recovery_count', 0) + int(existing),
                "max_compute_estimate_usd":3900*(.000164+16*.0000131+24*.00000222)}
    if not existing:
        Path("work/source.zip").write_bytes(source)
    Path("work/run_metadata.json").write_text(json.dumps(metadata,indent=2))
    volume.commit()
    print(json.dumps(metadata), flush=True)
    started=time.monotonic()
    with Path("work/training.log").open("a") as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        previous_cpu=0.; previous_time=started; last_commit=started; committed_checkpoint=None
        while process.poll() is None:
            time.sleep(30)
            now=time.monotonic()
            row={"elapsed_seconds":now-started,"cumulative_elapsed_seconds":budget_used+now-started,"time_unix":time.time()}
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
            checkpoint = persistent / 'league/latest_checkpoint.json'
            checkpoint_text = checkpoint.read_text() if checkpoint.exists() else None
            if now-last_commit>=120 or checkpoint_text != committed_checkpoint:
                metadata['budget_used_seconds'] = budget_used + now-started
                Path("work/run_metadata.json").write_text(json.dumps(metadata,indent=2))
                volume.commit();last_commit=now;committed_checkpoint=checkpoint_text
            watchdog_seconds = min(remaining_seconds + 120, 3900-budget_used-60)
            if now-started>=watchdog_seconds and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                metadata['watchdog_stop']=True
                break
        metadata.update(returncode=process.returncode,finished_at_unix=time.time(),
                        elapsed_seconds=time.monotonic()-started,
                        budget_used_seconds=budget_used+time.monotonic()-started)
    Path("work/run_metadata.json").write_text(json.dumps(metadata,indent=2))
    volume.commit()
    print("FINISHED "+json.dumps(metadata),flush=True)
    return metadata

@app.local_entrypoint()
def main(conf: str = "confs/nl_holdem.py", run_id: str = "", checkpoint_steps: int = 50000, recover: bool = False, resume_from: str = "", resume_subdir: str = "league", opponent_sampling: str = "", opponent_refresh: float = -1, entropy_coeff: float = -1, freeze_opponents: bool = False):
    import hashlib, io, json, zipfile, uuid
    root=Path(__file__).resolve().parents[1]
    import datetime
    run_id = run_id or ("six-player-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S") + "-t4-1h")
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("run-id must contain only letters, digits, hyphens and underscores")
    if resume_from and (resume_from == run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in resume_from)):
        raise ValueError('resume-from must be a different valid run ID')
    import math
    subdir_parts=resume_subdir.split('/')
    if not subdir_parts or any(not p or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in p) for p in subdir_parts):
        raise ValueError('resume-subdir must be a safe relative directory')
    if opponent_sampling not in ('','ranked','uniform') or (opponent_refresh != -1 and not 0<=opponent_refresh<=1):
        raise ValueError('Invalid opponent settings')
    if not math.isfinite(entropy_coeff) or (entropy_coeff != -1 and entropy_coeff<0):
        raise ValueError('Invalid entropy coefficient')
    config_path = Path(conf)
    if config_path.is_absolute() or ".." in config_path.parts or config_path.parts[0] != "confs" or not (root/config_path).is_file():
        raise ValueError("conf must name a config inside confs/")
    conf = config_path.as_posix()
    if checkpoint_steps <= 0:
        raise ValueError("checkpoint-steps must be positive")
    (root/"work").mkdir(exist_ok=True)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as archive:
        for folder in ["agi","rlcard","confs"]:
            for path in (root/folder).rglob("*"):
                if path.is_file() and path.suffix in (".py", ".json", ".txt", ".csv", ".npy", ".npz") and "__pycache__" not in path.parts:
                    archive.write(path,path.relative_to(root))
        for name in ["train_league.py","utils.py"]:archive.write(root/name,name)
    source=buffer.getvalue();digest=hashlib.sha256(source).hexdigest()
    launch_token = uuid.uuid4().hex
    call=train.spawn(source,digest,run_id,conf,checkpoint_steps,launch_token,recover,resume_from,resume_subdir,opponent_sampling,opponent_refresh,entropy_coeff,freeze_opponents)
    launch={"run_id":run_id,"resume_from":resume_from,"resume_subdir":resume_subdir,"freeze_opponents":freeze_opponents,"opponent_sampling":opponent_sampling,"opponent_refresh":opponent_refresh,"entropy_coeff":entropy_coeff,"config":conf,"recover":recover,"launch_token":launch_token,"call_id":call.object_id,"source_sha256":digest,
            "volume":"alpha-holdem-budget-results"}
    (root/"work/t4_one_hour_launch.json").write_text(json.dumps(launch,indent=2))
    print("LAUNCHED "+json.dumps(launch),flush=True)
