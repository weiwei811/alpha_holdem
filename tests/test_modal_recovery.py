"""Exercise the real Modal function body without allocating cloud hardware."""
import ast
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
import pytest


def load_train(tmp_path, monkeypatch):
    import torch, subprocess, sys
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace())
    tree = ast.parse(Path('tools/modal_t4_one_hour.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    function.decorator_list = []
    root = tmp_path / 'workspace'; runs = tmp_path / 'runs'; root.mkdir(); runs.mkdir()
    def cloud_path(value):
        value = str(value)
        if value.startswith('/workspace/alpha_holdem'):
            return root / value.removeprefix('/workspace/alpha_holdem').lstrip('/')
        if value.startswith('/runs'):
            return runs / value.removeprefix('/runs').lstrip('/')
        if value == 'work' or value.startswith('work/'):
            return runs / 'run' / value.removeprefix('work').lstrip('/')
        return Path(value)
    monkeypatch.setattr(Path, 'symlink_to', lambda *a, **k: None)
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: None)
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda n: None)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda n: 'Fake T4')
    commands = []
    def popen(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(poll=lambda: 0, returncode=0)
    monkeypatch.setattr(subprocess, 'Popen', popen)
    monkeypatch.chdir(tmp_path)
    namespace = {'Path': cloud_path, 'volume': SimpleNamespace(commit=lambda: None)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual-modal-function>', 'exec'), namespace)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        z.writestr('confs/test.py', repr({'model': {'custom_model': 'NlHoldemStructuredNet'}}))
    return namespace['train'], buffer.getvalue(), runs / 'run', commands


def test_same_invocation_restarts_existing_run_from_checkpoint(tmp_path, monkeypatch):
    train, source, run, commands = load_train(tmp_path, monkeypatch)
    run.mkdir(); (run / 'league').mkdir()
    (run / 'run_metadata.json').write_text(json.dumps({'source_sha256': 'digest', 'config': 'confs/test.py', 'started_at_unix': 1}))
    (run / 'source.zip').write_bytes(source)
    (run / 'resources.jsonl').write_text(json.dumps({'elapsed_seconds': 2940}) + '\n')
    (run / 'league/latest_checkpoint.json').write_text(json.dumps({'path': 'checkpoints/saved', 'trained_steps': 1084000}))
    train(source, 'digest', 'run', 'confs/test.py', 10000)
    command = commands[0]
    assert command[command.index('--restore-state') + 1] == 'work/league'
    assert 0 < float(command[command.index('--training-seconds') + 1]) <= 660
    assert (run / 'league/training_config.json').exists()


def test_unrelated_launch_cannot_overwrite_run(tmp_path, monkeypatch):
    train, source, run, commands = load_train(tmp_path, monkeypatch)
    run.mkdir()
    (run / 'run_metadata.json').write_text(json.dumps({'launch_token': 'original', 'config': 'confs/test.py'}))
    with pytest.raises(ValueError, match='another launch'):
        train(source, 'digest', 'run', 'confs/test.py', 10000, 'unrelated')
    assert not commands


def test_repeated_preemption_consumes_cumulative_budget(tmp_path, monkeypatch):
    train, source, run, commands = load_train(tmp_path, monkeypatch)
    run.mkdir(); (run / 'league').mkdir()
    (run / 'run_metadata.json').write_text(json.dumps({'source_sha256': 'digest', 'config': 'confs/test.py', 'budget_used_seconds': 3550}))
    (run / 'source.zip').write_bytes(source)
    (run / 'league/latest_checkpoint.json').write_text('{}')
    result = train(source, 'digest', 'run', 'confs/test.py', 10000)
    assert result['budget_exhausted'] and not commands


def test_explicit_recovery_uses_original_training_source(tmp_path, monkeypatch):
    train, source, run, commands = load_train(tmp_path, monkeypatch)
    run.mkdir(); (run / 'league').mkdir()
    (run / 'run_metadata.json').write_text(json.dumps({'source_sha256': 'original', 'config': 'confs/test.py', 'launch_token': 'old', 'budget_used_seconds': 3000}))
    (run / 'source.zip').write_bytes(source)
    (run / 'league/latest_checkpoint.json').write_text('{}')
    result = train(b'new source ignored', 'new', 'run', 'confs/test.py', 10000, 'new-token', True)
    assert result['source_sha256'] == 'original'
    assert result['remaining_training_seconds'] == 450
    assert '--restore-state' in commands[0]


def test_continuation_restores_prior_full_state_and_saved_config(tmp_path, monkeypatch):
    train, source, run, commands = load_train(tmp_path, monkeypatch)
    prior = run.parent / 'prior' / 'league'; prior.mkdir(parents=True)
    original = {'model': {'custom_model': 'NlHoldemStructuredNet'}, 'lr': 0.0003, 'entropy_coeff': 0.1, 'env_config': {'custom_options': {'betting_features': True}}}
    (prior / 'training_config.json').write_text(json.dumps(original))
    (prior / 'latest_checkpoint.json').write_text(json.dumps({'trained_steps': 1284000, 'path': 'checkpoints/final'}))
    result = train(source, 'digest', 'run', 'confs/test.py', 10000, 'new', False, 'prior')
    command = commands[0]
    assert command[command.index('--restore-state') + 1].replace('\\', '/').endswith('/prior/league')
    saved = json.loads((run / 'league/training_config.json').read_text())
    assert saved['env_config'] == original['env_config'] and saved['lr'] == original['lr']
    assert result['remaining_training_seconds'] == 3600


def test_selected_candidate_flags_are_carried_into_full_continuation(tmp_path, monkeypatch):
    train, source, run, commands = load_train(tmp_path, monkeypatch)
    prior=run.parent/'prior'/'selected_resume';prior.mkdir(parents=True)
    (prior/'training_config.json').write_text(json.dumps({'model':{'custom_model':'NlHoldemStructuredNet'},'entropy_coeff':.1}))
    (prior/'latest_checkpoint.json').write_text('{}')
    result=train(source,'digest','run','confs/test.py',10000,'new',False,'prior','selected_resume','uniform',.25,.02)
    command=commands[0]
    assert command[command.index('--entropy-coeff')+1]=='0.02'
    assert command[command.index('--opponent-sampling')+1]=='uniform'
    assert command[command.index('--exg_oppo_prob')+1]=='0.25'
    assert json.loads((run/'league/training_config.json').read_text())['entropy_coeff']==.02
    assert result['resume_subdir']=='selected_resume'
