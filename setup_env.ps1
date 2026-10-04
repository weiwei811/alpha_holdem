param([ValidateSet('auto','cpu','cuda')][string]$Device = 'auto')
$ErrorActionPreference = 'Stop'
py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 environment creation failed' }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
if ($Device -eq 'auto') {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi --query-gpu=name --format=csv,noheader > $null
        $Device = if ($LASTEXITCODE -eq 0) { 'cuda' } else { 'cpu' }
    } else { $Device = 'cpu' }
}
$taskTorchVersion = & .\.venv\Scripts\python.exe -c 'from importlib.metadata import version; print(version("torch"))'
if ($Device -eq 'cuda' -and $taskTorchVersion -ne '2.8.0+cu126') {
    .\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
}
if ($Device -eq 'cpu' -and $taskTorchVersion -ne '2.8.0+cpu') {
    .\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed' }
}
.\.venv\Scripts\python.exe -c "from agi.devices import resolve_device; print('Selected device:', resolve_device('$Device'))"
if ($LASTEXITCODE -ne 0) { throw 'Device validation failed' }
