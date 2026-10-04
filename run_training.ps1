$ErrorActionPreference = 'Stop'
$taskExitCode = 0
Push-Location -LiteralPath $PSScriptRoot
try {
    & .\.venv\Scripts\python.exe train_league.py @args
    $taskExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $taskExitCode
