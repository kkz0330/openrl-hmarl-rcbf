param(
    [string]$BaseDir = "C:\Users\shirosk\Documents\New project\tqsdk_codex_start_prep",
    [switch]$RecreateVenv
)

$ErrorActionPreference = "Stop"

$repoDir = Join-Path $BaseDir "repos\tqsdk-python"
$venvDir = Join-Path $BaseDir ".venv"
$pyExe = Join-Path $venvDir "Scripts\python.exe"
$tmpDir = Join-Path $BaseDir "tmp"

if (!(Test-Path $repoDir)) {
    git clone https://gitee.com/tianqin_quantification_tqsdk/tqsdk-python.git $repoDir
}

if ($RecreateVenv -and (Test-Path $venvDir)) {
    Remove-Item -Recurse -Force $venvDir
}

if (!(Test-Path $pyExe)) {
    python -m venv $venvDir
}

New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
$env:TEMP = $tmpDir
$env:TMP = $tmpDir

& $pyExe -m ensurepip --upgrade
& $pyExe -m pip install -r (Join-Path $repoDir "requirements.txt")
& $pyExe -m pip install -e $repoDir

Write-Host "Setup finished."
