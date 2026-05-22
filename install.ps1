$Env:HF_HOME = "huggingface"

. "$PSScriptRoot\install_preflight.ps1"

function InstallFail {
    Write-Output "Install failed."
    Read-Host | Out-Null
    Exit 1
}

if (-not (Test-InstallScriptFreshness)) { InstallFail }
if (-not (Test-InstallPython)) { InstallFail }

# Ensure the pinned vendor/sd-scripts submodule (Anima training engine) is
# present. Safe to run repeatedly; skips silently when not a git checkout.
if ((Test-Path -Path ".git") -or (Test-Path -Path ".git" -PathType Leaf)) {
    Write-Output "Syncing git submodules (vendor/sd-scripts)..."
    git submodule update --init --recursive
    if ($LASTEXITCODE -ne 0) {
        Write-Output "Warning: submodule init failed; Anima training may not start. Run 'git submodule update --init --recursive' manually."
    }
}

if (!(Test-Path -Path "venv")) {
    Write-Output  "Creating venv for python..."
    python -m venv venv
}
.\venv\Scripts\activate

Write-Output "Installing deps..."

pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { Write-Output "torch install failed. Delete venv and retry."; InstallFail }
pip install -U -I --no-deps xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { Write-Output "xformers install failed."; InstallFail }
pip install --upgrade -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Output "requirements install failed."; InstallFail }

Write-Output "Prefetching default WD tagger wd14-convnextv2-v2 (~388 MB)..."
python scripts/prefetch_default_tagger.py --if-missing --no-mirror
if ($LASTEXITCODE -ne 0) {
    Write-Output "Warning: default tagger prefetch failed; it will download on first tag run."
}

Write-Output "Install completed"
Write-Output ""
Write-Output "Optional: run install_flash_attn.bat to enable Flash Attention 2 acceleration."
Read-Host | Out-Null
