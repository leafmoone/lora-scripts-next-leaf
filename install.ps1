$Env:HF_HOME = "huggingface"

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

pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -U -I --no-deps xformers==0.0.30 --extra-index-url https://download.pytorch.org/whl/cu128
pip install --upgrade -r requirements.txt

Write-Output "Installing Flash Attention 2 (optional, for training acceleration)..."
pip install flash-attn --no-build-isolation 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Output "Flash Attention 2 installed successfully"
} else {
    Write-Output "Flash Attention 2 install failed (non-fatal, will use PyTorch SDPA instead)"
}

Write-Output "Install completed"
Read-Host | Out-Null ;
