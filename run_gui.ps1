$Env:HF_HOME = "huggingface"
$Env:PYTHONUTF8 = "1"
$Env:MIKAZUKI_PORT = "28000"


if (Test-Path -Path "venv\Scripts\activate") {
    Write-Host -ForegroundColor green "Activating virtual environment..."
    .\venv\Scripts\activate
}
elseif (Test-Path -Path "python\python.exe") {
    Write-Host -ForegroundColor green "Using python from python folder..."
    $py_path = (Get-Item "python").FullName
    $env:PATH = "$py_path;$env:PATH"
}
else {
    Write-Host -ForegroundColor Blue "No virtual environment found, using system python..."
}

# Auto-install flash-attn if missing (one-time, non-blocking)
python -c "import flash_attn" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host -ForegroundColor Cyan "Installing Flash Attention 2 for training acceleration..."
    pip install flash-attn --no-build-isolation 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host -ForegroundColor Green "Flash Attention 2 installed successfully"
    } else {
        Write-Host -ForegroundColor Yellow "Flash Attention 2 install failed (non-fatal, using PyTorch SDPA)"
    }
}

python gui.py