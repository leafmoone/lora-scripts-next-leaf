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

# Auto-install flash-attn only when triton stack works (source/venv install; not portable embedded)
python -c "import triton; import flash_attn; from flash_attn.ops.triton.rotary import apply_rotary" 2>$null
if ($LASTEXITCODE -ne 0) {
    python -c "import triton" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host -ForegroundColor Cyan "Installing triton-windows..."
        pip install "triton-windows<3.4" 2>$null
    }
    python -c "import triton; import flash_attn; from flash_attn.ops.triton.rotary import apply_rotary" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host -ForegroundColor Cyan "Installing Flash Attention 2 (prebuilt wheel)..."
        $whl = "flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl"
        $url = "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/$whl"
        pip install $url 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host -ForegroundColor Green "Flash Attention 2 installed successfully"
        } else {
            Write-Host -ForegroundColor Yellow "Flash Attention 2 install failed (non-fatal, using PyTorch SDPA)"
        }
    }
}

python gui.py