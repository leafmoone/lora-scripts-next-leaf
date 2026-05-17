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

# Start train status monitor server (port 6008) in background
Start-Process -NoNewWindow python -ArgumentList "train_status_server.py"

python gui.py