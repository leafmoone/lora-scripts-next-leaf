$Env:HF_HOME = "huggingface"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
$Env:PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"

function InstallFail {
    Write-Output "瀹夎澶辫触銆?
    Read-Host | Out-Null
    Exit
}

function Check {
    param (
        $ErrorInfo
    )
    if (!($?)) {
        Write-Output $ErrorInfo
        InstallFail
    }
}

if (Test-Path -Path "python\python.exe") {
    Write-Output "浣跨敤 python 鏂囦欢澶逛腑鐨?python..."
    $py_path = (Get-Item "python").FullName
    $env:PATH = "$py_path;$env:PATH"
}
else {
    # Sync vendor/sd-scripts submodule (Anima training engine)
    if ((Test-Path -Path ".git") -or (Test-Path -Path ".git" -PathType Leaf)) {
        Write-Output "鍚屾 git 瀛愭ā鍧?(vendor/sd-scripts)..."
        git submodule update --init --recursive
        if ($LASTEXITCODE -ne 0) {
            Write-Output "璀﹀憡: 瀛愭ā鍧楀垵濮嬪寲澶辫触锛孉nima 璁粌鍙兘鏃犳硶鍚姩銆傝鎵嬪姩杩愯: git submodule update --init --recursive"
        }
    }

    if (!(Test-Path -Path "venv")) {
        Write-Output "姝ｅ湪鍒涘缓铏氭嫙鐜..."
        python -m venv venv
        Check "鍒涘缓铏氭嫙鐜澶辫触锛岃妫€鏌?python 鏄惁瀹夎姝ｇ‘浠ュ強 python 鐗堟湰鏄惁涓?64 浣嶇増鏈?(python 3.10)锛宲ython 鐨勭洰褰曟槸鍚﹀湪鐜鍙橀噺 PATH 涓€?
    }

    Write-Output "妫€娴嬪埌铏氭嫙鐜锛屾鍦ㄦ縺娲?.."
    .\venv\Scripts\activate
    Check "婵€娲昏櫄鎷熺幆澧冨け璐ャ€?
}

Write-Output "瀹夎璁粌渚濊禆 (宸茶繘琛屽浗鍐呭姞閫燂紝濡傚湪鍥藉鏃犳硶浣跨敤鍔犻€熸簮璇锋崲鐢?install.ps1 鑴氭湰)"
Write-Output "娉ㄦ剰锛氬湪鍥藉唴鍔犻€熼暅鍍忎腑 torch 瀹夎鏃犳硶浣跨敤闀滃儚婧愶紝瀹夎杈冧负缂撴參銆?
$install_torch = Read-Host "鏄惁闇€瑕佸畨瑁?Torch+xformers? [y/n] (榛樿涓?y)"
if ($install_torch -eq "y" -or $install_torch -eq "Y" -or $install_torch -eq "") {
    python -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128
    Check "torch 瀹夎澶辫触锛岃鍒犻櫎 venv 鏂囦欢澶瑰悗閲嶆柊杩愯銆?
    python -m pip install -U -I --no-deps xformers===0.0.30 --extra-index-url https://download.pytorch.org/whl/cu128
    Check "xformers 瀹夎澶辫触銆?
}

python -m pip install --upgrade -r requirements.txt
Check "璁粌渚濊禆搴撳畨瑁呭け璐ャ€?


Write-Output "Installing Flash Attention 2 (prebuilt wheel)..."
$pyver = python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')" 2>$null
if ($pyver -match "^cp3(10|11|12)$") {
    $whl = "flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-$pyver-$pyver-win_amd64.whl"
    $url = "https://hf-mirror.com/lldacing/flash-attention-windows-wheel/resolve/main/$whl"
    python -m pip install $url 2>$null
} else {
    python -m pip install flash-attn --no-build-isolation 2>$null
}
if ($LASTEXITCODE -eq 0) {
    Write-Output "Flash Attention 2 installed"
} else {
    Write-Output "Flash Attention 2 install failed (non-fatal)"
}

Write-Output "瀹夎瀹屾垚"
Read-Host | Out-Null
