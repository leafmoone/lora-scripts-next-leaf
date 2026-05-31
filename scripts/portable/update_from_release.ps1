# Sync portable SD-Trainer from the latest GitHub Release 7z (keeps user data).
param(
    [Parameter(Mandatory = $true)]
    [string]$PortableRoot,
    [switch]$DryRun,
    [string]$Tag = "",
    [string]$AssetName = ""
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Step([string]$Message) {
    Write-Host $Message
}

function Resolve-SevenZip {
    $cmd = Get-Command 7z -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        "${env:ProgramFiles}\7-Zip\7z.exe",
        "${env:ProgramFiles(x86)}\7-Zip\7z.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) { return $path }
    }
    return $null
}

function Invoke-Download([string]$Url, [string]$Destination) {
    $curl = Get-Command curl -ErrorAction SilentlyContinue
    if (-not $curl) {
        throw "curl not found. Install curl or use Git update (Update-SD-Trainer.bat)."
    }
    if (Test-Path $Destination) { Remove-Item $Destination -Force }
    & curl.exe -fL --retry 3 --retry-delay 2 -o $Destination $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $Url"
    }
}

function Get-ReleaseAsset {
    param(
        [string]$Repository,
        [string]$TagName = "",
        [string]$PreferredAssetName = ""
    )
    $headers = @{ "User-Agent" = "SD-Trainer-Portable-Updater" }
    if ($TagName) {
        $uri = "https://api.github.com/repos/$Repository/releases/tags/$TagName"
    } else {
        $uri = "https://api.github.com/repos/$Repository/releases/latest"
    }
    $release = Invoke-RestMethod -Uri $uri -Headers $headers
    $assets = @($release.assets | Where-Object { $_.name -like "SD-Trainer-v*.7z" })
    if ($PreferredAssetName) {
        $match = $assets | Where-Object { $_.name -eq $PreferredAssetName } | Select-Object -First 1
        if ($match) { return $match }
    }
    $asset = $assets | Sort-Object { $_.name } -Descending | Select-Object -First 1
    if (-not $asset) {
        throw "No SD-Trainer-v*.7z asset found in release $($release.tag_name)."
    }
    return $asset
}

$PortableRoot = (Resolve-Path $PortableRoot).Path.TrimEnd('\')
$TrainerDir = Join-Path $PortableRoot "SD-Trainer"
if (-not (Test-Path (Join-Path $TrainerDir "gui.py"))) {
    throw "SD-Trainer not found under: $PortableRoot"
}

$repo = "wochenlong/lora-scripts-next"
$asset = Get-ReleaseAsset -Repository $repo -TagName $Tag -PreferredAssetName $AssetName
$currentVersion = ""
$versionFile = Join-Path $TrainerDir "VERSION"
if (Test-Path $versionFile) {
    $currentVersion = (Get-Content $versionFile -TotalCount 1).Trim()
}

Write-Step "Release tag / 发布标签: $($asset.name -replace '\.7z$','' -replace '^SD-Trainer-v','v')"
Write-Step "Current VERSION / 当前版本: $(if ($currentVersion) { $currentVersion } else { '(unknown)' })"
Write-Step "Asset URL: $($asset.browser_download_url)"

if ($DryRun) {
    $sevenZip = Resolve-SevenZip
    Write-Step "[DryRun] 7-Zip: $(if ($sevenZip) { $sevenZip } else { 'NOT FOUND' })"
    Write-Step "[DryRun] OK — release metadata reachable."
    exit 0
}

$sevenZip = Resolve-SevenZip
if (-not $sevenZip) {
    throw "7-Zip not found. Install 7-Zip or add 7z.exe to PATH."
}

$cacheDir = Join-Path $PortableRoot "update\.cache"
New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
$archivePath = Join-Path $cacheDir $asset.name
$extractDir = Join-Path $cacheDir ("extract-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
$stagingRoot = Join-Path $extractDir "package"

$mirrors = @(
    $asset.browser_download_url,
    "https://ghfast.top/$($asset.browser_download_url)",
    "https://mirror.ghproxy.com/$($asset.browser_download_url)"
)

Write-Step ""
Write-Step "Downloading / 下载整合包..."
$downloaded = $false
foreach ($url in $mirrors) {
    Write-Step "  Try: $url"
    try {
        Invoke-Download -Url $url -Destination $archivePath
        $downloaded = $true
        Write-Step "  OK"
        break
    } catch {
        Write-Step "  Failed: $($_.Exception.Message)"
    }
}
if (-not $downloaded) {
    throw "All download mirrors failed."
}

Write-Step ""
Write-Step "Extracting / 解压..."
New-Item -ItemType Directory -Path $stagingRoot -Force | Out-Null
& $sevenZip x $archivePath "-o$stagingRoot" -y | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "7z extract failed."
}

$stagingTrainer = Join-Path $stagingRoot "SD-Trainer"
if (-not (Test-Path (Join-Path $stagingTrainer "gui.py"))) {
    # Some archives may unpack with a single top folder.
    $nested = Get-ChildItem $stagingRoot -Directory | Select-Object -First 1
    if ($nested -and (Test-Path (Join-Path $nested.FullName "SD-Trainer\gui.py"))) {
        $stagingRoot = $nested.FullName
        $stagingTrainer = Join-Path $stagingRoot "SD-Trainer"
    } else {
        throw "Extracted package missing SD-Trainer\gui.py"
    }
}

Write-Step ""
Write-Step "Merging SD-Trainer / 合并项目文件（保留用户数据）..."
$robocopyArgs = @(
    $stagingTrainer,
    $TrainerDir,
    "/E", "/XO", "/R:2", "/W:2", "/NFL", "/NDL", "/NJH", "/NJS", "/NC", "/NS",
    "/XD", "extensions", ".cache", "__pycache__", "node_modules", ".vscode", ".cursor",
    "/XD", "config\autosave", "output", "logs"
)
& robocopy @robocopyArgs | Out-Null
if ($LASTEXITCODE -ge 8) {
    throw "robocopy failed with exit code $LASTEXITCODE"
}

Write-Step ""
Write-Step "Refreshing root launchers / 刷新根目录启动脚本..."
$rootFiles = @(
    "run_gui.bat",
    "run_gui_portable.bat",
    "Update-SD-Trainer.bat",
    "Update-SD-Trainer-Release.bat",
    "Download-Anima-Model.bat",
    "install_xformers.bat"
)
foreach ($name in $rootFiles) {
    $src = Join-Path $stagingRoot $name
    if (Test-Path $src) {
        Copy-Item $src -Destination (Join-Path $PortableRoot $name) -Force
        Write-Step "  Updated: $name"
    }
}

if (Test-Path (Join-Path $stagingRoot "update")) {
    $destUpdate = Join-Path $PortableRoot "update"
    New-Item -ItemType Directory -Path $destUpdate -Force | Out-Null
    Copy-Item (Join-Path $stagingRoot "update\*") $destUpdate -Recurse -Force
}

$newVersion = ""
if (Test-Path (Join-Path $stagingTrainer "VERSION")) {
    $newVersion = (Get-Content (Join-Path $stagingTrainer "VERSION") -TotalCount 1).Trim()
}

Write-Step ""
Write-Step "========================================"
Write-Step "  Done / 更新完成"
Write-Step "========================================"
if ($newVersion) {
    Write-Step "  New VERSION / 新版本: $newVersion"
}
Write-Step ""
Write-Step "Preserved / 已保留:"
Write-Step "  sd-models\  output\  logs\  huggingface\  tagger-models\"
Write-Step "  SD-Trainer\extensions\  (Anima Fast plugin, if installed)"
Write-Step ""
if ($newVersion -and $currentVersion -and $newVersion -ne $currentVersion) {
    Write-Step "If WebUI fails to start, run update\update_dependencies.bat"
    Write-Step "WebUI startup failed? Run update\update_dependencies.bat to sync deps."
}

exit 0
