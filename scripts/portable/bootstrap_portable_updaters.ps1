param(
    [Parameter(Mandatory = $true)]
    [string]$PortableRoot,
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

. (Join-Path $PSScriptRoot "portable_updater_common.ps1")

$PortableRoot = $PortableRoot.TrimEnd('\')
$trainerDir = Join-Path $PortableRoot "SD-Trainer"
$updated = $false

function Get-FileSha256([string]$Path) {
    if (-not (Test-Path $Path)) { return "" }
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

if (-not $SkipDownload) {
    Write-Host "Checking latest updater scripts on GitHub / 检查 GitHub 最新更新脚本..."
    $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("sd-trainer-updater-bootstrap-" + [guid]::NewGuid().ToString("n"))
    New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
    try {
        foreach ($item in (Get-PortableUpdaterManifest)) {
            $tempFile = Join-Path $tempRoot (($item.Src -replace '[/\\]', '_'))
            try {
                Invoke-PortableRawDownload -RelativePath $item.Src -Destination $tempFile | Out-Null
            } catch {
                Write-Host "  [skip] $($item.Src) ($($_.Exception.Message))"
                continue
            }
            $dest = Join-Path $PortableRoot ($item.Dest -replace '/', '\')
            $destDir = Split-Path $dest -Parent
            if ($destDir -and -not (Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null
            }
            $oldHash = Get-FileSha256 $dest
            $newHash = Get-FileSha256 $tempFile
            if ($oldHash -ne $newHash) {
                Copy-Item $tempFile $dest -Force
                Write-Host "  [updated] $($item.Dest)"
                $updated = $true
            }
        }
    } finally {
        if (Test-Path $tempRoot) { Remove-Item $tempRoot -Recurse -Force -ErrorAction SilentlyContinue }
    }
    if ($updated) {
        Write-Host "Updater scripts synced from GitHub main / 已从 GitHub main 同步更新脚本。"
        Write-Host ""
    } else {
        Write-Host "Updater scripts already current / 更新脚本已是最新。"
        Write-Host ""
    }
}

if ($updated) {
    exit 10
}
exit 0
