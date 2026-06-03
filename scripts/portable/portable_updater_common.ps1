# Shared helpers for portable Git / Release updaters.
$script:PortableUpdaterRepo = "wochenlong/lora-scripts-next"
$script:PortableUpdaterBranch = "main"

function Get-PortableUpdaterManifest {
    @(
        @{ Src = "build-scripts/templates/Update-SD-Trainer.bat"; Dest = "Update-SD-Trainer.bat" },
        @{ Src = "build-scripts/templates/Update-SD-Trainer-Release.bat"; Dest = "Update-SD-Trainer-Release.bat" },
        @{ Src = "scripts/portable/update_from_release.ps1"; Dest = "SD-Trainer/scripts/portable/update_from_release.ps1" },
        @{ Src = "scripts/portable/bootstrap_portable_updaters.ps1"; Dest = "SD-Trainer/scripts/portable/bootstrap_portable_updaters.ps1" },
        @{ Src = "scripts/portable/show_portable_update_status.ps1"; Dest = "SD-Trainer/scripts/portable/show_portable_update_status.ps1" },
        @{ Src = "scripts/portable/portable_updater_common.ps1"; Dest = "SD-Trainer/scripts/portable/portable_updater_common.ps1" },
        @{ Src = "scripts/portable/sync_portable_root_launchers.bat"; Dest = "SD-Trainer/scripts/portable/sync_portable_root_launchers.bat" },
        @{ Src = "scripts/portable/UPDATER_VERSION"; Dest = "SD-Trainer/scripts/portable/UPDATER_VERSION" },
        @{ Src = "build-scripts/templates/Update-SD-Trainer.bat"; Dest = "SD-Trainer/scripts/portable/templates/Update-SD-Trainer.bat" },
        @{ Src = "build-scripts/templates/Update-SD-Trainer-Release.bat"; Dest = "SD-Trainer/scripts/portable/templates/Update-SD-Trainer-Release.bat" }
    )
}

function Get-RawGitHubUrls([string]$RelativePath) {
    $rel = $RelativePath -replace '\\', '/'
    $base = "https://raw.githubusercontent.com/$($script:PortableUpdaterRepo)/$($script:PortableUpdaterBranch)/$rel"
    @(
        $base,
        "https://ghfast.top/$base",
        "https://mirror.ghproxy.com/$base"
    )
}

function Invoke-PortableRawDownload {
    param(
        [string]$RelativePath,
        [string]$Destination
    )
    $curl = Get-Command curl -ErrorAction SilentlyContinue
    if (-not $curl) {
        throw "curl not found"
    }
    $dir = Split-Path $Destination -Parent
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    if (Test-Path $Destination) { Remove-Item $Destination -Force }
    $lastError = ""
    foreach ($url in (Get-RawGitHubUrls $RelativePath)) {
        & curl.exe -fsSL --retry 2 --retry-delay 1 -o $Destination $url 2>$null
        if ($LASTEXITCODE -eq 0 -and (Test-Path $Destination) -and ((Get-Item $Destination).Length -gt 0)) {
            return $true
        }
        $lastError = "curl exit $LASTEXITCODE for $url"
    }
    throw "Download failed for $RelativePath ($lastError)"
}

function Get-RemoteTextFromMain {
    param([string]$RelativePath)
    $temp = Join-Path ([System.IO.Path]::GetTempPath()) ("sd-trainer-updater-" + [guid]::NewGuid().ToString("n") + ".txt")
    try {
        Invoke-PortableRawDownload -RelativePath $RelativePath -Destination $temp | Out-Null
        return ((Get-Content $temp -TotalCount 1 -ErrorAction Stop) -join "").Trim()
    } catch {
        return ""
    } finally {
        if (Test-Path $temp) { Remove-Item $temp -Force -ErrorAction SilentlyContinue }
    }
}

function Get-RemoteMainProductVersion {
    Get-RemoteTextFromMain "VERSION"
}

function Get-RemoteUpdaterVersionOnline {
    Get-RemoteTextFromMain "scripts/portable/UPDATER_VERSION"
}

function Get-RemoteLatestReleaseTag {
    try {
        $headers = @{ "User-Agent" = "SD-Trainer-Portable-Updater" }
        $uri = "https://api.github.com/repos/$($script:PortableUpdaterRepo)/releases/latest"
        $release = Invoke-RestMethod -Uri $uri -Headers $headers
        $tag = [string]$release.tag_name
        if ($tag -match '^v?(?<ver>.+)$') { return $Matches['ver'] }
        return $tag.TrimStart('v')
    } catch {
        return ""
    }
}

function Read-LocalProductVersion([string]$TrainerDir) {
    $path = Join-Path $TrainerDir "VERSION"
    if (-not (Test-Path $path)) { return "" }
    return ((Get-Content $path -TotalCount 1) -join "").Trim()
}

function Read-LocalPortableBuild([string]$TrainerDir) {
    $path = Join-Path $TrainerDir "PORTABLE_BUILD"
    if (-not (Test-Path $path)) { return "" }
    return ((Get-Content $path -TotalCount 1) -join "").Trim()
}

function Read-LocalUpdaterVersion([string]$TrainerDir) {
    $path = Join-Path $TrainerDir "scripts/portable/UPDATER_VERSION"
    if (-not (Test-Path $path)) { return "unknown" }
    return ((Get-Content $path -TotalCount 1) -join "").Trim()
}

function Get-LocalGitCommit([string]$TrainerDir) {
    if (-not (Test-Path (Join-Path $TrainerDir ".git/HEAD"))) { return "" }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    $hash = (& git -C $TrainerDir rev-parse --short HEAD 2>$null | Select-Object -First 1)
    $ErrorActionPreference = $prev
    if ($hash) { return $hash.Trim() }
    return ""
}

function Write-PortableUpdateStatusBanner {
    param(
        [string]$PortableRoot,
        [string]$UpdaterLabel = "Portable",
        [string]$UpdaterFile = ""
    )
    $PortableRoot = $PortableRoot.TrimEnd('\')
    $trainerDir = Join-Path $PortableRoot "SD-Trainer"
    $localVersion = Read-LocalProductVersion $trainerDir
    $localBuild = Read-LocalPortableBuild $trainerDir
    $localUpdater = Read-LocalUpdaterVersion $trainerDir
    $localGit = Get-LocalGitCommit $trainerDir

    $remoteMain = Get-RemoteMainProductVersion
    $remoteRelease = Get-RemoteLatestReleaseTag
    $remoteUpdater = Get-RemoteUpdaterVersionOnline

    Write-Host "--- Package status / 当前整合包 ---"
    Write-Host ("  VERSION (local / 当前): " + $(if ($localVersion) { $localVersion } else { "(missing)" }))
    if ($localBuild) { Write-Host "  PORTABLE_BUILD (local / 当前): $localBuild" }
    if ($localGit) {
        Write-Host "  Git commit (local / 当前): $localGit"
    } else {
        Write-Host "  Git commit (local / 当前): (no .git / 无 git 仓库)"
    }

    Write-Host "--- Online / 线上最新 ---"
    Write-Host ("  main branch VERSION / main 分支: " + $(if ($remoteMain) { $remoteMain } else { "(unavailable / 无法获取)" }))
    Write-Host ("  Latest Release / 最新 Release: " + $(if ($remoteRelease) { $remoteRelease } else { "(unavailable / 无法获取)" }))
    Write-Host ("  Updater script (online / 线上更新脚本): " + $(if ($remoteUpdater) { $remoteUpdater } else { "(unavailable / 无法获取)" }))

    Write-Host "--- Updater / 本地更新脚本 ---"
    Write-Host ("  Updater script (local / 当前更新脚本): $localUpdater")
    Write-Host ("  Updater kind / 更新类型: $UpdaterLabel")
    if ($UpdaterFile) { Write-Host "  Updater file / 脚本路径: $UpdaterFile" }
    Write-Host ""
}
