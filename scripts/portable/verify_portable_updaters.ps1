# Pre-release checks for portable update scripts (git + release).
param(
    [Parameter(Mandatory = $true)]
    [string]$PortableRoot,
    [switch]$SkipReleaseApi
)

$ErrorActionPreference = "Stop"
$PortableRoot = (Resolve-Path $PortableRoot).Path.TrimEnd('\')
$failures = @()

function Test-FileExists([string]$RelativePath, [string]$Label) {
    $full = Join-Path $PortableRoot $RelativePath
    if (-not (Test-Path $full)) {
        $script:failures += "[FAIL] $Label missing: $RelativePath"
        return $false
    }
    Write-Host ('[OK] ' + $Label)
    return $true
}

Write-Host ""
Write-Host "Portable updater verification" -ForegroundColor Cyan
Write-Host "Root: $PortableRoot"
Write-Host ""

Test-FileExists "Update-SD-Trainer.bat" "Git updater" | Out-Null
Test-FileExists "Update-SD-Trainer-Release.bat" "Release updater" | Out-Null
Test-FileExists "update\update_sd_trainer.bat" "update_sd_trainer shortcut" | Out-Null
Test-FileExists "update\update_from_release.bat" "update_from_release shortcut" | Out-Null
Test-FileExists "SD-Trainer\scripts\portable\update_from_release.ps1" "update_from_release.ps1" | Out-Null

$gitHead = Join-Path $PortableRoot "SD-Trainer\.git\HEAD"
if (Test-Path $gitHead) {
    Write-Host '[OK] SD-Trainer\.git\HEAD'
} else {
    $failures += "[FAIL] SD-Trainer\.git\HEAD missing (git update will not work)"
}

$ps1 = Join-Path $PortableRoot "SD-Trainer\scripts\portable\update_from_release.ps1"
if (Test-Path $ps1) {
    $content = Get-Content $ps1 -Raw
    if ($content -notmatch "SD-Trainer-v") {
        $failures += "[FAIL] update_from_release.ps1 missing asset filter"
    }
    if ($content -match "extensions" -and $content -match "autosave") {
        Write-Host '[OK] update_from_release.ps1 user-data exclusions'
    } else {
        $failures += "[FAIL] update_from_release.ps1 missing user-data exclusions"
    }
}

if (-not $SkipReleaseApi) {
    Write-Host ""
    Write-Host "Release API dry-run..."
    if (Test-Path $ps1) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $ps1 -PortableRoot $PortableRoot -DryRun
        if ($LASTEXITCODE -ne 0) {
            $failures += "[FAIL] update_from_release.ps1 -DryRun failed (network or API)"
        } else {
            Write-Host '[OK] Release API reachable'
        }
    }
}

Write-Host ""
if ($failures.Count -gt 0) {
    Write-Host ('FAILED: ' + $failures.Count + ' issue(s)') -ForegroundColor Red
    $failures | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    exit 1
}

Write-Host 'All updater checks passed.' -ForegroundColor Green
exit 0
