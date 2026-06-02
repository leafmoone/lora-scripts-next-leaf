param(
    [Parameter(Mandatory = $true)]
    [string]$PortableRoot,
    [string]$UpdaterLabel = "Portable",
    [string]$UpdaterFile = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "portable_updater_common.ps1")
Write-PortableUpdateStatusBanner -PortableRoot $PortableRoot -UpdaterLabel $UpdaterLabel -UpdaterFile $UpdaterFile
