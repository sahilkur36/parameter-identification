$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$UiPath = Join-Path $Root "src\parameter_identifier\ui\main_window.ui"
$OutputPath = Join-Path $Root "src\parameter_identifier\ui\ui_main_window.py"

pyuic5 $UiPath -o $OutputPath
Write-Host "Generated $OutputPath"
