# Run the test suite with pytest, using xdist if available.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Push-Location $RepoRoot

$PytestArgs = @()
python -c "import xdist" 2>$null
if ($LASTEXITCODE -eq 0) {
    $PytestArgs += "-n", "4"
}

Write-Host "Running pytest with: $PytestArgs $args"
python -m pytest $PytestArgs tests/ $args
Pop-Location
