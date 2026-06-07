# setup.ps1
# ──────────────────────────────────────────────────────────────
# RedTeamRL — Windows PowerShell Setup
# Run from the project root:  .\setup.ps1
# ──────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$ProjectName = "RedTeamRL"
$VenvDir     = ".venv"

Write-Host ""
Write-Host "  $ProjectName — Environment Setup" -ForegroundColor Cyan
Write-Host "  ────────────────────────────────" -ForegroundColor Cyan
Write-Host ""

# ── Execution policy (if not already set) ─────────────────────
$policy = Get-ExecutionPolicy -Scope CurrentUser
if ($policy -eq "Restricted") {
    Write-Host "  Setting PowerShell execution policy to RemoteSigned …" -ForegroundColor Yellow
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
}

# ── Create venv ───────────────────────────────────────────────
Write-Host "  Creating virtual environment in .\$VenvDir …" -ForegroundColor Green
python -m venv $VenvDir

# ── Activate ──────────────────────────────────────────────────
$Activate = ".\$VenvDir\Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) {
    Write-Host "  ERROR: venv activation script not found at $Activate" -ForegroundColor Red
    exit 1
}
. $Activate
Write-Host "  Virtual environment activated." -ForegroundColor Green

# ── Upgrade pip ───────────────────────────────────────────────
python -m pip install --upgrade pip --quiet

# ── Install CPU-only PyTorch first (avoids pulling CUDA build) ─
Write-Host "  Installing PyTorch (CPU build) …" -ForegroundColor Yellow
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet

# ── Install remaining requirements ────────────────────────────
Write-Host "  Installing project requirements …" -ForegroundColor Yellow
pip install -r requirements.txt --quiet

# ── Verify ────────────────────────────────────────────────────
Write-Host ""
Write-Host "  Verifying installation …" -ForegroundColor Green

$checks = @("gymnasium", "stable_baselines3", "networkx", "torch", "matplotlib")
foreach ($pkg in $checks) {
    $ver = python -c "import $pkg; print($pkg.__version__)" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓  $pkg  $ver" -ForegroundColor Green
    } else {
        Write-Host "  ✗  $pkg  FAILED" -ForegroundColor Red
    }
}

# ── Create placeholder log dirs ───────────────────────────────
New-Item -ItemType Directory -Force -Path "logs"               | Out-Null
New-Item -ItemType Directory -Force -Path "models"             | Out-Null
New-Item -ItemType Directory -Force -Path "models\checkpoints" | Out-Null

Write-Host ""
Write-Host "  Setup complete!  Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "    1. Activate venv (if not active):"
Write-Host "       .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "    2. Sanity check:"
Write-Host "       python main.py --mode demo"
Write-Host ""
Write-Host "    3. Train the agent (500 k steps, ~30–60 min on CPU):"
Write-Host "       python main.py --mode train"
Write-Host ""
Write-Host "    4. Monitor training:"
Write-Host "       tensorboard --logdir logs"
Write-Host ""
Write-Host "    5. Visualise attack path after training:"
Write-Host "       python main.py --mode visualize"
Write-Host ""
