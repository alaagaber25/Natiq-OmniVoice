# ============================================================================
# train.ps1 - Fine-tune natiq_v2 from the natiq_v1 base checkpoint.
#
# Launches omnivoice.cli.train via accelerate on a single GPU using the SDPA
# train config (the stable recipe for the 20GB RTX 4000 Ada). The list of
# channel manifests (data.lst) is taken from $DataConfig. Checkpoints are
# written to $OutputDir every save_steps (see the train config).
#
# Run tokenize.ps1 first so every data.lst referenced by $DataConfig exists.
#
# Usage:
#   .\train.ps1
# ============================================================================

# ---------------------------- CONFIG (edit me) ------------------------------
$Repo        = "F:\VOOM-AI\GitHubs\TTS\omnivoice-v2\Natiq-OmniVoice"
# Invoke accelerate as a module via the venv python (the accelerate.exe shim is a
# uv trampoline that errors with "failed to canonicalize script path").
$Python      = "F:\VOOM-AI\GitHubs\TTS\omnivoice-v2\.venv\Scripts\python.exe"
$TrainConfig = "examples\config\train_config_natiq_v2_dialect.json"  # corrected ratios; SDPA = stable here
$DataConfig  = "examples\config\data_config_natiq_v2_all.json"     # all 8 channels
$OutputDir   = "exp\natiq_v2"
$GpuIds      = "0"
$NumProc     = 1
# ---------------------------------------------------------------------------

$ErrorActionPreference = "Stop"
Set-Location $Repo
# Use the LOCAL omnivoice (site-packages copy lacks the local fixes).
$env:PYTHONPATH = $Repo
# Avoid the fragmentation stall seen with variable-length SDPA batches.
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

Write-Host "Training: $TrainConfig + $DataConfig -> $OutputDir" -ForegroundColor Cyan

& $Python -m accelerate.commands.launch `
  --gpu_ids $GpuIds --num_processes $NumProc `
  -m omnivoice.cli.train `
  --train_config $TrainConfig `
  --data_config  $DataConfig `
  --output_dir   $OutputDir
if ($LASTEXITCODE -ne 0) { throw "training exited with code $LASTEXITCODE" }
