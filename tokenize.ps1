# ============================================================================
# tokenize.ps1 - Extract Higgs audio tokens for every channel in $ChannelMap.
#
# For each channel it runs the tokenizer from INSIDE the channel's data dir so
# the relative "audio/..." paths in metadata.jsonl resolve, then writes the
# WebDataset shards + data.lst to  data\tokens\<name>\<split>\ .
#
# NOTE: the language_id baked into the shards comes from each metadata.jsonl
#       at tokenize time. Training reads language_id from the shards (NOT from
#       data_config_*.json), so fix metadata.jsonl BEFORE tokenizing.
#
# Usage:
#   .\tokenize.ps1                 # tokenize all channels in $ChannelMap
#
# Edit the CONFIG block below to change paths / channels / params.
# ============================================================================

# ---------------------------- CONFIG (edit me) ------------------------------
$Repo      = "F:\VOOM-AI\GitHubs\TTS\omnivoice-v2\Natiq-OmniVoice"
$Python    = "F:\VOOM-AI\GitHubs\TTS\omnivoice-v2\.venv\Scripts\python.exe"
$Tokenizer = "F:\VOOM-AI\GitHubs\TTS\omnivoice-v2\models\higgs_audio_v2"
$DataDir   = "$Repo\data"     # holds the raw <channel> dirs and the tokens\ output
$Split     = "train"          # output goes to data\tokens\<name>\<Split>\
$NjPerGpu  = 3                # tokenizer worker processes per GPU
$Shuffle   = "True"
$MinLength = 0.0              # seconds; set >0 (e.g. 1.0) to drop short clips

# raw data dir  ->  token output name (the name MUST match the paths in your
# data_config_*.json, e.g. mokhbireqtisadi -> mokbeir).
$ChannelMap = [ordered]@{
  "abajora_podcast"        = "abajora"          # sa
  "adam_podcast"           = "adam"             # sa
  "dupamicaffeine_podcast" = "dupamicaffeine"   # sa
  "genaya_podcast"         = "genaya"           # sa
  "nakhat_elbet"           = "nakhat"           # sa
  "wasaya_podcast"         = "wasaya"           # sa
  "nadia_elsayed"          = "nadia"            # eg
  "mokhbireqtisadi"        = "mokbeir"          # eg
}
# ---------------------------------------------------------------------------

$ErrorActionPreference = "Stop"
# Use the LOCAL omnivoice: the venv has a non-editable copy in site-packages
# that lacks the local fixes (F:\ drive-letter handler, lang_map dialects, etc.)
$env:PYTHONPATH = $Repo

foreach ($raw in $ChannelMap.Keys) {
  $name   = $ChannelMap[$raw]
  $rawDir = Join-Path $DataDir $raw
  if (-not (Test-Path (Join-Path $rawDir "metadata.jsonl"))) {
    Write-Warning "Skipping '$raw': no metadata.jsonl in $rawDir"
    continue
  }

  Write-Host "`n=== Tokenizing $raw -> tokens\$name\$Split ===" -ForegroundColor Cyan

  # Output patterns are relative to the channel dir (cwd). "..\tokens" therefore
  # resolves to data\tokens\... and data.lst is written with absolute paths.
  $tarPattern   = "..\tokens\$name\$Split\audios\shard-%06d.tar"
  $jsonlPattern = "..\tokens\$name\$Split\txts\shard-%06d.jsonl"

  Push-Location $rawDir
  try {
    & $Python -m omnivoice.scripts.extract_audio_tokens `
      --input_jsonl metadata.jsonl `
      --tar_output_pattern   $tarPattern `
      --jsonl_output_pattern $jsonlPattern `
      --tokenizer_path $Tokenizer `
      --min_length $MinLength `
      --nj_per_gpu $NjPerGpu `
      --shuffle $Shuffle
    if ($LASTEXITCODE -ne 0) { throw "tokenizer exited with code $LASTEXITCODE for '$raw'" }
  }
  finally {
    Pop-Location
  }
}

Write-Host "`nAll channels tokenized. Manifests (shard counts):" -ForegroundColor Green
Get-ChildItem "$DataDir\tokens\*\$Split\data.lst" |
  ForEach-Object { "{0,-16} {1} shards" -f $_.Directory.Parent.Name, (Get-Content $_).Count }
