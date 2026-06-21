# ============================================================================
# tokenize.ps1 - Extract Higgs audio tokens for every channel in $ChannelMap,
#                then carve a small dev split out of each channel.
#
# For each channel it runs the tokenizer from INSIDE the channel's data dir so
# the relative "audio/..." paths in metadata.jsonl resolve, writing WebDataset
# shards + data.lst to  data\tokens\<name>\train\ . It then moves
# $DevShardsPerChannel full shard(s) per channel into data\tokens\<name>\dev\
# and rebuilds both data.lst files. The data_config's train/dev entries point at
# these (e.g. data_config_natiq_v2_all.json).
#
# NOTE: the language_id baked into the shards comes from each metadata.jsonl at
#       tokenize time. Training reads language_id from the shards (NOT from
#       data_config_*.json), so fix metadata.jsonl BEFORE tokenizing.
#
# Usage:
#   .\tokenize.ps1                 # tokenize all channels + carve dev
#
# Edit the CONFIG block below to change paths / channels / params.
# ============================================================================

# ---------------------------- CONFIG (edit me) ------------------------------
$Repo      = "F:\VOOM-AI\GitHubs\TTS\omnivoice-v2\Natiq-OmniVoice"
$Python    = "F:\VOOM-AI\GitHubs\TTS\omnivoice-v2\.venv\Scripts\python.exe"
$Tokenizer = "F:\VOOM-AI\GitHubs\TTS\omnivoice-v2\models\higgs_audio_v2"
$DataDir   = "$Repo\data"     # holds the raw <channel> dirs and the tokens\ output
$Split     = "train"          # tokenizer output goes to data\tokens\<name>\<Split>\
$NjPerGpu  = 3                # tokenizer worker processes per GPU
$Shuffle   = "True"
$MinLength = 0.0              # seconds; set >0 (e.g. 1.0) to drop short clips

$MakeDevSplit        = $true  # carve a dev split after tokenizing
$DevShardsPerChannel = 1      # full shards moved to dev per channel (1/33 ~= 3%)

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

# --- Stage 1: tokenize each channel into <name>\<Split>\ ---
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

# --- Stage 2: carve dev split (move full shards <Split> -> dev, rebuild data.lst) ---
# Done in Python: reliable UTF-8/JSONL handling and atomic data.lst rebuild.
if ($MakeDevSplit) {
  Write-Host "`n=== Carving dev split ($DevShardsPerChannel full shard(s)/channel) ===" -ForegroundColor Cyan
  $env:TOKENS_DIR = Join-Path $DataDir "tokens"
  $env:SRC_SPLIT  = $Split
  $env:DEV_SHARDS = "$DevShardsPerChannel"
  $carve = @'
import os, shutil, pathlib
TOKENS = pathlib.Path(os.environ["TOKENS_DIR"])
SRC    = os.environ.get("SRC_SPLIT", "train")
K      = int(os.environ.get("DEV_SHARDS", "1"))

def read_lst(p):
    return [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

for ch in sorted(TOKENS.iterdir()):
    if not ch.is_dir():
        continue
    src_lst = ch / SRC / "data.lst"
    if not src_lst.exists():
        continue
    # A fresh tokenize wrote the FULL set to <SRC>/, so drop any stale dev/.
    if (ch / "dev").exists():
        shutil.rmtree(ch / "dev")
    lines = read_lst(src_lst)
    if len(lines) <= K:
        print("%-16s SKIP (only %d shards)" % (ch.name, len(lines)))
        continue
    # Pick the K largest (i.e. full, not the remainder) shards for dev.
    order = sorted(range(len(lines)), key=lambda i: int(lines[i].split()[2]), reverse=True)
    move_idx = set(order[:K])
    keep = [l for i, l in enumerate(lines) if i not in move_idx]
    move = [l for i, l in enumerate(lines) if i in move_idx]
    devd = ch / "dev"
    (devd / "audios").mkdir(parents=True, exist_ok=True)
    (devd / "txts").mkdir(parents=True, exist_ok=True)
    dev_lines, dev_n = [], 0
    for line in move:
        tar, jsonl, cnt, dur = line.split()
        dtar   = devd / "audios" / pathlib.Path(tar).name
        djsonl = devd / "txts"   / pathlib.Path(jsonl).name
        shutil.move(tar, str(dtar))
        shutil.move(jsonl, str(djsonl))
        dev_lines.append("%s %s %s %s" % (os.path.abspath(dtar), os.path.abspath(djsonl), cnt, dur))
        dev_n += int(cnt)
    (devd / "data.lst").write_text("\n".join(dev_lines) + "\n", encoding="utf-8")
    src_lst.write_text("\n".join(keep) + "\n", encoding="utf-8")
    keep_n = sum(int(l.split()[2]) for l in keep)
    print("%-16s %s %2d/%-6d | dev %d/%d" % (ch.name, SRC, len(keep), keep_n, len(move), dev_n))
'@
  $carve | & $Python -
  if ($LASTEXITCODE -ne 0) { throw "dev carve failed with code $LASTEXITCODE" }
}

# --- Summary ---
Write-Host "`nDone. Token manifests (shard counts):" -ForegroundColor Green
foreach ($name in $ChannelMap.Values) {
  $t = Join-Path $DataDir "tokens\$name\$Split\data.lst"
  $d = Join-Path $DataDir "tokens\$name\dev\data.lst"
  $ts = 0; if (Test-Path $t) { $ts = (Get-Content $t).Count }
  $ds = 0; if (Test-Path $d) { $ds = (Get-Content $d).Count }
  "{0,-16} $Split={1,2}  dev={2}" -f $name, $ts, $ds
}
