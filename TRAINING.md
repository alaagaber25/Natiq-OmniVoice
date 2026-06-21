# Natiq-OmniVoice — Training Guide

End-to-end steps to fine-tune **natiq_v2** from the **natiq_v1** base checkpoint
on the multi-channel Arabic/Egyptian data, on a single Windows GPU
(20 GB RTX 4000 Ada).

Two helper scripts wrap the commands below; edit the `CONFIG` block at the top
of each to change paths/params:

- [`tokenize.ps1`](tokenize.ps1) — extract audio tokens for every channel
- [`train.ps1`](train.ps1) — launch fine-tuning

> Run training on **native Windows, not WSL** (WSL is unreliable on this box).

---

## 0. Channels

| Raw data dir (`data\`)   | Token name (`data\tokens\`) | `language_id` |
| ------------------------ | --------------------------- | ------------- |
| `abajora_podcast`        | `abajora`                   | `sa`          |
| `adam_podcast`           | `adam`                      | `sa`          |
| `dupamicaffeine_podcast` | `dupamicaffeine`            | `sa`          |
| `genaya_podcast`         | `genaya`                    | `sa`          |
| `nakhat_elbet`           | `nakhat`                    | `sa`          |
| `wasaya_podcast`         | `wasaya`                    | `sa`          |
| `nadia_elsayed`          | `nadia`                     | `eg`          |
| `mokhbireqtisadi`        | `mokbeir`                   | `eg`          |

Each raw dir contains `metadata.jsonl` (one JSON object per line:
`id`, `audio_path` relative `audio/...`, `text`, `language_id`) and an `audio\`
folder of 24 kHz wavs.

---

## 1. Prerequisites

- Python venv at `..\.venv` (Python 3.11).
- Base checkpoint at `..\models\natiq_v1` (HF-format model folder).
- Higgs audio tokenizer at `..\models\higgs_audio_v2`.
- GPU free (check with `nvidia-smi`).

### The `PYTHONPATH` gotcha (important)

`omnivoice` is installed **non-editable** in the venv's `site-packages`. Without
`PYTHONPATH`, `import omnivoice` loads that copy, which **lacks the local fixes**
(the `F:\` drive-letter WebDataset handler, the `lang_map` dialect tags, the
`num_workers`/builder patches). Both scripts set `PYTHONPATH` to this repo so the
**local** `omnivoice` is used. A `.env` file does **not** work for this — nothing
here loads `.env`, and `PYTHONPATH` must exist before the interpreter starts.

To remove the need permanently, install editable instead:

```powershell
..\.venv\Scripts\python.exe -m pip install -e .
```

---

## 2. Prepare data / fix `language_id`

`language_id` is baked into the token shards **at tokenize time** and training
reads it from the shards — **not** from `data_config_*.json` (the config's
`language_id` field is cosmetic). So set `language_id` correctly in each
`metadata.jsonl` **before** tokenizing. If a channel is already tokenized and you
only need to change the tag, either re-tokenize it or edit the baked
`data\tokens\<name>\train\txts\*.jsonl` shards in place (audio tokens are
unaffected).

Sanity-check the tags:

```powershell
Get-ChildItem data\*\metadata.jsonl | ForEach-Object {
  $ids = (Select-String -Path $_ -Pattern '"language_id": "(\w+)"' -AllMatches).Matches |
         ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
  "{0,-24} {1}" -f $_.Directory.Name, ($ids -join ',')
}
```

---

## 3. Tokenize all channels

Run the helper:

```powershell
.\tokenize.ps1
```

It loops over `$ChannelMap` and, for each channel, runs the tokenizer from
**inside the channel dir** (so the relative `audio/...` paths resolve) and writes
shards + `data.lst` to `data\tokens\<name>\train\`.

Equivalent manual command for one channel (e.g. nadia):

```powershell
$env:PYTHONPATH = "F:\VOOM-AI\GitHubs\TTS\omnivoice-v2\Natiq-OmniVoice"
Push-Location ".\data\nadia_elsayed"
& "..\..\..\.venv\Scripts\python.exe" -m omnivoice.scripts.extract_audio_tokens `
  --input_jsonl metadata.jsonl `
  --tar_output_pattern   "..\tokens\nadia\train\audios\shard-%06d.tar" `
  --jsonl_output_pattern "..\tokens\nadia\train\txts\shard-%06d.jsonl" `
  --tokenizer_path "..\..\..\models\higgs_audio_v2" `
  --nj_per_gpu 3 --shuffle True
Pop-Location
```

Output layout per channel:

```
data\tokens\<name>\train\
├── audios\shard-000000.tar   # audio tokens (.npy) + per-sample metadata
├── txts\shard-000000.jsonl   # baked metadata incl. language_id
├── data.lst                  # <tar> <jsonl> <count> <duration>  (absolute paths)
└── errors.jsonl
```

Notes:
- `data.lst` stores **absolute** shard paths baked in at tokenize time. If you
  move the repo or switch OS (Windows vs WSL paths differ), **re-tokenize**.
- `--min_length 1.0` drops sub-1 s clips (optional; off by default).
- Re-running a channel overwrites its `tokens\<name>` output.

---

## 4. Configure

- **Data config** — [`examples/config/data_config_natiq_v2_all.json`](examples/config/data_config_natiq_v2_all.json):
  one entry per channel pointing at its `data.lst`, with a `repeat` weight. The
  `language_id` field here is cosmetic (the real tag comes from the shards).
- **Train config** — [`examples/config/train_config_natiq_v2_sdpa.json`](examples/config/train_config_natiq_v2_sdpa.json):
  the **stable** recipe for this GPU. Key fields:
  - `init_from_checkpoint`: `...\models\natiq_v1` (fresh optimizer, step 0)
  - `attn_implementation: sdpa` — **do not use `flex_attention`** (unusable on
    this GPU; that's what plain `train_config_natiq_v2.json` uses)
  - `num_workers: 0`, `batch_tokens: 2048`, `max_batch_size: 16`
  - `learning_rate: 1e-5`, `steps: 15000`, `mixed_precision: bf16`
  - `save_steps / eval_steps: 500`, `logging_steps: 25`

  Alternative `train_config_natiq_v2_long.json` uses a cosine schedule + warmup
  at `5e-6` (same stable SDPA/batch settings).

---

## 5. Train

```powershell
.\train.ps1
```

Equivalent manual command (from this dir):

```powershell
$env:PYTHONPATH = "F:\VOOM-AI\GitHubs\TTS\omnivoice-v2\Natiq-OmniVoice"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
# Use the module form, not accelerate.exe (the uv trampoline shim is broken).
& "..\.venv\Scripts\python.exe" -m accelerate.commands.launch `
  --gpu_ids 0 --num_processes 1 `
  -m omnivoice.cli.train `
  --train_config examples\config\train_config_natiq_v2_sdpa.json `
  --data_config  examples\config\data_config_natiq_v2_all.json `
  --output_dir   exp\natiq_v2
```

**Healthy start:** loss ~5.2 dropping to ~3.5 within ~80 steps. Checkpoints land
in `exp\natiq_v2` every 500 steps; you can stop early once a checkpoint is good.
Each checkpoint is a drop-in HF-format model folder (use as a new natiq model).

### Fresh run vs. resume

- **Fresh:** a clean `--output_dir` starts from `natiq_v1` at step 0. To keep a
  previous run, point `--output_dir` (`$OutputDir` in `train.ps1`) at a new
  folder, e.g. `exp\natiq_v2_all`.
- **Resume:** set `"resume_from_checkpoint"` in the train config to a checkpoint
  path under the output dir.

---

## 6. Verify before/after

Shard counts per channel:

```powershell
Get-ChildItem data\tokens\*\train\data.lst |
  ForEach-Object { "{0,-16} {1} shards" -f $_.Directory.Parent.Name, (Get-Content $_).Count }
```

GPU usage while training:

```powershell
nvidia-smi
```

---

## 7. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ModuleNotFoundError: omnivoice` or local edits ignored | `PYTHONPATH` not set to this repo (or use `pip install -e .`). |
| Training runs then **freezes** (steps/sec decays to a stall) | GPU memory fragmentation. Keep `num_workers: 0`, `attn_implementation: sdpa`, `batch_tokens: 2048`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. |
| `num_workers > 0` crashes/stalls on Windows | Use `num_workers: 0` (Windows spawn + non-picklable issues). |
| Tokenizer can't find audio (all samples skipped) | Audio paths resolve relative to **cwd**; run from inside the channel dir (the scripts do this). |
| WebDataset can't open `F:\...` paths | Use the **local** omnivoice (drive-letter handler) — i.e. `PYTHONPATH`. |
| Inference warns "Language not recognized" for `sa`/`eg` | Add the dialect tags to the **local** `omnivoice/utils/lang_map.py` `LANG_IDS`. |
| `flex_attention` errors / OOM at attention | Use the SDPA config; flex_attention is unusable on this GPU. |
| `uv trampoline failed to canonicalize script path` from `accelerate.exe` | The uv console-script shim is broken; call the module instead: `python -m accelerate.commands.launch ...` (what `train.ps1` does). |
