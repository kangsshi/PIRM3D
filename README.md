# PIRM3D

Official implementation of "Bridging the Gap: Physics-Inspired 3D Radio Map Construction from Limited Measurements".

![Model architecture](Figs/framework.png)

Source-only main experiment for physics-inspired 3D radio map construction under limited measurements. Includes the 3D Res-UNet, depth generation, residual pseudo-labels, dual-branch loss, training and evaluation. Baselines, ablations, data, pretrained weights and manuscript files are excluded.

## Requested release configuration

| Split | Distinct maps | TX samples per map | 3D samples |
|---|---:|---:|---:|
| Training | 500 | 200 | 100000 |
| Validation | 50 | 200 | 10000 |
| Test | 100 | 200 | 20000 |

The 200 TX samples are separate transmitter-location radio maps, not simultaneous transmitters. Requires at least 650 eligible maps, each with 200 complete 20-height TX samples. Counts are never silently reduced or samples duplicated. Scene IDs are disjoint across splits.

This is a NEW configuration, not the protocol for previously reported paper scores. Core training/model/data code is copied unchanged from the workspace main experiment; only release paths and data counts change. Other training settings retain source defaults, including 50 epochs. No full-data or GPU result is claimed for this configuration.

## Installation

Use Python 3.9 or later and a suitable CUDA-enabled PyTorch installation (source supports PyTorch 1.11 or later). Reuse a compatible existing environment rather than unnecessarily reinstalling CUDA.

```bash
python -m pip install -r requirements-main.txt
# Optional synthetic software checks (tested locally with PyTorch 2.8):
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Dataset

Obtain UrbanRadio3D separately from its maintainers: https://github.com/UNIC-Lab/UrbanRadio3D . Follow its access and usage terms. This repository does not redistribute data.

Edit `data_root` in `main_config.json` or place data at the default path:

```text
data/UrbanRadio3D/
  Building_Infomation/buildingsWHeight/<scene_id>.png
  train/h1/<scene_id>_X<x>_Y<y>.png
  ...
  train/h20/<scene_id>_X<x>_Y<y>.png
  test/h1/<scene_id>_X<x>_Y<y>.png
  ...
  test/h20/<scene_id>_X<x>_Y<y>.png
```

Voxel shape is [20,256,256] in z,y,x order, with 1 m spacing. RSS PNG values are divided by 255. Building PNGs encode heights up to 19.8 m. TX coordinates follow the source convention: row=255-X, column=Y.

Preparation pools eligible maps across the physical train/test folders and creates a NEW seeded scene split. Source folder membership does not determine the logical split. Files are not moved or deleted. A hashed manifest preserves the selected source paths. Insufficient eligible data raises an error.

## Prepare and run

Run from this repository directory. All relative config paths resolve relative to the config file.

```bash
# Select the full requested scene-level split.
python prepare_main.py --config main_config.json --manifest
# Prepare only two training and two validation volumes.
python prepare_main.py --config main_config.json --smoke-data
python train_main.py --config main_config.json --smoke --require-cuda
```

The smoke run uses a separate _smoke output directory. It does not validate accuracy on the full dataset. `--smoke-data` prepares NO test volumes, so do not run test_main.py --smoke until the required test depths/trajectories have also been prepared.

For full preparation and training:

```bash
python prepare_main.py --config main_config.json --trajectories --depth
python prepare_main.py --config main_config.json --check --storage-report
python train_main.py --config main_config.json --require-cuda
# Resume an interrupted run with the same config and manifest:
python train_main.py --config main_config.json --require-cuda --resume
# Evaluate the validation-selected best checkpoint:
python test_main.py --config main_config.json
```

Depth preprocessing is CPU-based and expensive. One full float16 depth volume is 2.5 MiB; 130000 selected volumes require approximately 317.4 GiB for depth arrays alone. Reserve additional space for RSS PNGs, trajectories and weights. Depth files live under data_root/Precomputed_Depth_3D; reuse only matching depth/geometry conventions. The default pseudo-label cache is bounded at 4 GiB.

If raw PNGs are damaged or empty, optional full-decoding selection is available:

```bash
python prepare_valid_data.py --config main_config.json --output artifacts/valid500_tx200 --decode-workers 4 --resume
```

If using this alternative, use the printed main_config_valid.json for ALL subsequent commands. Do not mix configs/manifests between protocols.

Use a persistent terminal for long remote runs:

```bash
tmux new -s pirm3d
python train_main.py --config main_config.json --require-cuda
```

Detach with Ctrl+B then D; reconnect with `tmux attach -t pirm3d`.

## Protocol and outputs

Twenty coverage trajectories define the supervision mask, targeting a 20% union of the full voxel grid. One complete trajectory is the input, nominally about 1% of the grid (5% of supervision). Training selects one route; validation/test use a fixed route. These are samples from simulated reference maps, not real UAV flight measurements.

Inputs are TX location, building occupancy, physics-guided depth, sparse observations and their mask. Depth is coarse distance-and-occlusion guidance, not a full-wave simulation; the source code logarithmically normalizes raw depth. Pseudo-labels propagate measured residuals by nearest assignment, smooth with sigma=1/truncate=2/reflect boundaries, add depth, restore measurements and zero buildings. The loss uses measured-region MSE plus blind-region structural terms with lambda_structure=0.1. Blind-zone dense values are not numerical training labels.

Checkpoints are selected only by validation RMSE. Metrics use clean dense references excluding building interiors and sample-wise macro averages. Mean RMSE therefore need not equal sqrt(mean MSE).

Outputs under outputs/500train_tx200: run_config.json, train.log, history.jsonl, latest_model.pth, best_model.pth, test_results.json and test_per_sample.csv. Artifacts under artifacts/500train_tx200 include the manifest, routes and bounded pseudo cache.

## Code map

- model_3d.py: 3D Res-UNet.
- dataset_main.py / main_trajectories.py: data interface and sparse sampling.
- preprocess_depth.py: depth generation.
- main_pseudo.py / main_loss.py: pseudo-labels and joint loss.
- main_protocol.py / main_runtime.py / main_metrics.py: manifests, checkpoints and metrics.
- prepare_main.py / prepare_valid_data.py: preparation and optional validity filtering.
- train_main.py / test_main.py: main entry points.

See GITHUB_UPLOAD_ZH.md for upload steps. No license or author identity is invented in this package. Before publication, choose a license you are authorized to grant and add the final paper citation. Public visibility alone does not grant a reuse license.

## Contact

Kangshuo Shi: shikangshuo@hnu.edu.cn
