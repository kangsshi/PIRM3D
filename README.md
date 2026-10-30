## PIRM3D: Physics-Inspired 3D Radio Map Construction from Limited Measurements

![model framework](Figs/framework.png "Model Architecture")

The official implementation of "Bridging the Gap: Physics-Inspired 3D Radio Map Construction from Limited Measurements".

In this project, we propose PIRM3D, a physics-inspired 3D framework designed for high-fidelity radio map construction under limited measurements. PIRM3D integrates deterministic electromagnetic propagation laws via a 3D radio depth map and natively captures correlations across different heights with a 3D Res-UNet architecture. By introducing a soft-supervision paradigm with physics-knowledge-driven pseudo-labels and a dual-branch joint loss, PIRM3D effectively mitigates overfitting in unmeasured blind zones and ensures physical topological consistency across diverse urban environments.

## Installation
### Environment
- Linux
- Python >= 3.9
- PyTorch >= 1.11 (CUDA-enabled for GPU training)

```bash
pip install -r requirements.txt
```

## Data
The data used for training and evaluation can be found in [UrbanRadio3D](https://github.com/UNIC-Lab/UrbanRadio3D).
Set `data_root` in `main_config.json` to your dataset directory. Keep the original `train/h1`--`train/h20`, `test/h1`--`test/h20`, and `Building_Infomation/buildingsWHeight` folder structure.

The default split uses 500 training maps, 50 validation maps, and 100 test maps, with 200 transmitter-location samples per map. Map IDs are disjoint across splits. All other main-experiment settings are unchanged.

Before training, generate the data split, UAV trajectories, and physical depth maps:

```bash
python prepare_main.py --config main_config.json --manifest --trajectories --depth
```

Physics-knowledge-driven pseudo-labels are generated during training through nearest-measurement residual propagation and Gaussian smoothing.

## Model Training and Evaluation

To train PIRM3D, run:

```bash
python train_main.py --config main_config.json --require-cuda
```

The trained weights are saved to `outputs/500train_tx200`. Add `--resume` to continue interrupted training.

To evaluate the validation-selected best model on the testing set, run:

```bash
python test_main.py --config main_config.json
```

## 📧 Contact

If you have any questions or want to use the code, feel free to contact:
* Kangshuo Shi (shikangshuo@hnu.edu.cn)
