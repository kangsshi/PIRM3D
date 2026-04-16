## PIRM3D: Physics-Inspired 3D Radio Map Construction from Limited Measurements

![model framework](Figs/framework.png "Model Architecture")

The official implementation of "Bridging the Gap: Physics-Inspired 3D Radio Map Construction from Limited Measurements". 

In this project, we propose PIRM3D, a physics-inspired 3D framework designed for high-fidelity radio map construction under limited measurements. PIRM3D integrates deterministic electromagnetic propagation laws via a 3D radio depth map and natively captures altitude-dimension continuity with a 3D Res-UNet architecture. By introducing a soft-supervision paradigm with physics-knowledge-driven pseudo-labels and a dual-branch joint loss, PIRM3D effectively mitigates overfitting in unmeasured blind zones and ensures physical topological consistency across diverse urban environments.

## Installation
### Environment
- Tested OS: Linux
- Python >= 3.8
- torch >= 2.0.0
- scipy >= 1.7.0
- numpy >= 1.21.0

## Data
The data used for training and evaluation can be found in [UrbanRadio3D](https://github.com/UNIC-Lab/UrbanRadio3D).
After downloading the data, move the building topology maps, BS locations, and ground truth radio maps to `./data`.

Before training, you need to process the original data to generate the limited measurement masks, physical depth maps, and pseudo-labels:
1. Generate the constrained 3D UAV sampling mask:
   run: `generate_uav_mask.py`
2. Construct the physics-guided 3D radio depth maps:
   run: `preprocess_depth.py`
3. Generate physics-knowledge-driven pseudo-labels via Regression Kriging:
   run: `generate_3d_labels.py`

## Model Training and Evaluation

To train PIRM3D, run: `train_main.py`
- For training the 3D Res-UNet with the dual-branch joint loss, configure the data paths and run the main script.
- The trained weights will be saved to `./Checkpoints`.

To evaluate the model, run: `test_3d.py`
- This will load the saved model and evaluate it on the unseen testing set, outputting volumetric prediction metrics (RMSE, NMSE, and PSNR).

## 📧 Contact

If you have any questions or want to use the code, feel free to contact:
* Kangshuo Shi (shikangshuo@hnu.edu.cn)
