# PIRM3D 更新与运行

仓库：https://github.com/kangsshi/PIRM3D

本次为新版主实验代码：训练500张、验证50张、测试100张地图，每张地图200个独立发射机位置样本。训练、验证、测试地图ID互不重叠。至少需要650张符合完整性要求的地图；数量不足会报错，不会重复采样或自动减少数量。

更新后的入口为 prepare_main.py、train_main.py、test_main.py。旧的 dataset_3d.py、generate_3d_labels.py、generate_uav_mask.py、test_3d.py 已由新版模块替代，历史版本仍在Git记录中。原框架图和联系方式保留。

## 在算力云运行

```bash
git clone https://github.com/kangsshi/PIRM3D.git
cd PIRM3D
python -m pip install -r requirements-main.txt
```

修改 main_config.json 的 data_root 为实际数据根目录。默认路径为 data/UrbanRadio3D。若使用ZIP包，解压后进入包内PIRM3D目录再执行相同命令。

```bash
python prepare_main.py --config main_config.json --manifest
python prepare_main.py --config main_config.json --smoke-data
python train_main.py --config main_config.json --smoke --require-cuda
python prepare_main.py --config main_config.json --trajectories --depth
python prepare_main.py --config main_config.json --check --storage-report
tmux new -s pirm3d
python train_main.py --config main_config.json --require-cuda
python test_main.py --config main_config.json
```

训练中断后以同一配置添加 --resume。新版划分使用独立 artifacts/500train_tx200 和 outputs/500train_tx200 目录，不可沿用旧划分的清单或权重断点。默认训练50轮，其余模型、优化器、采样和损失参数保留新版主实验默认值。

共100000个训练、10000个验证、20000个测试三维样本。仅130000份float16深度体积约需317.4 GiB，预处理前应确认存储空间。这是新配置，不是旧论文分数对应的原实验划分。

## 后续更新

在已有克隆中修改所需文件，检查 git diff 后提交，再 git pull --ff-only 和 git push。不使用强制推送。数据、模型权重和生成的实验输出不随源码包上传。
