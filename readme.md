# 3D Point Cloud Autoencoder


## Installation

Please install CUDA 11.7. Then, in a conda environment, you can run
```
pip install -r requirements.txt
```


## Packages and Features

This installable package provides two main features:

1. `robosuite_envs` are robosuite-based environments for gymnasium-robotics API (more specifically, the Multi-Goal Env API / `GoalEnv` Interface):
    - Basically plug-and-play for any RL algorithms based on Gym, including DDPG, HER, HGG, etc.
    - Additionally, it provides a ObservationEncoder interface which can be used to modularly plug in a vision module (see below) into any robosuite environment. Concrete implementations of this is only provided in the vision module (simple ground-truth observer included that work as a passthrough)

2. `pointcloud_vision` package contains the vision module:
    - The underlying PyTorch model and training procedure, including training data generation and training script
    - Some point cloud loss functions (Chamfer Distance, Weighted Earth Mover Distance)
    - various utilities for visualization, PyTorch Dataset for point clouds, point cloud transformations in `torchvision.transforms` style (downsampling, filtering, normalization, etc.)


## Structure

`cfg.py`: global configuration that has an effect on the training and simulation, in order to make it easier to switch between different configurations and increase reproducibility and reduce the number of command line arguments

`robosuite_envs`: a subpackage for a gym-robosuite_envs interface with obs-encoders
- `simulator.py`: run and view the env with obs-encoder and agent
- `base_env.py`: the inheritable environment wrapper around robosuite
- `envs.py`: predefined scenes and tasks
- `sensors.py`: base sensor module and passthrough sensor
- `encoders.py`: base observation encoder module and pasthrough encoder
- `utils.py`: includes utils for rendering, UI, point cloud creation, etc.

`pointcloud_vision/`
- `input/`: contains training, validation, and test data for different environments
- `loss/`: point cloud loss functions including EMD, Chamfer and weighted EMD
- `models/`: source code for the models
- `output/`: trained weights of the models
- `ae_viewer.py`: interactive live viewer for the vision model
- `pc_viewer.py`: browser based viewer for the point clouds
- `generate_pc.py`: generate training data for the vision model
- `train.py`: train the vision model
- `utils.py`: includes pc transforms, and seg visualization


## How to Use

### Generating Point Cloud Training Data

Run the following scripts in the `pointcloud_vision/` directory.

For example for the RobosuiteReach-v0 dataset (validation), this will create 25 * 8 = 200 point cloud frames with randomized movement. The `--show_distribution` flag shows all generated data combined including the ground truth (red) and goal (green) if they're simple 3D points.
```
python generate_pc.py input/Table/val --env RoboReach-v0 --horizon 25 --runs 8 --show_distribution
```

To view any of the generated point cloud in the browser, you can run e.g.:
```
python pc_viewer.py input/Table/train/0.npz
```

### Training a Point Cloud Encoder

In the `pointcloud_vision/` folder, run:
```
python train.py Table Autoencoder --backbone PointNet2
```
Tensorboard will log some sample Point Clouds in the Mesh section.

To interactively view the Encoder+Decoder, run:
```
python ae_viewer.py --view=sidebyside --animation_speed=0.5 Lift Lift Segmenter
```

### Train a RL Agent using [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)

Make sure you also have [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) and [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) installed.
```
python -m rl_zoo3.train --algo tqc --env RoboReach-v0 --progress --tensorboard-log ./output --save-freq 100000
```

Multiprocessing (be careful of RAM usage!):
```
python -m rl_zoo3.train --algo tqc --env RoboReach-v0 --progress --tensorboard-log ./output --save-freq 100000 --vec-env subproc -params n_envs:4
```

Viewing the result:  
```
python -m rl_zoo3.enjoy --algo tqc --env RoboReach-v0 --folder ./logs
```




## Remarks

### `simulator.py`
When using onscreen renderer, you need to run `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so` or have it in your `~/.bashrc`.  
When using offscreen renderer, do `unset LD_PRELOAD` or just not do the export above.
