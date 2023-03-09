# 3D Point Cloud Autoencoder

## Structure

cfg.py: global configuration that has an effect on the training and simulation, in order to make it easier to switch between different configurations and increase reproducibility and reduce the number of command line arguments
main.py: script to train, test, run the vision, rl and simulator

sim
    robosuite_envs: a subpackage for a gym-robosuite_envs interface with obs-encoders
    simulator.py: run and view the env with obs-encoder and agent
    utils.py: includes utils for rendering, UI, [TODO: move to vision.util:] point cloud creation, etc.

vision
    input: contains training, validation, and test data for different environments
    output: output data of the model (autoencoder)
    loss: point cloud loss functions including EMD, Chamfer and weighted EMD
    models: source code for the models
    weights: trained weights of the models
    ae_viewer.py: interactive live viewer for the vision model
    pc_viewer.py: browser based viewer for the point clouds
    generate_pc.py: generate training data for the vision model
    train.py: train the vision model
    utils.py: includes pc transforms, and seg visualization

rl
    coming soon [TODO]

## Scripts

`simulator.py`: Run a RoboSuite simulation and interact with it to explore the task environment.

`generate_pc.py`: Used to run a RoboSuite simulation and automatically capture point clouds from the RGBD cameras to generate training data.

`preprocess.py`: Apply preprocessing to the raw captured point clouds e. g. sampling N points and normalizing the coordinates.

`main.py`: Used to train/evaluate/visualize the Autoencoder

`viewer.py`: Browser-based point cloud viewer. TODO: integrate into `main.py viewer`


## Remarks

### `simulator.py`
When using onscreen renderer, you need to run `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so` or have it in your `~/.bashrc`.  
When using offscreen renderer, do `unset LD_PRELOAD` or just not do the export above.