# 3D Point Cloud Autoencoder

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