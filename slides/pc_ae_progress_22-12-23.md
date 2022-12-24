---
marp: true
---

# Point Cloud Autoencoder Training
### Bachelor's Thesis - Progress Report 2022-12-23 🎄
Joong-Won Seo

---

# Recap: Situation & Goal

![bg right:40%](https://i.imgur.com/LwLFtx9.png)

**Goal**: Improve MultiCam-HGG for Robotic Object Manipulation Tasks

**Current Limitations**:
Camera configuration must be static
=> occulusion, depth-perception, State-Goal Distance metric

**Solution Idea**:
Depth-sensing cameras to produce point clouds
=> number of cameras, position of the cameras, even resolution can change dynamically without affecting the vision module at all.

---

# Recap: Assumptions

Using a RGB-D camera like Kinect 2
![kinect w:200](https://img-prod-cms-rt-microsoft-com.akamaized.net/cms/api/am/imageFileData/RWqOsq?ver=7638)
Can combine multiple cameras
![PC w:300](https://user-images.githubusercontent.com/5233906/66374855-82eb9700-e9de-11e9-99b9-246afedcc6aa.gif)

---

# Simulator, Sensor, PC Capture

Simulator: RoboSuite with predefined robots, tasks, etc.
Sensor: Depth-camera with RGB-D output

### Converting to PC:
Given camera pose (absolute position, orientation) matrix, RGB-D image:
1. Calculate world-to-pixel transformation matrix
2. Inverse the matrix (pixel-to-world)
3. Matrix multiplication: 2.5D Pixels -> 3D Points

Multiple Cameras for better coverage (currently 2-3), simply merge point clouds


---

# Generating Training Set

## Raw Capture
1. Run the simulator, randomize the domain (cube position, robot position)
2. Convert RGB-D images to XYZRGB point clouds
3. Save as NumPy arrays

## Preprocessing
1. Filter out points outside of task area bounding box (e.g. floor, walls, etc.)
2. Furthest Point Sampling Algorithm to create even density and reduced fixed number of points much lower than raw number of points
3. Normalize coordinates to [0; 1]

---

# Naive Attempts

**Architecture**: PointNet Encoder + FC Decoder (with 6 channels per point: XYZRGB)
**Bottleneck Size**: 1024
**Loss Function**: Chamfer Distance (Symmetric)

### Result: Failure
Predicted point cloud is very fuzzy, mostly a fog around the most dense areas (such as the table legs)

### Problem:
Chamfer Distance is mean distance to closest neighbor point! => many points can anchor to 1 central point

---

# Improvement 1: EMD

Earth Mover's Distance: bijective matching from predicted points to input points
Minimum of mean squared distance between point pairs for all possible point pairs

XYZ channels with EMD, RGB channels with common loss functions like MSE.
Algorithm won't match the cube points if not enough iterations ($\epsilon$ = 0.002, 5000 iters)

**Consequences**:
- Training much slower (from 2-5 min to ~30 min)
- Normalize PC coordinates to [0; 1] (task area bounding box needs to be defined)
- Input PC size $\overset{!}{=}$ Reconstructed PC size (chamfer does not require this)

---

# Improvement 2: Higher Loss Weighting for Task Relevant Area

Points on top of the table are much more important (cube, robot arm) than the rest (table surface, table legs, etc.)

Since cube is small, it has less points compared to the table. So it has proportionately less impact on the loss than bigger objects

Define a *Task Relevant Area Bounding Box*, whose points will receive a **bonus weighting** for the loss value.

---

# Other Experiments

### 1. Changing Bottleneck Size
PointNet Encoder outputs a global feature vector of size 1024. Using a fully connected layer, you can adjust the bottleneck size.
For simple cube-only training set, bottleneck of size 3 works well. With the bigger bottleneck, it takes longer to generalize (worse validation score for longer, afterwards similar performance)

### 2. Normalize Latent Space
tanh on the bottleneck to normalize the latent vector to [-1; 1] somehow makes it predict the mean position of the cube and the encoder always creates a latent vector with +-1 elements. Not sure why yet.

---

# More Experiments

### 1. Changing Encoders and Decoders
PointNet Encoder is simple and fast, but is not SOTA. Other encoder implementations have much more strict requirements, installation attempts so far failed.

Decoder is simply fully-connected. Seems to work well no matter what.

### 2. Tuning Loss Weights (XYZ vs RGB)
Currently the loss is $L_{EMD}(A_{XYZ}, B_{XYZ}) + L_{MSE}(A_{RGB}, B_{RGB})$, instead introduce $\alpha \cdot L_{EMD}(A_{XYZ}, B_{XYZ}) + (1-\alpha) \cdot L_{MSE}(A_{RGB}, B_{RGB})$ to control point position accuracy vs point color accuracy

### 3. Disentangled VAE

Evident from the simple cube-only test, the latent space dimensions are entangled. 

---

# Next Steps

### 1. More challenging scenarios
More camera & lighting variations, obstacles, cube shape/color/size, etc.

### 2. Integrating into the simulation and HGG

Challenges:
- Dependency conflict between RoboSuite and PyTorch3D + EMD
- Goal State Representation, State-Goal Distance Metric
    - Or: use the decoder to generate the actual 3D position of the goal position and current position and calculate the euclidean distance! (requires proper segmentation rather than simple RGB)
