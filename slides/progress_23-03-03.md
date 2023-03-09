---
marp: true
---

# Point-Cloud-Based 3D Vision Module for Robotic Reinforcement Learning Tasks
### Bachelor's Thesis - **Progress Report 2023-03-03**
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
=> number of cameras, position of the cameras, even resolution can change dynamically without affecting the vision module (or just barely)

---

# Recap: Progress & Findings

**Loss Functions**:
- Chamfer Distance (CD), while fast, did not work well
- Earth Mover Distance (EMD) better, but not enough yet:
    - **Issue**: Huge distribution imbalance of points belonging to Cube vs Env, since the cube is too small.
    - **Workaround**: Define a *Task Area Bounding Box*, whose points will receive a **bonus weighting** for the loss value.
    => worked for very simple [Table+Cube] point clouds
![w:200](https://i.imgur.com/7e7R8GO.png)

---

# New Problems

Since EMD with TABB worked for [Table+Cube], try [Table+Cube+Robot]:
=> **Failure Again**: Both the Cube and the Robot is in the TABB, so this approach will not generalize!
![w:300](https://i.imgur.com/8rLuYSo.png)
What we really want is each object (Table, Cube, Robot) to be reconstructed equally well, regardless of how many points belong to that object!

---

# Idea: **Weighted EMD Loss**

If we have a segmented point cloud (each point has a label indicating the object/class it belongs to), then we can explicitly calculate the class-point distribution of the target PC, and then calculate appropriate weights accordingly!

Each object has an assigned weight that is antiproportional to the number of points, in order to compensate for imbalanced distribution.

=> Segmented training data required (**no longer unsupervised**)
=> Segmenting AE (XYZRGB -> XYZL)
=> Combined with improved Encoders (PointNet++, PointMLP, etc.)
=> Now [Table+Cube+Robot] can also be reconstructed
=> We can even explicitly extract reconstructed cube position

---

# Re-evaluating the use of Autoencoders

But if we require supervised learning anyways (labeled data), then why are we bothering with the latent space of the AE?
Using the latent vector space to represent the state is a big headache:
- Need to disentangle the features
- New hyperparameter: latent space dimension (=bottleneck size)
- Encoder vs Decoder capacity
- Goal-State representation is hard (need to have observation of goal state)
- Latent space distance metric is even harder

---

# State/Observation Spaces

- **GT-State Space** $G$: The fundamental state of the environment (e.g. pose of the objects), only directly accessible in simulation
- **Raw Observation Space** $O$: Observation of the environment, i.e. direct sensor outputs (RGBD/PC), available in both reality and simulation.
- **Encoded Observation Space** $E$: Output of the Encoder, actual observation space used by the RL Agent

$$G \rightarrow [Sensor] \rightarrow O \rightarrow [Encoder] \rightarrow E \rightarrow [Agent]$$

**Goal Formulation**: Easy in $G$, but we need it in $E$. Not always feasible!
**Goal Checking**: Easy in $G$, very hard in $E$

---

# Ground-Truth-Predictor

If $E = G$, all of the above problems disappear:

$$G \rightarrow [Sensor] \rightarrow O \rightarrow [Encoder] \rightarrow G \rightarrow [Agent]$$

Simply put, we train the Encoder to predict the original!

**Advantages**:
- Goal can be directly given to Agent without any transformations
- Goal checking is trivial
- Agent is now *fully decoupled from the Encoder*!
- No decoding step, easier to train (*needs verification*)
- ...

---

# Future Direction:

Compare the vision module pipelines:
1. Pure Autoencoder (Unsupervised)
2. Segmenting Autoencoder (Labeled PC)
3. Ground-Truth-Predictor (Supervised by Simulation)
4. Combinations

w.r.t. various environments and tasks

---

# Other Progresses

- Created a wrapper around RoboSuite Env to be compatible with Gym API
- Created a modular ObservationEncoder
- Better Encoders: PointNet++, PointMLP
- 