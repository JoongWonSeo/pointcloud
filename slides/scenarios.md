---
marp: true
---

# Scenarios & Training Data
For robotic manipulation tasks

---

# Simulating Input

**Azure Kinect** with depth camera resolution $1024 \times 1024 = 1048576$ Points (+RGB!)

![kinect w:300](https://img-prod-cms-rt-microsoft-com.akamaized.net/cms/api/am/imageFileData/RWqOsq?ver=7638) ![point cloud w:700](https://ed.ilogues.com/Tutorials/kinect2/images/kinect3.gif)

---

# Multiple Kinects

2 Kinects:
![](https://user-images.githubusercontent.com/5233906/66374855-82eb9700-e9de-11e9-99b9-246afedcc6aa.gif)

3 Kinects (Room):
https://www.youtube.com/watch?v=c9KRys3bSFg

---

# Common Elements in Scene

![bg /*fit*/ right:40%](https://i.imgur.com/LwLFtx9.png)

1. Robot arm
2. Actuator
3. Object to manipulate
4. Table/Floor (**ok** to collide)
5. Goal (**should** collide)
6. *Obstacles/Walls (should **not** collide)*
7. *Unknown?*

In reality usually only partial view of scene

Assign each point one of the above labels
--> Objects could be color-coded since Kinect detects RGB for each point!

---

# Training Data

1) Gather color-coded scene data (XYZRGB point cloud) with Kinect (or simulated)
2) *(simple pre-processing conversion XYZRGB color-coding -> XYZL labeled points)*
3) Train Autoencoder in either ways:
    - Autoencoder XYZL -> Latent z -> XYZL
    - Autoencoder XYZ -> Latent z -> XYZL
    - *(Encoder XYZ -> Latent z, then n Decoders Latent z -> XYZ for each label)*
4) RL agent uses Encoder

---

# Autoencoder Usage Ideas

* Goal can just be given as 3D position into the Encoder as usual

* Loss or State-Goal Distance Metric can be:
    * If the input is RGB and thus the objects can be color-coded, then use raw input data to find object position and calculate distance directly
    * If the input is only positional, use the segmenting decoder to calculate object position

* Anomaly Detection: if loss is too high, unknown object might exist
(maybe spatially localize the loss and detect regions with high local loss?)

* Unknown Object Detection: 1 Encoder -> N Decoders for N Classes
(subtract all N decoded points from input, then the remainder is unknown objects?)
(train by minimizing sum/avg loss for all N paths?)