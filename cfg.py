# this file is meant to be imported by other python scripts
# it contains the configuration, basically functioning as a global variable file
# note that some of these can be changed by the e.g. main.py script
# most of these have effect on the training or simulation
# machine-specific things should be CMD arguments instead, e.g. paths


########## General Settings ##########

# device to use for PyTorch
device = 'cuda'

# this enables more verbose output and more sanity checks (performance impact)
debug = True


########## Vision: Env and Point Cloud Settings ##########

env = 'Lift'
robot = 'Panda'

# bounding box of the environment (for cropping and normalization)
bbox = [[-0.5, 0.5], [-0.5, 0.5], [0, 1.5]] # (x_min, x_max), (y_min, y_max), (z_min, z_max)

# name and color of each segmentation class
classes = [ # (name, RGB_for_visualization)
    ('env', [0, 0, 0]),
    ('cube', [1, 0, 0]),
    ('arm', [0.5, 0.5, 0.5]),
    ('base', [0, 1, 0]),
    ('gripper', [0, 0, 1]),
]

# name and training weight per point # TODO: automatically calculate these in EMD based on the distribution of the classes
class_weights = [ # (name, weight)
    ('env', 1.0),
    ('cube', 150.0),
    ('arm', 5.0),
    ('base', 10.0),
    ('gripper', 15.0),
]

# RGBD sensors to generate pointclouds from
camera_poses = { # name: (position, quaternion)
    'frontview': ([0, -1.2, 1.8], [0.3972332, 0, 0, 0.9177177]),
    'agentview': ([0. , 1.2, 1.8], [0, 0.3972332, 0.9177177, 0]),
    'birdview': ([1.1, 0, 1.6], [0.35629062, 0.35629062, 0.61078392, 0.61078392])
}
camera_size = (128, 128) # width, height

# number of points to sample from the raw point cloud
pc_sample_points = 2048

def pc_preprocessor():
    from torchvision.transforms import Compose
    from vision.utils import FilterClasses, FilterBBox, SampleRandomPoints, SampleFurthestPoints, Normalize
    return Compose([
        # FilterClasses(whitelist=[0, 1], seg_dim=6), # only keep table and cube
        FilterBBox(bbox),
        # SampleRandomPoints(pc_sample_points),
        SampleFurthestPoints(pc_sample_points),
        Normalize(bbox)
    ])


########## Vision: Model and Training Settings ##########

model = ['PNAutoencoder', 'PN2Autoencoder', 'PN2PosExtractor', 'PosDecoder'][1]

if model == 'PNAutoencoder':
    def create_vision_module():
        from vision.models.pn_autoencoder import PNAutoencoder
        return PNAutoencoder(2048, in_dim=6, out_dim=4)

    def get_dataset_args(input_dir):
        return {
            'root_dir': input_dir,
            'in_features': ['rgb'],
            'out_features': ['segmentation'] # for weighted EMD
        }

elif model == 'PN2Autoencoder':
    def create_vision_module():
        from vision.models.pn_autoencoder import PN2Autoencoder
        return PN2Autoencoder(pc_sample_points, in_dim=6, out_dim=4)

    def get_dataset_args(input_dir):
        return {
            'root_dir': input_dir,
            'in_features': ['rgb'],
            'out_features': ['segmentation'] # for weighted EMD
        }

elif model == 'PN2PosExtractor':
    def create_vision_module():
        from vision.models.pn_autoencoder import PN2PosExtractor
        return PN2PosExtractor(6)

    def get_dataset_args(input_dir):
        from vision.utils import mean_cube_pos
        return {
            'root_dir': input_dir,
            'out_transform': mean_cube_pos,
            'in_features': ['rgb'], # too easy to predict with color
            'out_features': ['segmentation'] # for mean cube pos
        }

elif model == 'PosDecoder':
    def create_vision_module():
        from vision.models.pn_autoencoder import PosDecoder
        return PosDecoder(pc_sample_points, 4) 

    def get_dataset_args(input_dir):
        from vision.utils import mean_cube_pos
        return {
            'root_dir': input_dir,
            'in_transform': mean_cube_pos,
            'in_features': ['segmentation'], # for mean cube pos
            'out_features': ['segmentation'] # for weighted EMD
        }



# Training settings
vision_training_set = 'vision/input/' + env + '/train'
vision_validation_set = 'vision/input/' + env + '/val'
vision_output = 'vision/output/' + env

vision_batch_size = 25 # can be overwritten by args
vision_epochs = 100 # can be overwritten by args
vision_lr = 1e-3 # default for Adam

val_every = 4 # validation every n batches/steps

# Earth Mover's Distance loss precision
emd_eps = 0.002
emd_iterations = 5000


########## Sim Settings ##########

dense_reward = False # dense or sparse reward, requires retraining the agent
renderer_camera = 'agentview'
renderer_default_pose = [1.1, 0, 1.6], [0.35, 0.35, 0.60, 0.60]
