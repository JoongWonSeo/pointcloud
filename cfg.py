# this file is meant to be imported by other python scripts
# it contains the configuration, basically functioning as a global variable file
# note that some of these can be changed by the e.g. main.py script

# device to use for PyTorch
device = 'cuda:0'

# this enables more verbose output and more sanity checks (performance impact)
debug = True


########## Vision: Env Specific Variables ##########

classes = [ # name and color
    ('env', [0, 0, 0]),
    ('cube', [1, 0, 0]),
    ('arm', [0.5, 0.5, 0.5]),
    ('base', [0, 1, 0]),
    ('gripper', [0, 0, 1]),
]

class_weights = [ # name and training weight # TODO: automatically calculate these in EMD based on the distribution of the classes
    ('env', 1.5),
    ('cube', 150.0),
    ('arm', 5.0),
    ('base', 10.0),
    ('gripper', 15.0),
    # ('env', 1.0),
    # ('cube', 100.0),
    # ('arm', 0.0),
    # ('base', 0.0),
    # ('gripper', 0.0),
]

bbox = [[-0.5, 0.5], [-0.5, 0.5], [0, 1.5]]