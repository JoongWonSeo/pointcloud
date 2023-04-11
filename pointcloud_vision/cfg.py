# this file is meant to be imported by other python scripts
# it contains the configuration, basically functioning as a global variable file
# note that some of these can be changed by the e.g. main.py script
# most of these have effect on the training or simulation
# machine-specific things should be CMD arguments instead, e.g. paths


########## General Settings ##########

# device to use for PyTorch and Lightning
device = 'cuda'
accelerator = 'gpu'
precision = '16-mixed' # 16 for mixed precision, 32 for full precision

# this enables more verbose output and more sanity checks (performance impact)
debug = True


########## Vision: Model and Training Settings ##########

models = ['Autoencoder', 'Segmenter', 'GTEncoder', 'GTDecoder', 'GTSegmenter', 'ObjectFilter']
encoder_backbones = ['PointNet', 'PointNet2', 'PointMLP', 'PointMLPE']

bottleneck_size = 16 # TODO: should depend on the env and the model


# Training settings
vision_dataloader_workers = 6

vision_batch_size = 25 # can be overwritten by args
vision_epochs = 100 # can be overwritten by args
vision_lr = 1e-3 # default for Adam

val_every = 4 # validation every n batches/steps

# Earth Mover's Distance loss precision
# during training
emd_eps = 0.005
emd_iterations = 50

# during testing
emd_test_eps = 0.002
emd_test_iterations = 10000
