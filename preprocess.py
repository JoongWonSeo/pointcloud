from torchvision.transforms import Compose
from models.pn_autoencoder import PointcloudDataset
from train_utils import *


train_set = PointcloudDataset(root_dir='input', files=None, transform=Compose([
    SampleFurthestPoints(2048),
    # SampleRandomPoints(2048),
    Normalize([[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]]), #3D bounding box [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    # Normalize((-0.4, 0.4, -0.4, 0.4, 0.8, 1.5)),
    ]))

for i in range(len(train_set)):
    train_set.save(i, 'prep')
    print(('#' * round(i/len(train_set) * 100)).ljust(100, '-'), end='\r')
print('\ndone')
