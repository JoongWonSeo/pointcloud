import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcloud_vision.models.pointnet import PointNetEncoder
from pointcloud_vision.models.pointnet2 import PointNet2Encoder
from pointcloud_vision.models.pointmlp import PointMLP, PointMLPElite

# constructors for the different point cloud encoders
backbone_factory = {
    'PointNet': PointNetEncoder,
    'PointNet2': PointNet2Encoder,
    'PointMLP': PointMLP,
    'PointMLPE': PointMLPElite
}



# some are pre-defined model factories, pretends to be a class, hence the PascalCase


###### Point Cloud Autoencoder Architectures ######

class PCEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoding = None

    def forward(self, X):
        self.encoding = self.encoder(X)
        return self.decoder(self.encoding)

def AE(preencoder, out_points=2048, out_dim=6, bottleneck=16):
    encoder = Bottle(preencoder, bottleneck)
    decoder = PCDecoder(bottleneck, out_points, out_dim)
    return PCEncoderDecoder(encoder, decoder)

def SegAE(preencoder, num_classes, out_points=2048, bottleneck=16):
    encoder = Bottle(preencoder, bottleneck)
    decoder = PCSegmenter(bottleneck, out_points, num_classes)
    return PCEncoderDecoder(encoder, decoder)

def FilterAE(preencoder, out_points, bottleneck=16):
    encoder = Bottle(preencoder, bottleneck)
    decoder = PCDecoder(bottleneck, out_points, out_dim=3, hidden_sizes=[256, 512])
    return PCEncoderDecoder(encoder, decoder)

class MultiSegAE(nn.Module):
    def __init__(self, preencoder, name_points_dims):
        super().__init__()
        self.preencoder = preencoder
        self.autoencoders = nn.ModuleDict({
            name: PCEncoderDecoder(
                encoder=nn.Sequential(
                    nn.Linear(preencoder.ENCODING_DIM, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, bottleneck),
                    # nn.ReLU()?
                ),
                decoder=PCDecoder(bottleneck, num_points, out_dim=3, hidden_sizes=[256, 512])
            )
            for name, num_points, bottleneck in name_points_dims
        })
    
    def forward(self, X):
        self.global_encoding = self.preencoder(X)
        return {name: ae(self.global_encoding) for name, ae in self.autoencoders.items()}




###### Point Cloud Encoders ######

def Bottle(preencoder, bottleneck_dim):
    return nn.Sequential(
        preencoder,
        nn.Linear(preencoder.ENCODING_DIM, bottleneck_dim),
        nn.ReLU() #TODO: compare with sigmoid or other activation functions
    )


def GTEncoder(preencoder, out_dim):
    return nn.Sequential(
            preencoder,
            nn.Linear(preencoder.ENCODING_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.Sigmoid(),
    )


###### Point Cloud Decoders ######

# def PCDecoder(encoding_dim, out_points, out_dim):
#     return nn.Sequential(
#             nn.Linear(encoding_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, out_points * out_dim),
#             nn.Sigmoid(),
#             nn.Unflatten(1, (out_points, out_dim))
#     )

def PCDecoder(encoding_dim, out_points, out_dim, hidden_sizes=[512, 1024, 2048]):
    '''
    Creates a simple FC MLP decoder with a variable number of hidden layers.
    The input layer is the encoding dimension, the output layer is the number of points * the dimension of each point.
    '''
    # input layer
    layers = [nn.Linear(encoding_dim, hidden_sizes[0])]
    # hidden layers
    for i in range(len(hidden_sizes) - 1):
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
    layers.append(nn.ReLU())
    # output layer
    layers.append(nn.Linear(hidden_sizes[-1], out_points * out_dim))
    # output activation
    layers.append(nn.Sigmoid())
    layers.append(nn.Unflatten(1, (out_points, out_dim)))
    return nn.Sequential(*layers)


class PCSegmenter(nn.Module):
    def __init__(self, encoding_dim, out_points, num_classes):
        super().__init__()

        self.out_points = out_points
        self.num_classes = num_classes
        out_dim = 3 + num_classes # 3 for xyz, num_classes for class probabilities
        self.segmenter = nn.Sequential(
                nn.Linear(encoding_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, out_points * out_dim),
                nn.Unflatten(1, (out_points, out_dim))
        )
    
    def forward(self, X):
        X = self.segmenter(X)
        # split into xyz and class probabilities
        xyz, seg = X[:, :, :3], X[:, :, 3:]
        # sigmoid over xyz
        xyz = torch.sigmoid(xyz)
        # softmax over class probabilities
        # seg = F.softmax(seg, dim=2) # we don't need this, since we use CrossEntropyLoss
        return torch.cat((xyz, seg), dim=2)


class PCMultiDecoder(nn.Module):
    def __init__(self, encoding_dim, decoder_sizes, decoder_hidden_sizes=[512]):
        '''
        Creates a multi-decoder model, where each decoder has a different number of output points.
        encoding_dim: the dimension of the encoding
        decoder_sizes: a list of the number of output points for each decoder
        decoder_hidden_sizes: a list of the hidden layer sizes to be applied to all decoders
        '''
        super().__init__()
        self.decoder_sizes = decoder_sizes
        self.decoders = nn.ModuleList([
            PCDecoder(encoding_dim, out_points=size, out_dim=3, hidden_sizes=decoder_hidden_sizes)
            for size in decoder_sizes
        ])
    
    def forward(self, X):
        return [decoder(X) for decoder in self.decoders]