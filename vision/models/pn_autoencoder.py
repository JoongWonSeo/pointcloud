import torch
import torch.nn as nn
import torch.nn.functional as F
from vision.models.pointnet import PointNetEncoder
from vision.models.pointnet2 import PointNet2Encoder
from vision.models.pointmlp import PointMLP, PointMLPElite

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
            nn.Sigmoid(), # TODO: GT has no garantee to be in [0, 1], even dim-wise! Solution: normalize GT to [0, 1] before training
    )


###### Point Cloud Decoders ######

def PCDecoder(encoding_dim, out_points, out_dim):
    return nn.Sequential(
            nn.Linear(encoding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_points * out_dim),
            nn.Sigmoid(),
            nn.Unflatten(1, (out_points, out_dim))
    )

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
        seg = F.softmax(seg, dim=2)
        return torch.cat((xyz, seg), dim=2)




# def PNAutoencoder(out_points=2048, in_dim=6, out_dim=6, bottleneck=16):
#     encoder = Bottle(PointNetEncoder(in_channels=in_dim), 1024, bottleneck)
#     decoder = PCDecoder(bottleneck, out_points, out_dim)
#     return PCEncoderDecoder(encoder, decoder)

# def PN2Autoencoder(out_points=2048, in_dim=6, out_dim=6, bottleneck=16):
#     encoder = Bottle(PointNet2Encoder(space_dims=3, feature_dims=in_dim-3), 1024, bottleneck)
#     decoder = PCDecoder(bottleneck, out_points, out_dim)
#     return PCEncoderDecoder(encoder, decoder)

# def PMLPAutoencoder(inout_points=2048, out_dim=6, bottleneck=16):
#     encoder = Bottle(pointMLP(points=inout_points), 1024, bottleneck)
#     decoder = PCDecoder(bottleneck, inout_points, out_dim)
#     return PCEncoderDecoder(encoder, decoder)

# def PMLPEAutoencoder(inout_points=2048, out_dim=6, bottleneck=16):
#     encoder = Bottle(pointMLPElite(points=inout_points), 1024, bottleneck)
#     decoder = PCDecoder(bottleneck, inout_points, out_dim)
#     return PCEncoderDecoder(encoder, decoder)

