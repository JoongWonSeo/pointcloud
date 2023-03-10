import torch.nn as nn
import torch.nn.functional as F
from .pointnet import PointNetEncoder
from .pointnet2 import PointNet2Encoder
from .pointmlp import pointMLP, pointMLPElite

class PCEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoding = None

    def forward(self, X):
        self.encoding = self.encoder(X)
        return self.decoder(self.encoding)


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

def GTDecoder(encoding_dim, out_dim):
    return nn.Sequential(
            nn.Linear(encoding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.Sigmoid(),
    )

def Bottleneck(encoder_dim, bottleneck_dim):
    return nn.Sequential(
        nn.Linear(encoder_dim, bottleneck_dim),
        nn.ReLU()
    )


# pre-defined model factories, pretends to be a class, hence the PascalCase
def PNAutoencoder(out_points=2048, in_dim=6, out_dim=6, bottleneck=16):
    encoder = nn.Sequential(
        PointNetEncoder(in_channels=in_dim),
        Bottleneck(1024, bottleneck)
    )
    decoder = PCDecoder(bottleneck, out_points, out_dim)
    return PCEncoderDecoder(encoder, decoder)

def PN2Autoencoder(out_points=2048, in_dim=6, out_dim=6, bottleneck=16):
    encoder = nn.Sequential(
        PointNet2Encoder(space_dims=3, feature_dims=in_dim-3),
        Bottleneck(1024, bottleneck)
    )
    decoder = PCDecoder(bottleneck, out_points, out_dim)
    return PCEncoderDecoder(encoder, decoder)

def PMLPAutoencoder(out_points=2048, out_dim=6, bottleneck=16):
    encoder = nn.Sequential(
        pointMLP(points=out_points),
        Bottleneck(1024, bottleneck)
    )
    decoder = PCDecoder(bottleneck, out_points, out_dim)
    return PCEncoderDecoder(encoder, decoder)

def PMLPEAutoencoder(out_points=2048, out_dim=6, bottleneck=16):
    encoder = nn.Sequential(
        pointMLPElite(points=out_points),
        Bottleneck(1024, bottleneck)
    )
    decoder = PCDecoder(bottleneck, out_points, out_dim)
    return PCEncoderDecoder(encoder, decoder)

def PN2GTPredictor(in_dim=6, out_dim=3, bottleneck=16):
    # encoder = PointNet2Encoder(space_dims=3, feature_dims=in_dim-3)
    encoder = nn.Sequential(
        PointNet2Encoder(space_dims=3, feature_dims=in_dim-3),
        Bottleneck(1024, bottleneck)
    )
    decoder = GTDecoder(bottleneck, out_dim)
    return PCEncoderDecoder(encoder, decoder)

def PMLPEGTPredictor(out_points=2048, out_dim=3):
    encoder = pointMLPElite(points=out_points)
    decoder = GTDecoder(1024, out_dim)
    return PCEncoderDecoder(encoder, decoder)