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
    '''
    Generic point cloud autoencoder for global scene encoding
    '''
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoding = None

    def forward(self, X):
        self.encoding = self.encoder(X)
        return self.decoder(self.encoding)

def AE(preencoder, out_points=2048, out_dim=6, bottleneck=16):
    encoder = PCEncoder(preencoder, bottleneck, output_activation=None)
    decoder = PCDecoder(bottleneck, out_points, out_dim)
    return PCEncoderDecoder(encoder, decoder)

def SegAE(preencoder, num_classes, out_points=2048, bottleneck=16):
    encoder = PCEncoder(preencoder, bottleneck, output_activation=None)
    decoder = PCSegmenter(bottleneck, out_points, num_classes)
    return PCEncoderDecoder(encoder, decoder)


class MultiBottle(nn.Module):
    '''
    Generic module for preencoder + multiple bottlenecks (+ optionally decoder)
    '''
    def __init__(self, preencoder):
        super().__init__()
        self.preencoder = preencoder
        self.autoencoders = nn.ModuleDict()
    
    def forward(self, X):
        self.global_encoding = self.preencoder(X)
        return {name: ae(self.global_encoding) for name, ae in self.autoencoders.items()}
    
    def remove_unused(self, whitelist):
        blacklist = self.autoencoders.keys() - whitelist
        for name in blacklist:
            del self.autoencoders[name]

class MultiSegAE(MultiBottle):
    def __init__(self, preencoder, class_labels, name_points_dims):
        super().__init__(preencoder)
        self.class_labels = class_labels

        self.autoencoders = nn.ModuleDict({
            name: PCEncoderDecoder(
                encoder=MLP(preencoder.ENCODING_DIM, bottleneck, hidden_sizes=[512, 256], output_activation=None),
                decoder=PCDecoder(bottleneck, num_points, out_dim=3, hidden_sizes=[256, 512])
            )
            for name, num_points, bottleneck in name_points_dims
        })
    
    def forward_encoders(self, X):
        self.global_encoding = self.preencoder(X)
        return {name: ae.encoder(self.global_encoding) for name, ae in self.autoencoders.items()}
    
    def reconstruct_labeled(self, X):
        device = X.device
        self.global_encoding = self.preencoder(X)
        class_pcs = {self.class_labels[name]: ae(self.global_encoding) for name, ae in self.autoencoders.items()}
        # give each pc a label dimension
        class_pcs = [torch.cat([pc, torch.ones(pc.shape[0], pc.shape[1], 1, device=device) * l], dim=2) for l, pc in class_pcs.items()]
        # concatenate all pcs
        return torch.cat(class_pcs, dim=1)
    
    @property
    def local_encodings(self):
        return {name: sub_ae.encoding for name, sub_ae in self.autoencoders.items()}
    
    @property
    def flat_local_encodings(self):
        return torch.cat([sub_ae.encoding for sub_ae in self.autoencoders.values()], dim=1)

class MultiGTEncoder(MultiBottle):
    def __init__(self, preencoder, state_dims):
        super().__init__(preencoder)

        self.autoencoders = nn.ModuleDict({
            name: MLP(
                input_size=preencoder.ENCODING_DIM,
                hidden_sizes=[512, 256, 128],
                output_size=gt_dim,
                output_activation=nn.Sigmoid
            )
            for name, gt_dim in state_dims.items()
        })
    

###### Point Cloud Encoders ######
def PCEncoder(preencoder, bottleneck_dim, hidden_sizes=[], output_activation=None):
    return nn.Sequential(
        preencoder,
        *MLP(
            input_size=preencoder.ENCODING_DIM,
            hidden_sizes=hidden_sizes,
            output_size=bottleneck_dim,
            activation=nn.ReLU,
            output_activation=output_activation
        ).children()
    )


def GTEncoder(preencoder, out_dim, hidden_sizes=[512, 256, 128], output_activation=nn.Sigmoid):
    return nn.Sequential(
        preencoder,
        *MLP(
            input_size=preencoder.ENCODING_DIM,
            hidden_sizes=hidden_sizes,
            output_size=out_dim,
            activation=nn.ReLU,
            output_activation=output_activation
        ).children()
    )


###### Point Cloud Decoders ######
def PCDecoder(encoding_dim, out_points, out_dim, hidden_sizes=[512, 1024, 2048]):
    '''
    Creates a simple FC MLP decoder with a variable number of hidden layers.
    The input layer is the encoding dimension, the output layer is the number of points * the dimension of each point.
    '''
    return nn.Sequential(
        *MLP(
            input_size=encoding_dim,
            hidden_sizes=hidden_sizes,
            output_size=out_points * out_dim,
            activation=nn.ReLU,
            output_activation=nn.Sigmoid
        ).children(),
        nn.Unflatten(1, (out_points, out_dim))
    )

class PCSegmenter(nn.Module):
    def __init__(self, encoding_dim, out_points, num_classes, hidden_sizes=[512, 1024, 2048]):
        super().__init__()

        self.out_points = out_points
        self.num_classes = num_classes
        out_dim = 3 + num_classes # 3 for xyz, num_classes for class probabilities
        self.segmenter = nn.Sequential(
            *MLP(
                input_size=encoding_dim,
                hidden_sizes=hidden_sizes,
                output_size=out_points * out_dim,
                activation=nn.ReLU,
                output_activation=None
            ).children(),
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


###### Generic Network Architectures ######
def MLP(input_size, output_size, hidden_sizes, activation=nn.ReLU, output_activation=nn.ReLU):
    '''
    Creates a simple FC MLP decoder with a variable number of hidden layers.
    '''
    if hidden_sizes is None or len(hidden_sizes) == 0:
        layers = [nn.Linear(input_size, output_size)]
        if output_activation is not None:
            layers.append(output_activation())
        return nn.Sequential(*layers)
    
    # input layer
    layers = [nn.Linear(input_size, hidden_sizes[0])]
    layers.append(activation())
    # hidden layers
    for i in range(len(hidden_sizes) - 1):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        layers.append(activation())
    # output layer
    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)

