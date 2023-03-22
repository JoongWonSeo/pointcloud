import pointcloud_vision.cfg as cfg
import re
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import lightning.pytorch as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pointcloud_vision.models.pc_encoders import AE, SegAE, GTEncoder, backbone_factory
from pointcloud_vision.utils import PointCloudDataset, PointCloudGTDataset, Normalize, OneHotEncode, EarthMoverDistance, seg_to_color
from robosuite_envs.envs import cfg_vision


class Lit(pl.LightningModule):
    '''
    A very generic LightningModule to use for training any model.
    '''

    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.model(x)
        loss = self.loss_fn(prediction, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.model(x)
        loss = self.loss_fn(prediction, y)
        self.log('val_loss', loss)
        # log sample PC
        # TODO: only do this for autoencoders
        # if batch_idx == 0:
        #     logger = self.trainer.logger.experiment # raw tensorboard SummaryWriter
        #     pc = prediction[0, :, :3]
        #     col = seg_to_color(prediction[0, :, 3:].argmax(dim=1).unsqueeze(1))
        #     gt = y[0, :, :3]
        #     gt_col = seg_to_color(y[0, :, 3:].argmax(dim=1).unsqueeze(1))
        #     pc = torch.cat((pc.unsqueeze(0), gt.unsqueeze(0)), dim=0)
        #     col = torch.cat((col.unsqueeze(0), gt_col.unsqueeze(0)), dim=0)
        #     logger.add_mesh('Point Cloud', vertices=pc, colors=col*255, global_step=self.global_step)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=cfg.vision_lr)


def create_model(model_type, backbone, env, load_dir=None):
    if type(env) is str:
        env = SimpleNamespace(**cfg_vision[env]) # dot notation rather than dict notation

    # create the model and dataset
    model, dataset = None, None
    encoder_backbone = backbone_factory[backbone](feature_dims=3)

    if model_type == 'Autoencoder':
        model = Lit(
            AE(encoder_backbone, out_points=env.sample_points, out_dim=6, bottleneck=cfg.bottleneck_size),
            EarthMoverDistance(eps=cfg.emd_eps, its=cfg.emd_iterations, classes=None)
        )
        dataset = lambda input_dir: \
            PointCloudDataset(
                root_dir=input_dir,
                in_features=['rgb'],
                out_features=['rgb']
            )

    if model_type == 'Segmenter':
        C = len(env.classes)
        model = Lit(
            SegAE(encoder_backbone, num_classes=C, out_points=env.sample_points, bottleneck=cfg.bottleneck_size),
            EarthMoverDistance(eps=cfg.emd_eps, its=cfg.emd_iterations, classes=env.class_weights)
        )
        dataset = lambda input_dir: \
            PointCloudDataset(
                root_dir=input_dir,
                in_features=['rgb'],
                out_features=['segmentation'],
                out_transform=OneHotEncode(C, seg_dim=3)
            )

    if model_type == 'GTEncoder':
        model = Lit(
            GTEncoder(encoder_backbone, out_dim=env.gt_dim),
            F.mse_loss
        )
        dataset = lambda input_dir: \
            PointCloudGTDataset(
                root_dir=input_dir,
                in_features=['rgb']
            )
    
    if load_dir:
        model.load_state_dict(torch.load(load_dir)['state_dict'])
    return model, dataset


def train(model_type, backbone, dataset, epochs, batch_size, ckpt_path=None):
    model, open_dataset = create_model(model_type, backbone, env=dataset)

    # Train the created model and dataset
    if model and open_dataset:
        input_dir = f'pointcloud_vision/input/{dataset}'
        output_dir = f'pointcloud_vision/output/{dataset}/{model_type}_{backbone}'
        if ckpt_path:
            # use simple regex to extract the number X from str like 'version_X'
            version = int(re.search(r'version_(\d+)', ckpt_path).group(1))
            print('detected version number from ckpt path:', version)
        else:
            version = None

        # load training and validation data
        train, val = (
            DataLoader(
                open_dataset(f'{input_dir}/{split}'),
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=cfg.vision_dataloader_workers
            )
            for split in ['train', 'val']
        )

        trainer = pl.Trainer(
            logger=TensorBoardLogger(output_dir, name=None, version=version),
            max_epochs=epochs,
            log_every_n_steps=cfg.val_every,
            accelerator=cfg.accelerator,
            detect_anomaly=cfg.debug,
            default_root_dir=output_dir
        )
        trainer.fit(model, train, val, ckpt_path=ckpt_path)
    else:
        print('The model or dataset was not created!', model, open_dataset)


