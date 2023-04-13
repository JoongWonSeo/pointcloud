import pointcloud_vision.cfg as cfg
import re
import argparse
from types import SimpleNamespace
from math import ceil
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import lightning.pytorch as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pointcloud_vision.models.architectures import AE, SegAE, MultiSegAE, MultiGTEncoder, backbone_factory
import pointcloud_vision.pc_encoder as pc_encoder
from pointcloud_vision.utils import PointCloudDataset, PointCloudGTDataset, Normalize, Unnormalize, OneHotEncode, FilterClasses, ChamferDistance, FilteringChamferDistance, SegmentingChamferDistance, EarthMoverDistance, StatePredictionLoss, seg_to_color
from robosuite_envs.envs import cfg_scene


class Lit(pl.LightningModule):
    '''
    A very generic LightningModule to use for training any model.
    '''

    def __init__(self, model, loss_fn, log_info=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.log_info = log_info

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
        if self.log_info and batch_idx == 0:
            logger = self.trainer.logger.experiment # raw tensorboard SummaryWriter

            if self.log_info == 'Autoencoder':
                pc = prediction[0, :, :3]
                col = prediction[0, :, 3:]
                gt = y[0, :, :3]
                gt_col = y[0, :, 3:]
                pc = torch.cat((pc.unsqueeze(0), gt.unsqueeze(0)), dim=0)
                col = torch.cat((col.unsqueeze(0), gt_col.unsqueeze(0)), dim=0)
                logger.add_mesh('Point Cloud', vertices=pc, colors=col*255, global_step=self.global_step)

            if self.log_info == 'Segmenter':
                pass
                # pc = prediction[0, :, :3]
                # col = seg_to_color(prediction[0, :, 3:].argmax(dim=1).unsqueeze(1), classes=env.classes)
                # gt = y[0, :, :3]
                # gt_col = seg_to_color(y[0, :, 3:].argmax(dim=1).unsqueeze(1), classes=env.classes)
                # pc = torch.cat((pc.unsqueeze(0), gt.unsqueeze(0)), dim=0)
                # col = torch.cat((col.unsqueeze(0), gt_col.unsqueeze(0)), dim=0)
                # logger.add_mesh('Point Cloud', vertices=pc, colors=col*255, global_step=self.global_step)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=cfg.vision_lr)


def create_model(model_type, backbone, scene, load_dir=None):
    scene_name = scene
    scene = SimpleNamespace(**cfg_scene[scene_name]) # dot notation rather than dict notation

    # create the model and dataset
    model, dataset = None, None
    encoder_backbone = backbone_factory[backbone](feature_dims=3) # RGB input

    if model_type == 'Autoencoder':
        model = Lit(
            AE(encoder_backbone, out_points=scene.sample_points, out_dim=6, bottleneck=sum(scene.class_latent_dim)),
            EarthMoverDistance(eps=cfg.emd_eps, its=cfg.emd_iterations, num_classes=None),
            log_info=model_type
        )
        dataset = lambda input_dir: \
            PointCloudDataset(
                root_dir=input_dir,
                in_features=['rgb'],
                out_features=['rgb'],
                in_transform=Normalize(scene.bbox),
                # out_transform=Normalize(scene.bbox), # since x==y identity-wise, only normalize once
            )

    elif model_type == 'Segmenter':
        C = len(scene.classes)
        model = Lit(
            SegAE(encoder_backbone, num_classes=C, out_points=scene.sample_points, bottleneck=sum(scene.class_latent_dim)),
            EarthMoverDistance(eps=cfg.emd_eps, its=cfg.emd_iterations, num_classes=C),
            log_info=model_type
        )
        dataset = lambda input_dir: \
            PointCloudDataset(
                root_dir=input_dir,
                in_features=['rgb'],
                out_features=['segmentation'],
                in_transform=Normalize(scene.bbox),
                out_transform=Normalize(scene.bbox)
            )
    
    elif model_type == 'MultiSegmenter':
        name_points_dims = [ #(class name, number of points, latent dimension)
            (n, ceil(p*scene.sample_points), d)
            for (n, p, d) in zip(scene.classes, scene.class_distribution, scene.class_latent_dim)
            if d>0
        ]
        class_labels = {n: scene.classes.index(n) for (n, _, _) in name_points_dims}
        print(f'MultiFilter: {name_points_dims}')

        model = Lit(
            MultiSegAE(encoder_backbone, class_labels, name_points_dims),
            SegmentingChamferDistance(class_labels),
            log_info=model_type
        )
        dataset = lambda input_dir: \
            PointCloudDataset(
                root_dir=input_dir,
                in_features=['rgb'],
                out_features=['segmentation'],
                in_transform=Normalize(scene.bbox),
                out_transform=Normalize(scene.bbox)
            )

    elif model_type == 'StatePredictor':
        state_dims = {n: d for (n, d) in zip(scene.states, scene.state_dim) if d>0}
        transforms = pc_encoder.StatePredictor.from_state(scene)

        model = Lit(
            MultiGTEncoder(encoder_backbone, state_dims),
            StatePredictionLoss(state_dims, transforms),
        )
        dataset = lambda input_dir: \
            PointCloudGTDataset(
                root_dir=input_dir,
                in_features=['rgb'],
                in_transform=Normalize(scene.bbox),
            )

    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')
    
    if load_dir:
        model.load_state_dict(torch.load(load_dir)['state_dict'])
    
    model.loss_fn.log = model.log # let the loss function log to tensorboard

    return model, dataset


def train(model_type, backbone, scene, epochs, batch_size, ckpt_path=None, dataset_dir=None):
    model, open_dataset = create_model(model_type, backbone, scene=scene)

    dataset_dir = dataset_dir or scene

    # Train the created model and dataset
    if model and open_dataset:
        input_dir = f'input/{dataset_dir}'
        output_dir = f'output/{dataset_dir}/{model_type}_{backbone}'
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
                num_workers=cfg.vision_dataloader_workers,
                # pin_memory=True # TODO: try enabling?
            )
            for split in ['train', 'val']
        )

        torch.set_float32_matmul_precision('medium')
        trainer = pl.Trainer(
            logger=TensorBoardLogger(output_dir, name=None, version=version),
            max_epochs=epochs,
            log_every_n_steps=cfg.val_every,
            accelerator=cfg.accelerator,
            precision=cfg.precision,
            detect_anomaly=cfg.debug,
            default_root_dir=output_dir
        )
        trainer.fit(model, train, val, ckpt_path=ckpt_path)
    else:
        print('The model or dataset was not created!', model, open_dataset)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or evaluate a vision module')
    parser.add_argument('scene', type=str)
    parser.add_argument('model', choices=cfg.models)
    parser.add_argument('--scene_dir', default=None, type=str)
    parser.add_argument('--backbone', choices=cfg.encoder_backbones, default='PointNet2')
    parser.add_argument('--batch_size', default=cfg.vision_batch_size, type=int,
                        help='batch size for training')
    parser.add_argument('--epochs', default=cfg.vision_epochs, type=int,
                        help='number of epochs to train for')
    parser.add_argument('--ckpt', default=None, type=str,
                        help='path to checkpoint to load (to either resume training or evaluate)')
    a = parser.parse_args()


    print(f'device = {cfg.device}')

    train(a.model, a.backbone, a.scene, a.epochs, a.batch_size, a.ckpt, a.scene_dir)