import cfg
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import lightning.pytorch as pl
from vision.models.pn_autoencoder import PointcloudDataset, PNAutoencoder, PN2Autoencoder, PN2PosExtractor
from vision.utils import Normalize, EarthMoverDistance, mean_cube_pos
from torch.utils.tensorboard import SummaryWriter


class Lit(pl.LightningModule):
    def __init__(self, predictor, loss_fn):
        super().__init__()
        self.model = predictor
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
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=cfg.vision_lr)


def train(model_type, dataset, epochs, batch_size):
    model, train, val = None, None, None

    if model_type == 'PNAutoencoder':
        model = Lit(
            PNAutoencoder(cfg.pc_sample_points, in_dim=6, out_dim=6),
            EarthMoverDistance(eps=cfg.emd_eps, its=cfg.emd_iterations, classes=cfg.class_weights)
        )
        train, val = (
            DataLoader(
                PointcloudDataset(
                    root_dir=f'vision/input/{dataset}/{split}',
                    in_features=['rgb'],
                    out_features=['rgb']
                ),
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=cfg.vision_dataloader_workers
            )
            for split in ['train', 'val']
        )
    if model_type == 'GTPredictor':
        model = Lit(PN2PosExtractor(6), F.mse_loss)
        train, val = [
            DataLoader(
                PointcloudDataset(
                    root_dir=f'vision/input/{dataset}/{split}',
                    in_features=['rgb'],
                    out_features=['segmentation'],
                    out_transform=mean_cube_pos
                ),
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=cfg.vision_dataloader_workers
            )
            for split in ['train', 'val']
        ]

    if model:
        trainer = pl.Trainer(
            max_epochs=epochs,
            log_every_n_steps=cfg.val_every,
            accelerator=cfg.accelerator,
            detect_anomaly=cfg.debug
        )
        trainer.fit(model, train, val)
    else:
        print('Invalid model type')



# def train(train_dir, val_dir, model_path, num_epochs, batch_size):
#     device = cfg.device

#     # tensorboard
#     writer = SummaryWriter()

#     # training data
#     train_set = PointcloudDataset(**cfg.get_dataset_args(train_dir))
#     if val_dir is not None and os.path.exists(val_dir):
#         val_set = PointcloudDataset(**cfg.get_dataset_args(val_dir))
#     else:
#         print('Using 20% of training data for validation')
#         train_set, val_set = torch.utils.data.random_split(train_set, [0.8, 0.2])
#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

#     # model
#     ae = cfg.create_vision_module().to(device)

#     # training
#     # TODO: find a more balanced ep and it so that it doesn't take forever to train but also matches the cube
#     # alternatively, figure out a way to guarantee that the cube points get matched in the auction algoirthm
#     # number of points must be the same and a multiple of 1024
#     loss_fn = EarthMoverDistance(eps=cfg.emd_eps, iterations=cfg.emd_iterations, classes=cfg.class_weights)
#     optimizer = torch.optim.Adam(ae.parameters(), lr=cfg.vision_lr)

#     # training loop
#     loss_min = np.inf
#     for epoch in range(num_epochs):
#         loss_training = 0.0

#         for i, (X, Y) in enumerate(train_loader):
#             X = X.to(device)
#             Y = Y.to(device)

#             # compute prediction and loss
#             pred = ae(X)
#             loss = loss_fn(pred, Y)

#             # backprop
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             loss_training += loss.item()

#             # validation
#             if i % cfg.val_every == cfg.val_every - 1:
#                 # TODO: unload X and Y?
#                 loss_validation = 0.0
#                 ae.train(False)
#                 with torch.no_grad():
#                     for X_val, Y_val in val_loader:
#                         X_val = X_val.to(device)
#                         Y_val = Y_val.to(device)
#                         pred_val = ae(X_val)
#                         loss_validation += loss_fn(pred_val, Y_val).item()
#                 ae.train(True)

#                 loss_training = loss_training / cfg.val_every
#                 loss_validation = loss_validation / len(val_loader)

#                 # log to tensorboard
#                 writer.add_scalars('PC_AE_Training', {
#                     'Training Loss': loss_training,
#                     'Validation Loss': loss_validation
#                 }, epoch * len(train_loader) + i)

#                 # save best model
#                 if loss_validation < loss_min:
#                     print(f'saving new best model: {loss_min} -> {loss_validation}')
#                     loss_min = loss_validation
#                     torch.save(ae.state_dict(), model_path)

#                 loss_training = 0.0

#         print(f"Epoch {epoch}: loss = {loss}")

#     # save model
#     torch.save(ae.state_dict(), model_path.replace('.pth', '_last.pth'))

#     # log model to tensorboard
#     batch = next(iter(train_loader))[0]
#     writer.add_graph(ae, batch.to(device))

#     writer.flush()


# def eval(model_path, val_dir, output_dir, batch_size):
#     device = cfg.device

#     # in order to get the indices of the data in each batch, we wrap the dataloader in a special class
#     # from https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19
#     def WithIndex(cls):
#         """
#         Modifies the given Dataset class to return a tuple data, target, index
#         instead of just data, target.
#         """
#         def __getitem__(self, index):
#             data, target = cls.__getitem__(self, index)
#             return data, target, index

#         return type(cls.__name__, (cls,), {
#             '__getitem__': __getitem__,
#         })
#     PointCloudDatasetWithIndex = WithIndex(PointcloudDataset)


#     # evaluate
#     eval_set = PointCloudDatasetWithIndex(**cfg.get_dataset_args(val_dir))
#     eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
#     loss_fn = EarthMoverDistance(eps=cfg.emd_eps, iterations=cfg.emd_iterations, classes=cfg.class_weights)
    
#     ae = cfg.create_vision_module().to(device)
#     ae.load_state_dict(torch.load(model_path))
#     ae.eval()

#     with torch.no_grad():
#         for X, Y, IDX in eval_loader:
#             X = X.to(device)
#             Y = Y.to(device)

#             pred = ae(X)
#             # print(f"pred = {pred}", f"Y = {Y}")

#             # eval
#             loss = loss_fn(pred, Y)
#             print(f"eval loss = {loss}")

#             # save as npz
#             # split into points and feature
#             pred = pred.detach().cpu().numpy()
#             points = pred[:, :, :3]
#             feat = pred[:, :, 3:]
#             for i in range(len(IDX)):
#                 p, s, idx = points[i], feat[i], IDX[i]
#                 path = os.path.join(output_dir, eval_set.filename(idx))

#                 np.savez(path, points=p, rgb=s, classes=np.array(cfg.classes, dtype=object))
