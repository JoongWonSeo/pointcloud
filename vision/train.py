import cfg
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from vision.models.pn_autoencoder import PointcloudDataset, PNAutoencoder, PN2Autoencoder, PN2PosExtractor
from vision.utils import Normalize, EarthMoverDistance, get_class_points
from torch.utils.tensorboard import SummaryWriter

def mean_cube_pos(Y):
    mean_points = torch.zeros((Y.shape[0], 3))
    for i in range(Y.shape[0]):
        mean_points[i, :] = get_class_points(Y[i, :, :3], Y[i, :, 3:4], 1, len(cfg.classes)).mean(dim=0)
    return mean_points


def train(input_dir, model_path, num_epochs, batch_size, eps, iterations):
    device = cfg.device

    # tensorboard
    writer = SummaryWriter()

    # training data
    dataset = PointcloudDataset(root_dir=input_dir, files=None, in_features=['rgb'], out_features=['segmentation'])
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # model: XYZRGB -> XYZL (L = segmentation label)
    # ae = PNAutoencoder(2048, in_dim=6, out_dim=4).to(device)
    ae = PN2PosExtractor(6).to(device)

    # training
    # TODO: find a more balanced ep and it so that it doesn't take forever to train but also matches the cube
    # alternatively, figure out a way to guarantee that the cube points get matched in the auction algoirthm
    # number of points must be the same and a multiple of 1024
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-4)

    # training loop
    loss_min = np.inf
    for epoch in range(num_epochs):
        loss_training = 0.0

        for i, (X, Y) in enumerate(train_loader):
            X = X.to(device)
            Y = mean_cube_pos(Y).to(device) # the mean position of the cube

            # compute prediction and loss
            pred = ae(X)
            loss = loss_fn(pred, Y)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_training += loss.item()

            # validate every 4 batches (every 100 inputs)
            eval_every = 4
            if i % eval_every == eval_every - 1:
                loss_validation = 0.0
                ae.train(False)
                with torch.no_grad():
                    for X_val, Y_val in val_loader:
                        X_val = X_val.to(device)
                        Y_val = mean_cube_pos(Y_val).to(device)
                        pred_val = ae(X_val)
                        loss_validation += loss_fn(pred_val, Y_val).item()
                ae.train(True)

                loss_training = loss_training / eval_every
                loss_validation = loss_validation / len(val_loader)

                # log to tensorboard
                writer.add_scalars('PC_AE_Training', {
                    'Training Loss': loss_training,
                    'Validation Loss': loss_validation
                }, epoch * len(train_loader) + i)

                # save best model
                if loss_validation < loss_min:
                    print(f'saving new best model: {loss_min} -> {loss_validation}')
                    loss_min = loss_validation
                    torch.save(ae.state_dict(), model_path)

                loss_training = 0.0

        print(f"Epoch {epoch}: loss = {loss}")

    # save model
    torch.save(ae.state_dict(), model_path.replace('.pth', '_last.pth'))

    # log model to tensorboard
    batch = next(iter(train_loader))[0]
    writer.add_graph(ae, batch.to(device))

    writer.flush()


def eval(model_path, input_dir, output_dir, batch_size, eps=0.002, iterations=10000):
    device = cfg.device

    # in order to get the indices of the data in each batch, we wrap the dataloader in a special class
    # from https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19
    def WithIndex(cls):
        """
        Modifies the given Dataset class to return a tuple data, target, index
        instead of just data, target.
        """

        def __getitem__(self, index):
            data, target = cls.__getitem__(self, index)
            return data, target, index

        return type(cls.__name__, (cls,), {
            '__getitem__': __getitem__,
        })
    PointCloudDatasetWithIndex = WithIndex(PointcloudDataset)


    # evaluate
    eval_set = PointCloudDatasetWithIndex(root_dir=input_dir, files=None, in_features=['rgb'], out_features=['segmentation'])
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    loss_fn = torch.nn.MSELoss()
    
    # ae = PNAutoencoder(2048, in_dim=6, out_dim=4).to(device)
    ae = PN2PosExtractor(6).to(device)
    ae.load_state_dict(torch.load(model_path))
    ae.eval()

    with torch.no_grad():
        for X, Y, IDX in eval_loader:
            X = X.to(device)
            Y = mean_cube_pos(Y).to(device)

            pred = ae(X)
            print(f"pred = {pred}", f"Y = {Y}")

            # eval
            loss = loss_fn(pred, Y)
            print(f"eval loss = {loss}")

            # save as npz
            # split into points and seg
            # pred = pred.detach().cpu().numpy()
            # points = pred[:, :, :3]
            # seg = pred[:, :, 3:]
            # for i in range(len(IDX)):
            #     p, s, idx = points[i], seg[i], IDX[i]
            #     path = os.path.join(output_dir, eval_set.filename(idx))

            #     np.savez(path, points=p, segmentation=s, classes=np.array(cfg.classes, dtype=object))
