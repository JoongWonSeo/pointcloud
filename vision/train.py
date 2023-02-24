import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from vision.models.pn_autoencoder import PNAutoencoder, PointcloudDataset
from vision.utils import *
from torch.utils.tensorboard.writer import SummaryWriter


def train(input_dir, model_path, num_epochs, batch_size, eps, iterations):
    # tensorboard
    writer = SummaryWriter()

    # training data
    dataset = PointcloudDataset(root_dir=input_dir, files=None, transform=None)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # model
    ae = PNAutoencoder(2048, 6).to(device)

    # training
    # TODO: find a more balanced ep and it so that it doesn't take forever to train but also matches the cube
    # alternatively, figure out a way to guarantee that the cube points get matched in the auction algoirthm
    # number of points must be the same and a multiple of 1024
    bbox = Normalize([[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]])(torch.Tensor([[-0.4, -0.4, 0.8],[0.4, 0.4, 1.5]])).T.reshape((6))
    loss_fn = EarthMoverDistance(eps=eps, iterations=iterations, bbox=bbox, bbox_bonus=10)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-4)

    # training loop
    loss_min = np.inf
    for epoch in range(num_epochs):
        loss_training = 0.0

        for i, X in enumerate(train_loader):
            X = X.to(device)

            # compute prediction and loss
            pred = ae(X)
            loss = loss_fn(pred, X)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_training += loss.item()

            # validate every 4 batches (every 100 inputs)
            if i % 4 == 0:
                loss_validation = 0.0
                ae.train(False)
                with torch.no_grad():
                    for X_val in val_loader:
                        X_val = X_val.to(device)
                        pred_val = ae(X_val)
                        loss_validation += loss_fn(pred_val, X_val).item()
                ae.train(True)

                loss_training = loss_training / 4
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
    batch = next(iter(train_loader))
    writer.add_graph(ae, batch.to(device))

    writer.flush()


def eval(model_path, input_dir, output_dir, eps=0.002, iterations=10000):
    # evaluate
    eval_set = PointcloudDataset(
        root_dir=input_dir, files=None, transform=None)
    loss_fn = EarthMoverDistance(eps=eps, iterations=iterations)

    ae = PNAutoencoder(2048, 6).to(device)
    ae.load_state_dict(torch.load(model_path))
    ae.eval()

    with torch.no_grad():
        for i in range(len(eval_set)):
            X = eval_set[i].to(device)
            # X to batch of size 1
            X = X.reshape((1, ae.out_points, ae.dim_per_point))

            # encode
            embedding = ae.encoder(X)

            print(embedding)

            # decode
            pred = ae.decoder(embedding).reshape(
                (-1, ae.out_points, ae.dim_per_point))

            # eval
            loss = loss_fn(pred, X)
            print(f"eval loss = {loss}")

            # save as npz
            # split into points and rgb
            pred = pred.detach().cpu().numpy()
            points = pred[0, :, :3]
            rgb = pred[0, :, 3:]
            np.savez(os.path.join(output_dir, eval_set.filename(i)),
                     points=points, features=rgb)
