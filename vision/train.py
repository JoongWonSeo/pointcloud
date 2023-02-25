import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from vision.models.pn_autoencoder import PNAutoencoder, PointcloudDataset
from vision.utils import Normalize, EarthMoverDistance
from torch.utils.tensorboard import SummaryWriter


def train(input_dir, model_path, num_epochs, batch_size, eps, iterations, device='cuda:0'):
    # tensorboard
    writer = SummaryWriter()

    # training data
    dataset = PointcloudDataset(root_dir=input_dir, files=None, in_features=['rgb'], out_features=['segmentation'])
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # model: XYZRGB -> XYZL (L = segmentation label)
    ae = PNAutoencoder(2048, in_dim=6, out_dim=4).to(device)

    # training
    # TODO: find a more balanced ep and it so that it doesn't take forever to train but also matches the cube
    # alternatively, figure out a way to guarantee that the cube points get matched in the auction algoirthm
    # number of points must be the same and a multiple of 1024
    # bbox = Normalize([[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]])(torch.Tensor([[-0.4, -0.4, 0.8],[0.4, 0.4, 1.5]])).T.reshape((6))
    classes = [ # name and training weight
        ('env', 1.0),
        ('cube', 10.0),
        ('arm', 0.5),
        ('base', 0.5),
        ('gripper', 2.0),
    ]
    loss_fn = EarthMoverDistance(eps=eps, iterations=iterations, classes=classes)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-4)

    # training loop
    loss_min = np.inf
    for epoch in range(num_epochs):
        loss_training = 0.0

        for i, (X, Y) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)

            # compute prediction and loss
            pred = ae(X)
            loss = loss_fn(pred, Y)

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
                    for X_val, Y_val in val_loader:
                        X_val = X_val.to(device)
                        Y_val = Y_val.to(device)
                        pred_val = ae(X_val)
                        loss_validation += loss_fn(pred_val, Y_val).item()
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
    batch = next(iter(train_loader))[0]
    writer.add_graph(ae, batch.to(device))

    writer.flush()


def eval(model_path, input_dir, output_dir, eps=0.002, iterations=10000, device='cuda:0'):
    # evaluate
    eval_set = PointcloudDataset(root_dir=input_dir, files=None, in_features=['rgb'], out_features=['segmentation'])
    loss_fn = EarthMoverDistance(eps=eps, iterations=iterations)
    
    ae = PNAutoencoder(2048, in_dim=6, out_dim=4).to(device)
    ae.load_state_dict(torch.load(model_path))
    ae.eval()

    with torch.no_grad():
        for i in range(len(eval_set)):
            X, Y = eval_set[i]
            X = X.to(device)
            Y = Y.to(device)
            # to batch of size 1
            X = X.reshape((1, ae.out_points, ae.in_dim))
            Y = Y.reshape((1, ae.out_points, ae.out_dim))

            # encode
            embedding = ae.encoder(X)

            print(embedding)

            # decode
            pred = ae.decoder(embedding).reshape((-1, ae.out_points, ae.out_dim))

            # eval
            loss = loss_fn(pred, Y)
            print(f"eval loss = {loss}")

            # save as npz
            # split into points and seg
            pred = pred.detach().cpu().numpy()
            points = pred[0, :, :3]
            seg = pred[0, :, 3:]
            np.savez(os.path.join(output_dir, eval_set.filename(i)), points=points, segmentation=seg)
