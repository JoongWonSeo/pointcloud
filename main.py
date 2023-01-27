import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from models.pn_autoencoder import PNAutoencoder, PointcloudDataset
from train_utils import *
from torch.utils.tensorboard import SummaryWriter



def preprocess(input_dir, output_dir, num_points=2048):
    dataset = PointcloudDataset(root_dir=input_dir, files=None, transform=Compose([
        SampleFurthestPoints(num_points),
        Normalize((-0.5, 0.5, -0.5, 0.5, 0.5, 1.5)), # 3D bounding box x_min, x_max, y_min, y_max, z_min, z_max
    ]))

    for i in range(len(dataset)):
        dataset.save(i, output_dir)


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
    bbox = Normalize((-0.5, 0.5, -0.5, 0.5, 0.5, 1.5))(torch.Tensor([[-0.4, -0.4, 0.8],[0.4, 0.4, 1.5]])).T.reshape((6))
    loss_fn = earth_mover_distance(eps=eps, iterations=iterations, bbox=bbox, bbox_bonus=10)
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
    loss_fn = earth_mover_distance(eps=eps, iterations=iterations)

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



parser = argparse.ArgumentParser(
    description='Train or evaluate a pointnet autoencoder')
parser.add_argument('mode', choices=[
                    'train', 'eval', 'traineval', 'preprocess'], help='train or evaluate the model')
parser.add_argument('--model', default='weights/PC_AE.pth',
                    help='path to model weights (to save during training or load during evaluation)')
parser.add_argument('--input', default='prep',
                    help='path to training data (for training) or input data (for evaluation)')
parser.add_argument('--output', default='output',
                    help='path to output data (for evaluation)')
parser.add_argument('--device', default='cuda:0',
                    help='device to use for training or evaluation')
parser.add_argument('--batch_size', default=25, type=int,
                    help='batch size for training')
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of epochs to train for')
parser.add_argument('--eps', default=0.002, type=float, help='epsilon for EMD')
parser.add_argument('--iterations', default=5000, type=int,
                    help='number of iterations for EMD')
args = parser.parse_args()

device = args.device
print(f'device = {device}')

if args.mode == 'train':
    train(args.input, args.model, args.num_epochs,
          args.batch_size, args.eps, args.iterations)
elif args.mode == 'eval':
    eval(args.model, args.input, args.output, args.eps, args.iterations)
elif args.mode == 'traineval':
    train(args.input, args.model, args.num_epochs,
          args.batch_size, args.eps, args.iterations)
    eval(args.model, args.input, args.output, args.eps, args.iterations)
elif args.mode == 'preprocess':
    preprocess(args.input, args.output)
    
