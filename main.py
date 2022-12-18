import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.pn_autoencoder import PNAutoencoder, PointcloudDataset
from train_utils import *


def train(input_dir, model_path, num_epochs, batch_size, eps, iter):
    # training data
    train_set = PointcloudDataset(root_dir=input_dir, files=None, transform=None)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # model
    ae = PNAutoencoder(2048, 6).to(device)

    # training 
    # TODO: find a more balanced ep and it so that it doesn't take forever to train but also matches the cube
    # alternatively, figure out a way to guarantee that the cube points get matched in the auction algoirthm
    loss_fn = earth_mover_distance(eps=eps, iterations=iter) # number of points must be the same and a multiple of 1024
    optimizer = torch.optim.Adam(ae.parameters())

    # training loop
    min_loss = np.inf
    for epoch in range(num_epochs):
        # load one batch
        for X in train_loader:
            X = X.to(device)
            
            # compute prediction and loss
            pred = ae(X)
            loss = loss_fn(pred, X)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # TODO evaluate on validation set
        print(f"{epoch}: loss = {loss}")
        # save best model
        # if loss < min_loss:
        #     min_loss = loss
        #     print('saving new best model')
        #     torch.save(ae.state_dict(), 'weights/best.pth')

    # save model
    torch.save(ae.state_dict(), model_path)


def eval(model_path, input_dir, output_dir, eps=0.002, iter=10000):
    # evaluate
    eval_set = PointcloudDataset(root_dir=input_dir, files=None, transform=None)
    loss_fn = earth_mover_distance(eps=eps, iterations=iter)

    ae = PNAutoencoder(2048, 6).to(device)
    ae.load_state_dict(torch.load(model_path))
    ae.eval()

    with torch.no_grad():
        for i in range(len(eval_set)):
            X = eval_set[i].to(device)
            X = X.reshape((1, ae.out_points, ae.dim_per_point)) # X to batch of size 1

            # encode
            embedding = ae.encoder(X)

            print(embedding)

            # decode
            pred = ae.decoder(embedding).reshape((-1, ae.out_points, ae.dim_per_point))

            # eval
            loss = loss_fn(pred, X)
            print(f"eval loss = {loss}")

            # save as npz
            # split into points and rgb
            pred = pred.detach().cpu().numpy()
            points = pred[0, :, :3]
            rgb = pred[0, :, 3:]
            np.savez(os.path.join(output_dir, eval_set.filename(i)), points=points, features=rgb)


def decoder_viewer(model_path):
    import open3d as o3d

    # load model
    ae = PNAutoencoder(2048, 6)
    ae.load_state_dict(torch.load(model_path))
    ae = ae.to(device)
    ae.eval()


    # define functions to update pointcloud and change x, y, z
    x, y, z = 0, 0, 0

    def make_xyz_changer(axis, inc):
        def xyz_changer(vis):
            nonlocal x, y, z
            if axis == 'x':
                x += inc
            elif axis == 'y':
                y += inc
            elif axis == 'z':
                z += inc
            else:
                raise ValueError(f'invalid axis: {axis}')
            print(f'{axis} = {x if axis == "x" else y if axis == "y" else z}')
            update_pointcloud()
            return True
        return xyz_changer

    def update_pointcloud():
        nonlocal x, y, z
        embedding = torch.Tensor([x, y, z]).to('cuda')
        decoded =  torch.reshape(ae.decoder(embedding), (-1, ae.out_points, ae.dim_per_point)).detach().cpu().numpy()
        points = decoded[0, :, :3]
        rgb = decoded[0, :, 3:]
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        return True

    # create visualizer and window.
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(height=480, width=640)

    # a, s, d to increment x, y, z; z, x, c to decrement x, y, z; using GLFW key codes
    vis.register_key_callback(65, make_xyz_changer('x', 0.1))
    vis.register_key_callback(83, make_xyz_changer('y', 0.1))
    vis.register_key_callback(68, make_xyz_changer('z', 0.1))
    vis.register_key_callback(90, make_xyz_changer('x', -0.1))
    vis.register_key_callback(88, make_xyz_changer('y', -0.1))
    vis.register_key_callback(67, make_xyz_changer('z', -0.1))

    # initialize pointcloud instance.
    pcd = o3d.geometry.PointCloud()
    update_pointcloud() # initial points
    vis.add_geometry(pcd) # include it in the visualizer before non-blocking visualization.

    # run non-blocking visualization. 
    keep_running = True
    while keep_running:
        keep_running = vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()


parser = argparse.ArgumentParser(description='Train or evaluate a pointnet autoencoder')
parser.add_argument('mode', choices=['train', 'eval', 'traineval', 'decoder'], help='train or evaluate the model')
parser.add_argument('--model', default='weights/last.pth', help='path to model weights (to save during training or load during evaluation)')
parser.add_argument('--input', default='prep', help='path to training data (for training) or input data (for evaluation)')
parser.add_argument('--output', default='output', help='path to output data (for evaluation)')
parser.add_argument('--device', default='cuda:0', help='device to use for training or evaluation')
parser.add_argument('--batch_size', default=25, type=int, help='batch size for training')
parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs to train for')
parser.add_argument('--eps', default=0.002, type=float, help='epsilon for EMD')
parser.add_argument('--iter', default=10000, type=int, help='number of iterations for EMD')
args = parser.parse_args()

device = args.device
print(f'device = {device}')

if args.mode == 'train':
    train(args.input, args.model, args.num_epochs, args.batch_size, args.eps, args.iter)
elif args.mode == 'eval':
    eval(args.model, args.input, args.output, args.eps, args.iter)
elif args.mode == 'traineval':
    train(args.input, args.model, args.num_epochs, args.batch_size, args.eps, args.iter)
    eval(args.model, args.input, args.output, args.eps, args.iter)
elif args.mode == 'decoder':
    decoder_viewer(args.model)