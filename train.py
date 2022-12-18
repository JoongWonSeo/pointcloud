import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import numpy as np
from models.pn_autoencoder import PNAutoencoder, PointcloudDataset
from train_utils import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(f'device = {device}')


# training data
train_set = PointcloudDataset(root_dir='prep', files=None, transform=None)
train_loader = DataLoader(train_set, batch_size=25, shuffle=True)

# model
ae = PNAutoencoder(2048, 6).to(device)

# training parameters
# loss_fn = chamfer_distance(bbox = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2])
# TODO: find a more balanced ep and it so that it doesn't take forever to train but also matches the cube
# alternatively, figure out a way to guarantee that the cube points get matched in the auction algoirthm
loss_fn = earth_mover_distance(train=False) # number of points must be the same and a multiple of 1024
optimizer = torch.optim.Adam(ae.parameters())
num_epochs = 100

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
torch.save(ae.state_dict(), 'weights/last.pth')


# evaluate
eval_set = PointcloudDataset(root_dir='prep', files=None, transform=None)
eval_loader = DataLoader(eval_set, batch_size=1, shuffle=True)
# loss_fn = earth_mover_distance(train=False)

ae.eval()
with torch.no_grad():
    for i, X in enumerate(eval_loader):
        X = X.to(device)

        # encode
        embedding = ae.encoder(X)

        # print(embedding.shape)
        print(embedding)

        # decode
        pred = torch.reshape(ae.decoder(embedding), (-1, ae.out_points, ae.dim_per_point))

        # eval
        loss = loss_fn(pred, X)
        print(f"eval loss = {loss}")

        # save as npz
        # split into points and rgb
        points = pred[0, :, :3].cpu().numpy()
        rgb = pred[0, :, 3:].cpu().numpy()
        np.savez(f'output/{i}.npz', points=points, features=rgb)


