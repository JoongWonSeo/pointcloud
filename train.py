import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch3d import loss as pytorch3d_loss
import numpy as np
import pandas as pd
from models.pointnet import PointNetEncoder
from models.pn_autoencoder import PNAutoencoder, PointcloudDataset

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(f'device = {device}')


# training data
train_set = PointcloudDataset(root_dir='input', files=None, transform=None)
print(len(train_set))
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

# model
ae = PNAutoencoder(2048, 6).to(device)

# training parameters
loss_fn = pytorch3d_loss.chamfer_distance
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
        loss, _ = loss_fn(pred, X)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # TODO evaluate on validation set
    print(f"loss = {loss}")
    # save best model
    # if loss < min_loss:
    #     min_loss = loss
    #     print('saving new best model')
    #     torch.save(ae.state_dict(), 'weights/best.pth')


# save model
torch.save(ae.state_dict(), 'weights/last.pth')


# evaluate
eval_set = PointcloudDataset(root_dir='input', files=None, transform=None)
eval_loader = DataLoader(eval_set, batch_size=1, shuffle=True)

ae.eval() # FOR SOME REASON THIS MAKES THE MODEL OUTPUT VERY WEIRD, PROBABLY DUE TO SOME PARTS BEING TURNED OFF
# I THINK IT MAY HAVE TO DO WITH THE BATCH NORM LAYER
with torch.no_grad():
    for i, X in enumerate(eval_loader):
        X = X.to(device)

        # encode
        embedding = ae.encoder(X)

        print(embedding.shape)
        print(embedding)

        # decode
        pred = torch.reshape(ae.decoder(embedding), (-1, ae.out_points, ae.dim_per_point))

        # eval
        loss, _ = loss_fn(pred, X)
        print(f"eval loss = {loss}")

        # save as npz
        # split into points and rgb
        points = pred[0, :, :3].cpu().numpy()
        rgb = pred[0, :, 3:].cpu().numpy()
        np.savez(f'output/{i}.npz', points=points, rgb=rgb)


