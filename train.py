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


# 1 set (point cloud) of 2 points (0,0,0) and (1,1,1) of dim 3
#X = torch.tensor([[[0., 0., 0.], [1., 1., 1.]]])
# original = pd.read_csv('input/table.csv')
# #todo: uniform sampling
# X = torch.tensor(original.loc[:, ['x', 'y', 'z']].to_numpy(), dtype=torch.float).to(device)
# print(X)
# X = X.reshape((1, -1, 3)) # batch of 1 pc
# X = torch.vstack((X, X)) # batch of 2 pcs
# print(X.shape)

# df = pd.DataFrame(X.cpu()[0], columns=['x', 'y', 'z'])
# df.to_csv('input.csv')

train_set = PointcloudDataset(root_dir='input', files=None, transform=None)
print(len(train_set))
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

# models
ae = PNAutoencoder(2048, 6).to(device)

# training
loss_fn = pytorch3d_loss.chamfer_distance
optimizer = torch.optim.Adam(ae.parameters())

# training loop
for epoch in range(3):
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

    #if epoch % 50:
    print(f"loss = {loss}")


# save model
torch.save(ae.state_dict(), 'weights/model.pth')


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


