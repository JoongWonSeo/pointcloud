import torch
import torch.nn as nn
from pytorch3d import loss as pytorch3d_loss
import numpy as np
import pandas as pd
from models.pointnet import PointNetEncoder
from models.pn_autoencoder import PNAutoencoder

device = "cuda"


# 1 set (point cloud) of 2 points (0,0,0) and (1,1,1) of dim 3
#X = torch.tensor([[[0., 0., 0.], [1., 1., 1.]]])
original = pd.read_csv('input/table.csv')
#todo: uniform sampling
X = torch.tensor(original.loc[:, ['x', 'y', 'z']].to_numpy(), dtype=torch.float).to(device)
print(X)
X = X.reshape((1, -1, 3)) # batch of 1 pc
X = torch.vstack((X, X)) # batch of 2 pcs
print(X.shape)

# df = pd.DataFrame(X.cpu()[0], columns=['x', 'y', 'z'])
# df.to_csv('input.csv')

# models
ae = PNAutoencoder(2048, 3).to(device)

# training
loss_fn = pytorch3d_loss.chamfer_distance
optimizer = torch.optim.Adam(ae.parameters())

# training loop
for epoch in range(100):
    # load one batch
    # X = ...

    # compute prediction and loss
    pred = ae(X)
    loss, _ = loss_fn(pred, X)

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50:
        print(f"loss = {loss}")

#ae.eval() # FOR SOME REASON THIS MAKES THE MODEL OUTPUT VERY WEIRD, PROBABLY DUE TO SOME PARTS BEING TURNED OFF
with torch.no_grad():
    # encode
    embedding = ae.encoder(X)

    print(embedding.shape)
    print(embedding)

    # decode
    pred = torch.reshape(ae.decoder(embedding), (-1, ae.out_points, ae.dim_per_point))


    # eval
    loss, _ = pytorch3d_loss.chamfer_distance(pred, X)
    print(f"eval loss = {loss}")

    # save to csv
    df = pd.DataFrame(pred.cpu()[0], columns=['x', 'y', 'z'])
    df.to_csv('output/reconstructed.csv')



# save model
torch.save(ae.state_dict(), 'weights/model.pth')
# pn.eval()
# ae.eval()

# result = pn(X)
# reconst = ae(X)

# print(result)
# print(reconst)
