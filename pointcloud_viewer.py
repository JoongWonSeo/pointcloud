import numpy as np
import pandas as pd
import open3d as o3d

while True:
    df = pd.read_csv('point_cloud.csv')
    pc = df.loc[:, ['y', 'x', 'z']].to_numpy()
    print(pc)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    o3d.visualization.draw_geometries([pcd])