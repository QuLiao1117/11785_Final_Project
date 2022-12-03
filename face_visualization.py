import pandas as pd
import numpy as np
import os
import torchaudio
import torch
import matplotlib.pyplot as plt
import open3d as o3d

root_path = "/penstate_data/download"

face_vertices = pd.read_csv(root_path + "/qls.csv")

vertices = face_vertices.values[0][1:]
length = len(vertices)
x = vertices[:int(length/3)]
y = vertices[int(length/3):int(2 * length/3)]
z = vertices[int(2 * length/3):]

list_pcd = np.array([x, y, z])

np_pcd = np.asarray(list_pcd)
pcd = o3d.geometry.PointCloud()
v3d = o3d.utility.Vector3dVector
pcd.points = v3d(np_pcd)

o3d.io.write_point_cloud("face_point_cloud.ply", pcd, write_ascii=False, compressed=False, print_progress=False)