# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:06:05 2024

@author: Dev
"""

import numpy as np
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt, cm
from scipy.stats.qmc import LatinHypercube
# %matplotlib inline

import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader


vertex1 = np.array([0, 0])
vertex2 = np.array([0, 2])
vertex3 = np.array([2, 2])
vertex4 = np.array([2, 0])

left_edge = np.array([vertex1, vertex2])
top_edge = np.array([vertex2, vertex3])
right_edge = np.array([vertex3, vertex4])
bottom_edge = np.array([vertex4,  vertex1])

lhc = LatinHypercube(d=1)

n_samples = 25
samples = lhc.random(n=n_samples)

def get_boundary_points(edge, t_values):
    return [edge[0] + t * (edge[1] - edge[0]) for t in t_values]

top_points, bottom_points, left_points, right_points = [], [], [], []

top_points.extend(get_boundary_points(top_edge, samples.flatten()))
bottom_points.extend(get_boundary_points(bottom_edge, samples.flatten()))
left_points.extend(get_boundary_points(left_edge, samples.flatten()))
right_points.extend(get_boundary_points(right_edge, samples.flatten()))

import matplotlib.pyplot as plt

vertices = np.array([vertex1, vertex2, vertex3, vertex4, vertex1])
plt.plot(vertices[:,0], vertices[:,1], 'r-')

for point in top_points:
    plt.plot(point[0], point[1], 'bo')


plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Points on 2D square Edges')
plt.axis('equal')
plt.legend()
plt.show()


def random_point_in_triangle(v1, v2, v3):
    weights = np.random.rand(3)
    weights /= weights.sum()
    x = weights[0] * v1[0] + weights[1] * v2[0] + weights[2] * v3[0]
    y = weights[0] * v1[1] + weights[1] * v2[1] + weights[2] * v3[1]
    return [x, y]


def generate_random_points_in_rhombus(vertex1, vertex2, vertex3, vertex4, n):
    points = []
    for _ in range(n):
        if np.random.rand() < 0.5:
            point = random_point_in_triangle(vertex1, vertex2, vertex3)
        else:
            point = random_point_in_triangle(vertex1, vertex3, vertex4)
        points.append(point)
    return np.array(points)

internal_points = generate_random_points_in_rhombus(vertex1, vertex2, vertex3,
                                                    vertex4, 10000)
nx = 41
ny = 41
nt = 100
nit = 50
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

rho = 1
nu = .1
dt = .001

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

def build_up_b(b, rho, dt, u, v, dx, dy):

    b[1:-1, 1:-1] = (
                      rho *
                      (
                        1 / dt *
                        (
                          (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                          (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)
                        ) -
                        (
                            (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)
                        )**2 -
                        2 *
                        (
                            (u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                            (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)
                        ) -
                        (
                            (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)
                        )**2
                      )
                    )

    return b
def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    pn = p.copy()

    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) *
                          b[1:-1,1:-1])

        p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
        p[-1, :] = 0        # p = 0 at y = 2

    return p

def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))

    for n in range(nt):
        un = u.copy()
        vn = v.copy()

        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                       (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                       (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        u[0, :]  = 0
        u[:, 0]  = 0
        u[:, -1] = 0
        u[-1, :] = 1    # set velocity on cavity lid equal to 1
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0


    return u, v, p
u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)
xx = np.linspace(0, X[0,:].shape[0], 22)[1:-1].astype(int)
yy = np.linspace(0, Y[0,:].shape[0], 22)[1:-1].astype(int)

points = np.array(list(product(xx, yy)))
pp = p[points[:,0], points[:,1]]
xx = points[:,1]
yy = points[:,0]

r_x = torch.tensor(X[0,:][xx], dtype = torch.float32)
r_y = torch.tensor(Y[:,0][yy], dtype = torch.float32)
r_p = torch.tensor(pp, dtype = torch.float32).reshape(-1,1)

p_data_X = torch.column_stack([r_x, r_y])
p_data_Y = r_p.reshape(-1, 1)
plt.scatter(r_x, r_y, c=r_p)
plt.colorbar()

class EdgePointsDataset(Dataset):
    def __init__(self, points):
        self.X = torch.tensor(points[0].astype(np.float32))
        self.y = torch.tensor(points[1].astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class InternalPointsDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X.astype(np.float32), requires_grad=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
    

class PressurePointsDataset(Dataset):
    def __init__(self, points):
        self.X = points[0]
        self.y = points[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Creating points list
edge_points_X = np.array(top_points + bottom_points + left_points + right_points)
edge_points_y = np.array([[1.0, 0.0, 0.0]]*len(top_points) + [[0.0, 0.0, 0.0]]*len(bottom_points) + [[0.0, 0.0, 0.0]]*len(left_points) + [[0.0, 0.0, 0.0]]*len(right_points))
edge_points = [edge_points_X, edge_points_y]
pressure_points = [p_data_X, p_data_Y]

# Dataset class
edge_points_dataset = EdgePointsDataset(edge_points)
internal_points_dataset = InternalPointsDataset(internal_points)
pressure_points_dataset = PressurePointsDataset(pressure_points)

# Dataloader
edge_points_dataloader = DataLoader(edge_points_dataset, batch_size=10, shuffle=True)
internal_points_dataloader = DataLoader(internal_points_dataset, batch_size=1000, shuffle=True)
pressure_points_dataloader = DataLoader(pressure_points_dataset, batch_size=40, shuffle=True)

class NeuralNet(nn.Module):
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        super(NeuralNet, self).__init__()
        self.input_shape = input_neurons
        self.output_shape = output_neurons
        self.hidden_neurons = hidden_neurons

        self.input_layer = nn.Linear(input_neurons, hidden_neurons)
        self.hidden_layer1 = nn.Linear(hidden_neurons, hidden_neurons)
        self.hidden_layer2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.hidden_layer3 = nn.Linear(hidden_neurons, hidden_neurons)
        self.output_layer = nn.Linear(hidden_neurons, output_neurons)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.tanh(x)
        x = self.hidden_layer1(x)
        x = self.tanh(x)
        x = self.hidden_layer2(x)
        x = self.tanh(x)
        x = self.hidden_layer3(x)
        x = self.tanh(x)
        x = self.output_layer(x)

        return x
# Initializing model parameters and other hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NeuralNet(2, 10, 3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)
loss_fn = nn.MSELoss()
# Printing the model summary
summary(model, (1, 2))

# Training the model on the above initialized points. 
loss_values, edge_loss_values, internal_loss_values, pressure_loss_values = [], [], [], []
epoch = 10000

for i in range(epoch):
    total_loss, edge_loss_total, internal_points_loss_total, pressure_points_loss_total = 0.0, 0.0, 0.0, 0.0
    for (edge_points, outputs), internal_points, (pressure_points, pressure_val) in zip(edge_points_dataloader, internal_points_dataloader, pressure_points_dataloader):
        optimizer.zero_grad()

        edge_points = edge_points.to(device)
        outputs = outputs.to(device)
        out = model(edge_points)
        edge_points_loss = loss_fn(out, outputs)

        internal_points = internal_points.to(device)
        out_internal = model(internal_points)

        du_dx = torch.autograd.grad(out_internal[:,0], internal_points, torch.ones_like(out_internal[:,0]), create_graph=True)[0][:, 0]
        du2_dx2 = torch.autograd.grad(du_dx, internal_points, torch.ones_like(du_dx), create_graph=True)[0][:, 0]

        du_dy = torch.autograd.grad(out_internal[:,0], internal_points, torch.ones_like(out_internal[:,0]), create_graph=True)[0][:, 1]
        du2_dy2 = torch.autograd.grad(du_dy, internal_points, torch.ones_like(du_dy), create_graph=True)[0][:, 1]

        dv_dx = torch.autograd.grad(out_internal[:,1], internal_points, torch.ones_like(out_internal[:,1]), create_graph=True)[0][:, 0]
        dv2_dx2 = torch.autograd.grad(dv_dx, internal_points, torch.ones_like(dv_dx), create_graph=True)[0][:, 0]

        dv_dy = torch.autograd.grad(out_internal[:,1], internal_points, torch.ones_like(out_internal[:,1]), create_graph=True)[0][:, 1]
        dv2_dy2 = torch.autograd.grad(dv_dy, internal_points, torch.ones_like(dv_dy), create_graph=True)[0][:, 1]

        dp_dx = torch.autograd.grad(out_internal[:,2], internal_points, torch.ones_like(out_internal[:,2]), create_graph=True)[0][:,0]
        dp2_dx2 = torch.autograd.grad(dp_dx, internal_points, torch.ones_like(dp_dx), create_graph=True)[0][:, 0]

        dp_dy = torch.autograd.grad(out_internal[:,2], internal_points, torch.ones_like(out_internal[:,2]), create_graph=True)[0][:,1]
        dp2_dy2 = torch.autograd.grad(dp_dy, internal_points, torch.ones_like(dp_dy), create_graph=True)[0][:,1]

        u = out_internal[:,0]
        v = out_internal[:,1]
        p = out_internal[:,2]

        eq1 = u * du_dx + v * du_dy + (1/rho) * dp_dx - nu * (du2_dx2+du2_dy2)
        eq2 = u * dv_dx + v * dv_dy + (1/rho) * dp_dy - nu * (dv2_dx2+dv2_dy2)
        eq3 = du_dx + dv_dy

        internal_points_loss = torch.mean(eq1**2) + torch.mean(eq2**2)  + torch.mean(eq3**2)

        pressure_points = pressure_points.to(device)
        pressure_val = pressure_val.to(device)
        out_pressure = model(pressure_points)
        pressure_loss = loss_fn(out_pressure[:, 2].reshape(40, 1), pressure_val)

        loss = edge_points_loss + internal_points_loss + pressure_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        edge_loss_total += edge_points_loss.item()
        internal_points_loss_total += internal_points_loss.item()
        pressure_points_loss_total += pressure_loss.item()

    loss_values.append(total_loss)
    edge_loss_values.append(edge_loss_total)
    internal_loss_values.append(internal_points_loss_total)
    pressure_loss_values.append(pressure_points_loss_total)

    if (i+1) % 1000 == 0:
        print(f"Epoch {i+1}, Training Loss: {total_loss}, Edge Points Loss: {edge_loss_total}, Internal Points Loss: {internal_points_loss_total}, Pressure Points Loss: {pressure_points_loss_total}")

plt.plot(loss_values, label="netloss")
plt.plot(edge_loss_values, label="loss1")
plt.plot(internal_loss_values, label="loss2")
plt.plot(pressure_loss_values, label="loss3")
plt.legend()
plt.save("Output.png")