import torch

import matplotlib.pyplot as plt

from kan import *


torch.set_default_dtype(torch.float64)

# DEFINE DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# INITIALIZE MODEL
# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[2,5,1], grid=3, k=3, seed=42, device=device)

print(model)

# CREATE DATASET
# from kan.utils import create_dataset
# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, device=device)
# print(dataset)
print(type(dataset))
print(min(dataset['train_input'][:, [0]]), max(dataset['train_input'][:, [0]]))
print(min(dataset['train_input'][:, [1]]), max(dataset['train_input'][:, [1]]))
print("Shape of train input = ", dataset['train_input'].shape)
print("Shape of train label = ", dataset['train_label'].shape)

fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(xs=dataset['train_input'][:, [0]], ys=dataset['train_input'][:, [0]], zs=dataset['train_label'])
plt.tight_layout()
plt.savefig("hoang_dev/fig_dataset.jpg")

# PLOT KAN AT INITIALIZATION
model(dataset['train_input'])
model.plot(filename_save="hoang_dev/fig_model_init.jpg")

# TRAIN KAN WITH SPARSITY REGULARIZATION
# train the model
model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001, save_fig=True)
# model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.)

# PLOT TRAINED KAN
model.plot(filename_save="hoang_dev/fig_model_trained.jpg")

# PRUNE KAN AND REPLOT (KEEP THE ORIGINAL SHAPE)
model.prune()
model.plot(filename_save="hoang_dev/fig_model_pruned_OriginalShape.jpg")
