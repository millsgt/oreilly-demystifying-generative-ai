import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100) * 10
y = 3 * X + 2 + np.random.randn(100) * 2

n = len(X)

# Loss function
def loss(m, b):
    y_pred = m * X + b
    return np.mean((y - y_pred)**2)

# Create grid for surface
m_range = np.linspace(0, 6, 50)
b_range = np.linspace(-4, 8, 50)
M, B = np.meshgrid(m_range, b_range)
Loss = np.array([[loss(m_val, b_val) for m_val in m_range] for b_val in b_range])

# Run GD and collect history
m = 0.0
b = 0.0
L_gd = 0.01
epochs_gd = 10000

m_hist_gd = [m]
b_hist_gd = [b]
loss_hist_gd = [loss(m, b)]

for i in range(epochs_gd):
    y_pred = m * X + b
    D_m = (-2 / n) * np.sum(X * (y - y_pred))
    D_b = (-2 / n) * np.sum(y - y_pred)
    m -= L_gd * D_m
    b -= L_gd * D_b
    m_hist_gd.append(m)
    b_hist_gd.append(b)
    loss_hist_gd.append(loss(m, b))

# Run SGD and collect history
np.random.seed(42)  # Reset seed for reproducibility
m = 0.0
b = 0.0
L_sgd = 0.01
epochs_sgd = 10000
collect_interval = 100  # Collect every 100 iterations to reduce points

m_hist_sgd = [m]
b_hist_sgd = [b]
loss_hist_sgd = [loss(m, b)]

for i in range(epochs_sgd):
    idx = np.random.randint(n)
    x_i = X[idx]
    y_i = y[idx]
    y_pred_i = m * x_i + b
    D_m = -2 * (y_i - y_pred_i) * x_i
    D_b = -2 * (y_i - y_pred_i)
    m -= L_sgd * D_m
    b -= L_sgd * D_b
    if i % collect_interval == 0:
        m_hist_sgd.append(m)
        b_hist_sgd.append(b)
        loss_hist_sgd.append(loss(m, b))

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(M, B, Loss, cmap='viridis', alpha=0.8)
ax.plot(m_hist_gd, b_hist_gd, loss_hist_gd, color='red', marker='.', markersize=1, linewidth=2, label='GD Path')
ax.plot(m_hist_sgd, b_hist_sgd, loss_hist_sgd, color='blue', marker='.', markersize=1, linewidth=1, label='SGD Path')
ax.set_xlabel('Slope (m)')
ax.set_ylabel('Intercept (b)')
ax.set_zlabel('Loss (MSE)')
ax.set_title('3D Loss Surface with Gradient Descent and Stochastic GD Paths')
ax.legend()
plt.grid(True)

# Save and display
plt.savefig('loss_surface_3d_gd_sgd.svg')
plt.show()