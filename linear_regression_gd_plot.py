import numpy as np
import matplotlib.pyplot as plt

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
L = 0.01  # Learning rate from previous code
epochs = 1_000_000  # Increased for better convergence

m_hist = [m]
b_hist = [b]
loss_hist = [loss(m, b)]

for i in range(epochs):
    y_pred = m * X + b
    D_m = (-2 / n) * np.sum(X * (y - y_pred))
    D_b = (-2 / n) * np.sum(y - y_pred)
    m -= L * D_m
    b -= L * D_b
    m_hist.append(m)
    b_hist.append(b)
    loss_hist.append(loss(m, b))

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(M, B, Loss, cmap='viridis', alpha=0.8)
ax.plot(m_hist, b_hist, loss_hist, color='red', marker='.', markersize=1, linewidth=2, label='GD Path')
ax.set_xlabel('Slope (m)')
ax.set_ylabel('Intercept (b)')
ax.set_zlabel('Loss (MSE)')
ax.set_title('3D Loss Surface and Gradient Descent Path')
ax.legend()
plt.grid(True)
plt.savefig('linear_regression_gd_loss.png')

# Save and display
plt.show()