import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3 * X.flatten() + 2 + np.random.randn(100) * 2

n = X.shape[0]  # number of data points

# Initialize model parameters
m = 0.0
b = 0.0

L = 0.0001  # Learning rate
epochs = 1_000_000  # Number of iterations for gradient descent

# Performing Gradient Descent (using entire dataset)
for i in range(epochs):
    # Predicted values for all data points
    Y_pred = m * X.flatten() + b

    # Compute gradients using all data points
    D_m = (-2 / n) * np.sum(X.flatten() * (y - Y_pred))
    D_b = (-2 / n) * np.sum(y - Y_pred)

    # Update parameters
    m = m - L * D_m
    b = b - L * D_b

    # Print progress every 100 iterations
    if i % 100 == 0:
        print(f"Iteration {i}: m = {m:.4f}, b = {b:.4f}")

print(f"Final model: y = {m:.2f}x + {b:.2f}")

# Create the plot
plt.scatter(X, y, color='blue', label='Data points')
plt.plot([0, 10], [b, m * 10 + b], color='red', label=f'Regression line (y = {m:.2f}x + {b:.2f})')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression (Gradient Descent)')
plt.legend()
plt.grid(True)

# Save and display the plot
plt.savefig('linear_regression_gd.svg')
plt.show()