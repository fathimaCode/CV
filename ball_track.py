import numpy as np
import matplotlib.pyplot as plt
def kalman_filter(noisy_coords):
    # Initialization
    x_hat = np.array([noisy_coords[0][0], noisy_coords[0][1], 0, 0])  # Initial position estimate [x, y, vx, vy]
    P = np.eye(4)  # Initial state covariance matrix
    F = np.array([[1, 0, 0.2, 0], [0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]])  # Motion model matrix
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Observation model matrix

    # Covariance matrices (Q and R)
    Q = np.array([[0.16, 0, 0, 0],
              [0, 0.36, 0, 0],
              [0, 0, 0.16, 0],
              [0, 0, 0, 0.36]]) # Process noise covariance matrix
    R = np.eye(2) * 0.25  # Measurement noise covariance matrix

    estimated_coords = []
    for obs in noisy_coords:
        # Predict step
        x_hat = F.dot(x_hat)
        P = F.dot(P).dot(F.T) + Q

        # Update step
        y = np.array([obs[0] - H.dot(x_hat)[0], obs[1] - H.dot(x_hat)[1]])  # Residual
        S = H.dot(P).dot(H.T) + R  # Innovation covariance
        K = P.dot(H.T).dot(np.linalg.inv(S))  # Kalman gain

        x_hat = x_hat + K.dot(y)
        P = (np.eye(4) - K.dot(H)).dot(P)

        estimated_coords.append([x_hat[0], x_hat[1]])

    return np.array(estimated_coords)



noisy_coords = np.genfromtxt('na.csv', delimiter=',')  # Load noisy coordinates (x) from file
real_coords = np.genfromtxt('x-2.csv', delimiter=',')  # Load real coordinates (x) from file
noisy_coords_y = np.genfromtxt('nb.csv', delimiter=',')  # Load noisy coordinates (y) from file
real_coords_y = np.genfromtxt('y-2.csv', delimiter=',')  # Load real coordinates (y) from file

# Combine x and y coordinates
noisy_coords_combined = np.column_stack((noisy_coords, noisy_coords_y))
real_coords_combined = np.column_stack((real_coords, real_coords_y))

estimated_coords = kalman_filter(noisy_coords_combined)

# Plotting
# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(real_coords_combined[:, 0], real_coords_combined[:, 1], marker='*', color='g', label='Real Coordinates')
plt.scatter(noisy_coords_combined[:, 0], noisy_coords_combined[:, 1], marker='+', color='r', label='Noisy Coordinates')
plt.plot(estimated_coords[:, 0], estimated_coords[:, 1], 'b-', label='Estimated Coordinates')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kalman Filter - Estimated Trajectory')
plt.legend()
plt.grid(True)
plt.show()
