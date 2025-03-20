import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ace_tools as tools

# Define the range for x-axis (time in seconds)
x = np.linspace(0, 3600, 400)

# Define mean and standard deviation to fit within y-range [10, 100]
mean_x = 1800  # Centering in the middle of x-axis
std_dev_x = 600  # Adjusted standard deviation for proper spread

# Compute the Gaussian distribution, scaled to fit the y-axis range [10, 90]
y = 10 + 80 * np.exp(-0.5 * ((x - mean_x) / std_dev_x) ** 2)

# Introduce noise to make the graph appear more realistic
stronger_noise = np.random.normal(0, 5, size=len(x))  # Increased std deviation for more variation

# Apply stronger noise to the y-values while keeping them within bounds
y_strong_noise = np.clip(y + stronger_noise, 10, 100)  # Ensuring values stay between 10 and 100

# Create a DataFrame to store the x (time) and y (CPU utilization) values
data = pd.DataFrame({'Time in Seconds': x, 'CPU Utilization (%)': y_strong_noise})

# Display the DataFrame to the user
tools.display_dataframe_to_user(name="CPU Utilization Data", dataframe=data)

# Plot the noisy Gaussian distribution
plt.figure(figsize=(8, 5))
plt.plot(x, y_strong_noise, label='Noisy Gaussian Distribution', color='blue')
plt.xlim(0, 3600)
plt.ylim(10, 100)  # Keeping the y-axis limit up to 100

# Update axis labels and title
plt.title("Realistic CPU Utilization Over Time with Stronger Noise")
plt.xlabel("Time in Seconds")
plt.ylabel("CPU Utilization (%)")

plt.legend()
plt.grid(True)
plt.show()

