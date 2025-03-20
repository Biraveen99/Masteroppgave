# Define the new range for x-axis from 0 to 3600
x = np.linspace(0, 3600, 400)

# Define new mean and standard deviation to fit within y-range [10, 90]
mean_x = 1800  # Centering in the middle of x-axis
std_dev_x = 600  # Adjusted standard deviation for proper spread

# Compute the Gaussian distribution, scaled to fit the y-axis range [10, 90]
y = 10 + 80 * np.exp(-0.5 * ((x - mean_x) / std_dev_x) ** 2)

# Plot the Gaussian distribution with corrected scaling
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Gaussian Distribution', color='blue')
plt.xlim(0, 3600)
plt.ylim(10, 100)
plt.title("Gaussian Distribution (Centered in Y-range)")
plt.xlabel("time in seconds")
plt.ylabel("CPU utilization")
plt.legend()
plt.grid(True)
plt.show()
