# Increase noise intensity for a more realistic fluctuation
stronger_noise = np.random.normal(0, 5, size=len(x))  # Increased std deviation to 5

# Apply stronger noise to the y-values while keeping them within bounds
y_strong_noise = np.clip(y + stronger_noise, 10, 100)  # Ensuring values stay between 10 and 100

# Plot the Gaussian distribution with stronger noise
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
