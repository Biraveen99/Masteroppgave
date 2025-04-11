import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

# Definerer x-aksen
x = np.linspace(0, 3600, 3600)

# Define mean and standard deviation for Gaussian curve
mean = 1800  # midten av kurven
std_dev = 900  # standard deviation for å sikre spread 

# mekker Gaussian curve
y = norm.pdf(x, mean, std_dev)

# scales y-axis so it fits between 10 and 90
y = 10 + (y - np.min(y)) * (80 / (np.max(y) - np.min(y)))

# mekker random noise i y intervallet mellom -5 til 5
noise = np.random.uniform(-5, 5, size=y.shape)
y_noisy = y + noise

# creates a DataFrame with x and y_noisy values, for lagring
df = pd.DataFrame({'X': x, 'Y_noisy': y_noisy})

# save the DataFrame as a log file (CSV format with .log extension)
log_filename = "gaussian_curve.log"
df.to_csv(log_filename, index=False, sep='\t')  # Using tab separator for better readability

# Plot the noisy Gaussian curve
plt.figure(figsize=(10, 5))
plt.plot(x, y_noisy, label="Noisy Gaussian Curve", color='r')
plt.xlabel("X-axis (0 to 3600)")
plt.ylabel("Y-axis (10 to 90)")
plt.title("Gaussian Curve with Noise")
plt.legend()
plt.grid()

# Save the plot as an image file
plot_filename = "gaussian_curve.png"
plt.savefig(plot_filename, dpi=300)  # Save as high-resolution image
plt.show()

print(f"Data saved as {log_filename}")
print(f"Plot saved as {plot_filename}")
