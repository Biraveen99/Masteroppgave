import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# directory to save files
save_dir = "/Users/biraveennedunchelian/Documents/Masteroppgave/Masteroppgave/Newest addition/"

# Ensure the directory exists(bare for å sjekke egt)
os.makedirs(save_dir, exist_ok=True)

# Define file paths for å lage 
log_file_path = os.path.join(save_dir, "sinusoidal_log.csv")
plot_file_path = os.path.join(save_dir, "sinusoidal_plot.png")

# definerer x  aksen fra 0 til 3600
x = np.linspace(0, 3600, 3600)

# definerer y aksen og mekker en sinus kurce
y = 50 + 40 * np.sin((2 * np.pi / 1800) * x)  # for å få 2 peaks

# mekker noise mellom -5 og 5 
noise = np.random.uniform(-5, 5, size=x.shape)
y_noisy = y + noise

# Create a DataFrame and save as a CSV log file
df = pd.DataFrame({'X': x, 'Y_Noisy': y_noisy})
df.to_csv(log_file_path, index=False)

# Plot the noisy sinusoidal curve and save it as a PNG file
plt.figure(figsize=(10, 5))
plt.plot(x, y_noisy, label="Noisy Sinusoidal Curve with 2 Peaks", color='r')
plt.xlabel("X-axis (0 to 3600)")
plt.ylabel("Y-axis (10 to 90)")
plt.title("Noisy Sinusoidal Curve with Two Peaks")
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig(plot_file_path, dpi=300)

# Show the plot
plt.show()

print(f"Log file saved at: {log_file_path}")
print(f"Plot saved at: {plot_file_path}")
