import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# definerer x aksen
x = np.arange(0, 3600)

# Define y-axis values based on the given plateau structure
y = np.piecewise(x, [x < 900, (x >= 900) & (x < 2700), x >= 2700], [10, 90, 10])

# Create a DataFrame with x and y values
df = pd.DataFrame({'X': x, 'Y': y})

# Define file paths
csv_file_path = "/Users/biraveennedunchelian/Documents/Masteroppgave/Masteroppgave/Newest addition/Plateu/plateau_log.csv"
png_file_path = "/Users/biraveennedunchelian/Documents/Masteroppgave/Masteroppgave/Newest addition/Plateu/plateau_graph.png"

# Save DataFrame as a CSV log file
df.to_csv(csv_file_path, index=False)

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(x, y, linestyle='-', color='b')
plt.xlabel('X-axis (0 to 3600)')
plt.ylabel('Y-axis (10 to 90)')
plt.title('Plateau Graph')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig(png_file_path)  # Save as PNG
plt.close()

# Output file paths
print(f"CSV Log File saved at: {csv_file_path}")
print(f"PNG Graph saved at: {png_file_path}")