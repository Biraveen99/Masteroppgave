import time
import multiprocessing
import numpy as np

def generate_load(duration, intensity):
    """Generate CPU load for a given duration and intensity."""
    end_time = time.time() + duration
    while time.time() < end_time:
        _ = [x**2 for x in range(int(intensity * 1e6))]

def slope_anomaly(duration=1800, max_load=90):
    """Gradually increasing CPU utilization over 1 hour."""
    print("Running Slope Anomaly Test...")
    steps = 60  # Number of steps in the slope
    interval = duration / steps  # Time per step
    for load in np.linspace(10, max_load, steps):
        process = multiprocessing.Process(target=generate_load, args=(interval, load / 100))
        process.start()
        time.sleep(interval)
        process.terminate()

if __name__ == "__main__":
    slope_anomaly()
