import time
import multiprocessing
import numpy as np

def generate_load(duration, intensity):
    """Generate CPU load for a given duration and intensity."""
    end_time = time.time() + duration
    while time.time() < end_time:
        _ = [x**2 for x in range(int(intensity * 1e6))]

def curve_anomaly(duration=1800):
    """Sinusoidal CPU utilization pattern over 1 hour."""
    print("Running Curve Anomaly Test...")
    steps = 60  # Number of cycles in 1 hour
    interval = duration / steps
    for i in range(steps):
        load = 50 + 40 * np.sin(i / 10)  # Sinusoidal fluctuation
        process = multiprocessing.Process(target=generate_load, args=(interval, load / 100))
        process.start()
        time.sleep(interval)
        process.terminate()

if __name__ == "__main__":
    curve_anomaly()
