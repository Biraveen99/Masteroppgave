import threading
import time
import psutil
import numpy as np

def cpu_stress(load_percent, duration):
    """
    Function to generate CPU load.
    :param load_percent: Target CPU utilization (0-100%).
    :param duration: Time in seconds to maintain the load.
    """
    start_time = time.time()
    load_time = load_percent / 100  # Active time ratio
    sleep_time = 1 - load_time      # Idle time ratio

    while time.time() - start_time < duration:
        start_cycle = time.time()
        while (time.time() - start_cycle) < load_time:
            pass  # Busy loop for load
        time.sleep(sleep_time)  # Rest to balance load

def gaussian_curve_test():
    """
    Runs a Gaussian-like CPU load test over 1 hour.
    """
    duration = 3600  # 1 hour in seconds
    num_cores = psutil.cpu_count(logical=True)  # Get number of CPU cores

    # Generate Gaussian-like curve (bell shape) for CPU load
    x = np.linspace(-3, 3, num=duration)  # Standard Gaussian scale
    y = 10 + (80 * np.exp(-x**2))  # Gaussian function scaled to 10%-90%-10%
    
    print("[DEBUG] Starting Gaussian CPU test...")

    for i, load in enumerate(y):
        threads = []
        for _ in range(num_cores):  # One thread per core
            t = threading.Thread(target=cpu_stress, args=(load, 1))  # 1 sec per step
            t.start()
            threads.append(t)

        for t in threads:
            t.join()  # Ensure each second of load completes

        if i % 600 == 0:  # Print progress every 10 minutes
            print(f"[DEBUG] Time: {i//60} min, CPU Load: {int(load)}%")

    print("[DEBUG] CPU Gaussian test completed!")

if __name__ == "__main__":
    gaussian_curve_test()
