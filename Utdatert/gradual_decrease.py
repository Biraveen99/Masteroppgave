import threading
import time
import psutil

def cpu_stress(load_percent, duration):
    """
    Function to generate CPU load dynamically.
    :param load_percent: Target CPU utilization (0-100%).
    :param duration: Time in seconds to maintain the load.
    """
    print(f"[DEBUG] Setting CPU load to {load_percent}% for {duration // 60} minutes...")
    start_time = time.time()
    load_time = load_percent / 100  # Active time ratio
    sleep_time = 1 - load_time      # Idle time ratio

    while time.time() - start_time < duration:
        start_cycle = time.time()
        while (time.time() - start_cycle) < load_time:
            pass  # Busy loop for load
        time.sleep(sleep_time)  # Sleep to balance CPU load
        elapsed = int(time.time() - start_time)
        if elapsed % 60 == 0:  # Print progress every minute
            print(f"[DEBUG] CPU load {load_percent}% running for {elapsed // 60} minutes...")

def gradual_cpu_load():
    """
    Gradually increases CPU usage from 90% to 10% over 1 hour.
    """
    load_levels = [90, 80, 70, 60, 50, 40, 30, 20, 10]  # CPU utilization levels
    total_duration = 3600  # 1 hour in seconds
    step_duration = total_duration // len(load_levels)  # Time per step

    print("[DEBUG] Starting gradual CPU benchmark...")

    for load in load_levels:
        threads = []
        num_cores = psutil.cpu_count(logical=True)  # Get number of CPU cores

        print(f"[DEBUG] Increasing CPU load to {load}%...")
        for _ in range(num_cores):  # Spawn a thread per core
            t = threading.Thread(target=cpu_stress, args=(load, step_duration))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()  # Wait for all threads before increasing load

    print("[DEBUG] CPU benchmark completed!")

if __name__ == "__main__":
    gradual_cpu_load()
