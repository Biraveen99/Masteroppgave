import threading
import time
import psutil

def cpu_stress(load_percent, duration):
    """
    Function to generate sustained CPU load.
    :param load_percent: Target CPU utilization (0-100%).
    :param duration: Time in seconds to maintain the load.
    """
    print(f"[DEBUG] Running CPU load at {load_percent}% for {duration // 60} minutes...")
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

def plateau_test():
    """
    1. 15 min idle (normal)
    2. 30 min at 80% CPU
    3. 15 min idle (normal)
    """
    num_cores = psutil.cpu_count(logical=True)  # Get number of CPU cores

    print("[DEBUG] Starting plateau test...")
    
    # Step 1: 15 minutes normal (idle)
    print("[DEBUG] 15 minutes normal (idle)...")
    time.sleep(15 * 60)

    # Step 2: 30 minutes at 80% CPU
    print("[DEBUG] 30 minutes at 80% CPU load...")
    threads = []
    for _ in range(num_cores):  # One thread per core
        t = threading.Thread(target=cpu_stress, args=(80, 30 * 60))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()  # Wait for all threads to finish

    # Step 3: 15 minutes normal (idle)
    print("[DEBUG] 15 minutes normal (idle)...")
    time.sleep(15 * 60)

    print("[DEBUG] CPU plateau test completed!")

if __name__ == "__main__":
    plateau_test()
