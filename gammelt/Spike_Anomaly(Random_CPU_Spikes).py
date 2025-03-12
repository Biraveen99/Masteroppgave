import time
import random
import multiprocessing

def generate_load(duration, intensity):
    """Generate CPU load for a given duration and intensity."""
    end_time = time.time() + duration
    while time.time() < end_time:
        _ = [x**2 for x in range(int(intensity * 1e6))]

def spike_anomaly(duration=1800):
    """Randomly occurring CPU spikes over 1 hour."""
    print("Running Spike Anomaly Test...")
    end_time = time.time() + duration
    while time.time() < end_time:
        wait_time = random.randint(100, 200)  # Random wait before the next spike
        time.sleep(wait_time)
        process = multiprocessing.Process(target=generate_load, args=(10, 0.9))  # 90% CPU spike
        process.start()
        time.sleep(10)
        process.terminate()

if __name__ == "__main__":
    spike_anomaly()
