import time
import multiprocessing

def generate_load(duration, intensity):
    """Generate CPU load for a given duration and intensity."""
    end_time = time.time() + duration
    while time.time() < end_time:
        _ = [x**2 for x in range(int(intensity * 1e6))]

def spike(duration=1800, load=90):
    """Repeated short spikes of high CPU usage for 1 hour."""
    print("Running Spike Test...")
    end_time = time.time() + duration
    while time.time() < end_time:
        process = multiprocessing.Process(target=generate_load, args=(10, load / 100))
        process.start()
        time.sleep(10)  # Spike duration
        process.terminate()
        time.sleep(20)  # Cooldown before next spike

if __name__ == "__main__":
    spike()
