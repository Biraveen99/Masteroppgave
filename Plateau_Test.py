import time
import multiprocessing

def generate_load(duration, intensity):
    """Generate constant CPU load for a given duration and intensity."""
    end_time = time.time() + duration
    while time.time() < end_time:
        _ = [x**2 for x in range(int(intensity * 1e6))]

def plateau(duration=3600, load=70):
    """Sustained high CPU usage for 1 hour."""
    print("Running a plateau test...")
    process = multiprocessing.Process(target=generate_load, args=(duration, load / 100))
    process.start()
    time.sleep(duration)
    process.terminate()

if __name__ == "__main__":
    plateau()
