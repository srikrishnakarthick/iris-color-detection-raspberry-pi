from gpiozero import Button
from signal import pause
import subprocess
import time
import shutil
import os

# GPIO button on pin 17
button = Button(17, pull_up=True)

# Mounted Windows folder
windows_folder = "/mnt/pi_to_win"

def capture():
    # Timestamped filename
    filename = time.strftime("%Y-%m-%d_%H-%M-%S.jpg")
    local_path = os.path.join("/home/iiser", filename)

    print(f"Capturing {filename}...")

    # Capture image
    subprocess.run([
        "raspistill",
        "-o", local_path,
        "--width", "1920",
        "--height", "1080",
        "--nopreview"
    ])

    # Optional: open locally on Pi
    viewer = subprocess.Popen(["xdg-open", local_path])
    time.sleep(4)
    viewer.terminate()

    print("Photo saved locally.")

    # Copy to Windows share
    if os.path.ismount(windows_folder):
        dest_path = os.path.join(windows_folder, filename)
        shutil.copy(local_path, dest_path)
        print(f"Photo copied to Windows folder: {dest_path}")

        # Open photo on Windows (from Pi, via mounted folder)
        subprocess.run(["xdg-open", dest_path])
        print("Photo opened on Windows (from mounted folder).")
    else:
        print(f"Windows share {windows_folder} not mounted. Photo not copied.")

    print("Press the button to capture another photo.")

# Attach button
button.when_pressed = capture

print("Ready. Press the button to capture a photo.")
pause()
