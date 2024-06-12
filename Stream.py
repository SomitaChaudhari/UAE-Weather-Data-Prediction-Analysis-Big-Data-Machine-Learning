import shutil
import time
import os

# Source CSV file path
source_csv = "path"

# Destination folder path
destination_folder = "path"

# Function to periodically copy the CSV file
def copy_csv_periodically():
    while True:
        # Generate a new file name or timestamp
        timestamp = str(int(time.time()))
        destination_csv = os.path.join(destination_folder, f"source_{timestamp}.csv")

        # Copy the source CSV file to the destination folder
        shutil.copyfile(source_csv, destination_csv)
        print(f"Copied {source_csv} to {destination_csv}")

        # Sleep for some time (e.g., 10 seconds) before copying again
        time.sleep(10)

# Start copying the CSV file periodically
copy_csv_periodically()
