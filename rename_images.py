import os
import datetime

# ─────────────────────────────────────────
# AUTO RENAME IMAGES BY DATE CREATED
# Renames to CAL_001.jpg, CAL_002.jpg ...
# in the order the photos were taken
# ─────────────────────────────────────────

image_folder = 'images'
valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

# Get all image files in the folder
image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith(valid_extensions)
]

# Sort by date created (oldest first)
image_files.sort(key=lambda f: os.path.getctime(os.path.join(image_folder, f)))

print(f"Found {len(image_files)} images. Renaming by date created...")
print()

# ── Step 1: Rename all to TEMP names first ──
for index, filename in enumerate(image_files, start=1):
    old_path  = os.path.join(image_folder, filename)
    temp_name = f"TEMP_{index:03d}.jpg"
    temp_path = os.path.join(image_folder, temp_name)
    os.rename(old_path, temp_path)

# ── Step 2: Rename TEMP to final CAL names ──
temp_files = sorted([
    f for f in os.listdir(image_folder)
    if f.startswith("TEMP_")
])

for index, filename in enumerate(temp_files, start=1):
    temp_path = os.path.join(image_folder, filename)
    new_name  = f"CAL_{index:03d}.jpg"
    new_path  = os.path.join(image_folder, new_name)
    os.rename(temp_path, new_path)

    ctime    = os.path.getctime(new_path)
    date_str = datetime.datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
    print(f"  {filename}  ({date_str})  →  {new_name}")

print(f"\n✅ Done! {len(temp_files)} images renamed in order of date created.")
print("CAL_001 = your first photo, CAL_002 = second, and so on.")