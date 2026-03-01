import os

# ─────────────────────────────────────────
# AUTO RENAME IMAGES TO CAL_001.jpg FORMAT
# ─────────────────────────────────────────

image_folder = 'images'
valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

# Get all image files in the folder
image_files = [
    f for f in os.listdir(image_folder)
    if f.endswith(valid_extensions)
]

# Sort them so renaming is in order
image_files.sort()

print(f"Found {len(image_files)} images. Renaming...")

for index, filename in enumerate(image_files, start=1):
    old_path = os.path.join(image_folder, filename)
    new_name = f"CAL_{index:03d}.jpg"
    new_path = os.path.join(image_folder, new_name)
    os.rename(old_path, new_path)
    print(f"  {filename}  →  {new_name}")

print(f"\n✅ Done! {len(image_files)} images renamed.")
