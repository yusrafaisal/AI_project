import os

folder = "data\images"
extensions = [".jpg", ".jpeg", ".png", ".webp"]

files = [f for f in os.listdir(folder) 
         if os.path.splitext(f)[1].lower() in extensions]

files.sort()  

# Rename
for i, filename in enumerate(files, start=1):
    ext = os.path.splitext(filename)[1]  
    new_name = f"image_{i}{ext}"

    old_path = os.path.join(folder, filename)
    new_path = os.path.join(folder, new_name)

    os.rename(old_path, new_path)

print("Renaming complete!")
