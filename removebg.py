from rembg import remove
from PIL import Image
import os

input_folder = "pinterest_fin/sakinaangel04/outifts3"
output_folder = "removed_bg_2"
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    input_path = os.path.join(input_folder, file)
    output_path = os.path.join(output_folder, file.replace(".jpg", "_clean.png"))

    with open(input_path, "rb") as i:
        with open(output_path, "wb") as o:
            output = remove(i.read())
            o.write(output)

print("Background removal complete!")
