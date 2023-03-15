# Converts the .ppm images into .png images

from PIL import Image
import os

for fname in os.listdir("renders/"):
    if ".ppm" in fname:
        print(f"Transforming {fname} to a .png file")
        im = Image.open("renders/"+fname)
        im.save("renders/"+fname[:-4]+".png")
        im.close()
        os.remove("renders/"+fname)