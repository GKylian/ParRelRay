from PIL import Image
import os.path
import sys

ppm = open('render.ppm','r')
ppm.readline()
words = ppm.readline().split()
ppm.readline()

width = int(words[0])
height = int(words[1])

print("Converting image of size:",width,",",height)

img = Image.new('RGB', (width, height), color = (75,110,140))
pixels = img.load()

for l in range(height):
    line = ppm.readline()
    pxl = line.split()
    for w in range(width):
        r = int(pxl[w*3 + 0])
        g = int(pxl[w*3 + 1])
        b = int(pxl[w*3 + 2])
        pixels[w,l] = (r, g, b)
        

img.save('render.png')
img.close()
ppm.close()
