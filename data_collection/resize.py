import PIL
import os
from PIL import Image

f = r'data/'

for file in os.listdir(f):
    if file.startswith('.'): continue
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((256,144))
    img.save(f_img)