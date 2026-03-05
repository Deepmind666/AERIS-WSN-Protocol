import os
from PIL import Image
d = 'for_submission/figures_mega'
for f in sorted(os.listdir(d)):
    if f.endswith('.png'):
        im = Image.open(os.path.join(d, f))
        kb = os.path.getsize(os.path.join(d, f)) // 1024
        print(f'{f}: {im.size[0]}x{im.size[1]} px, {kb} KB')
