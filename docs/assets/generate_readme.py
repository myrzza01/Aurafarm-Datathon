"""Rebuild the original pixel-art README banner (Pillow required)."""
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FONT = Path('C:/Windows/Fonts/consolab.ttf')

def font(size):
    return ImageFont.truetype(str(FONT), size) if FONT.exists() else ImageFont.load_default(size=size)

def frame(t):
    im = Image.new('RGB', (1200, 420), '#200e1b')
    d = ImageDraw.Draw(im)
    for x in range(0, 1200, 40):
        for y in range(0, 420, 40):
            d.rectangle((x, y, x+38, y+38), fill=('#291222' if (x+y)//40 % 3 else '#321529'))
    d.rectangle((26, 26, 1173, 393), outline='#a31551', width=3)
    d.rectangle((52, 55, 59, 277), fill='#fc9385')
    d.text((82, 55), 'FORTE BANK  /  DATATHON', font=font(22), fill='#fc9385')
    d.text((78, 102), 'FORTE VISION', font=font(68), fill='#fff5f8')
    d.text((83, 190), 'MINE THE FRAMES. FIND THE SIGNAL.', font=font(22), fill='#e3bbce')
    for x, label in [(83, 'VIDEO'), (252, 'LOCAL AI'), (454, 'EVIDENCE')]:
        w = len(label)*13+36
        d.rectangle((x, 245, x+w, 283), fill='#a31551')
        d.text((x+18, 253), label, font=font(18), fill='white')
    # Original block-built bee: honey body, plum stripes, flapping wings.
    bx, by = 930, 158 + round(math.sin(t*math.tau)*9)
    wing = 14 if int(t*12)%2 else 0
    d.rectangle((bx-26, by-49-wing, bx+5, by-13), fill='#f7e7ef')
    d.rectangle((bx+20, by-42+wing, bx+56, by-8), fill='#cdaabd')
    d.rectangle((bx-64, by, bx+80, by+80), fill='#fc9385')
    for dx in [-32, 16]:
        d.rectangle((bx+dx, by, bx+dx+16, by+80), fill='#70243e')
    d.rectangle((bx+80, by+16, bx+112, by+64), fill='#ffd18d')
    d.rectangle((bx+94, by+25, bx+104, by+37), fill='#200e1b')
    d.rectangle((bx-80, by+32, bx-64, by+48), fill='#fff5f8')
    d.rectangle((bx+80, by-16, bx+88, by+1), fill='#fc9385')
    for x in range(0, 1200, 40):
        y = 337 + (x//40 % 3)*8
        d.rectangle((x, y, x+39, 420), fill='#4a0221')
        d.rectangle((x, y, x+39, y+8), fill='#a31551')
        d.rectangle((x+8, y+22, x+18, y+30), fill='#70243e')
    d.text((83, 355), 'BUILD TRUST, BLOCK BY BLOCK.', font=font(19), fill='#ffcfca')
    return im

frames = [frame(i/24) for i in range(24)]
frames[0].save(ROOT/'forte-vision.png', optimize=True)
frames[0].save(ROOT/'forte-vision.gif', save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
print('Generated 1200 x 420 banner; identical GIF frames may be merged.')
