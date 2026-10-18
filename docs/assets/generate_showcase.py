"""Generate local README panels, badges and a 12-second illustrative GIF.

Requires Pillow. Uses system fonts with Cyrillic support where available.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
BG, CARD, ACCENT, CORAL, WHITE = '#200e1b', '#381427', '#a31551', '#fc9385', '#fff5f8'

def font(size):
    for path in ('C:/Windows/Fonts/consola.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)

def canvas(size):
    im = Image.new('RGB', size, BG)
    d = ImageDraw.Draw(im)
    for x in range(0, size[0], 32):
        for y in range(0, size[1], 32):
            d.rectangle((x, y, x+30, y+30), fill='#291222')
    return im, d

im, d = canvas((1200, 260))
for i, (title, line1, line2) in enumerate([
    ('АНАЛИЗ', 'Локальная ML-модель', 'CPU / CUDA'),
    ('ТАЙМКОДЫ', 'От события к кадру', 'Проверка по видео'),
    ('ИСТОЧНИКИ ПРАВИЛ', 'Основание + контекст', 'Условия применимости'),
]):
    x = 20 + i*395
    d.rectangle((x+6, 26, x+386, 240), fill='#100811')
    d.rectangle((x, 20, x+380, 234), fill=CARD, outline=ACCENT, width=3)
    for j in range(3):
        d.rectangle((x+22+j*18, 43, x+33+j*18, 54), fill=CORAL)
    d.text((x+22, 78), title, font=font(29), fill=WHITE)
    d.text((x+22, 134), line1, font=font(23), fill='#e3bbce')
    d.text((x+22, 173), line2, font=font(20), fill=CORAL)
im.save(ROOT/'feature-panels.png', optimize=True)

for label in ('Python', 'FastAPI', 'PyTorch', 'Docker'):
    width = len(label)*9+40
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="30" role="img" aria-label="{label}">
<rect width="{width}" height="30" fill="{BG}"/><path d="M0 0H{width}V30H0Z" fill="none" stroke="{ACCENT}" stroke-width="3"/>
<rect x="9" y="11" width="8" height="8" fill="{CORAL}"/>
<text x="25" y="20" fill="{WHITE}" font-family="monospace" font-size="14">{label}</text></svg>'''
    (ROOT/f'badge-{label.lower()}.svg').write_text(svg, encoding='utf-8')

frames = []
for n in range(60):
    stage, step = divmod(n, 20)
    im, d = canvas((1200, 560))
    d.rectangle((20, 20, 1180, 540), outline=ACCENT, width=3)
    d.text((48, 38), 'FORTE VISION / КАК ПРОХОДИТ ПРОВЕРКА', font=font(29), fill=WHITE)
    d.text((48, 82), 'ИЛЛЮСТРАЦИЯ ПРОЦЕССА · НЕ ЗАПИСЬ ПРИЛОЖЕНИЯ', font=font(19), fill=CORAL)
    for i, title in enumerate(('01 ЗАГРУЗКА', '02 АНАЛИЗ', '03 СОБЫТИЯ')):
        x = 48+i*375
        d.rectangle((x, 130, x+350, 179), fill=ACCENT if i==stage else CARD)
        d.text((x+18, 143), title, font=font(23), fill=WHITE if i<=stage else '#987085')
    d.rectangle((48, 206, 1152, 462), fill=CARD)
    if stage==0:
        d.text((80, 233), 'video.mov', font=font(34), fill=WHITE)
        d.text((80, 289), 'Отправка одного видео на проверку', font=font(24), fill='#e3bbce')
        d.rectangle((80, 350, 1080, 389), fill=BG)
        progress = min(1, (step+1)/18)
        for k in range(int(progress*40)):
            d.rectangle((84+k*25, 354, 104+k*25, 385), fill=CORAL)
        d.text((80, 409), f'ЗАГРУЗКА {int(progress*100):3d}%', font=font(22), fill=WHITE)
    elif stage==1:
        d.text((80, 233), 'Локальная обработка', font=font(34), fill=WHITE)
        d.text((80, 289), 'Кадры → признаки → предполагаемые события', font=font(23), fill='#e3bbce')
        for k in range(11):
            x = 80+k*92
            d.rectangle((x, 348, x+72, 399), fill=BG, outline=ACCENT, width=2)
            if k==step%11:
                d.rectangle((x+6, 354, x+66, 393), fill=CORAL)
        d.text((80, 420), 'processing' + '.'*(step%4), font=font(22), fill=CORAL)
    else:
        d.text((80, 231), 'Пример события · 00:12', font=font(32), fill=WHITE)
        d.text((80, 286), 'Таймкод + кадры + контекст правила', font=font(24), fill='#e3bbce')
        for k, title in enumerate(('КАДР 1', 'КАДР 2', 'ИСТОЧНИК')):
            x = 80+k*330
            d.rectangle((x, 339, x+300, 397), fill=BG, outline=CORAL if step//7==k else ACCENT, width=2)
            d.text((x+24, 355), title, font=font(22), fill=WHITE)
        d.text((80, 421), 'Вымышленный пример · требуется проверка человеком', font=font(21), fill=CORAL)
    d.text((48, 494), 'ЗАГРУЗКА → АНАЛИЗ → СОБЫТИЯ     /     AURAFARM', font=font(21), fill='#e3bbce')
    frames.append(im)
frames[0].save(ROOT/'workflow-demo.gif', save_all=True, append_images=frames[1:], duration=200, loop=0, optimize=True)
frames[-1].save(ROOT/'workflow-preview.png', optimize=True)
print('Generated panels, four local badges and a 12-second workflow illustration.')
