"""gf_uifont_build.py -- build assets/ui_font.bin (the overlay's font: inventory, stores, PC settings / co-op menu,
notices) from the player's disc. First choice: the game's own fonts (gf_uifont_game.py: Trade Gothic + Magnesium,
every Windows-1252 letter). Fallback (UI_FONT_TTF=1 forces it): installed fonts (gf_uifont.py: bold sans;
gf_uifont_serif.py: serif titles). The controller icons come from the game's button prompt atlas either way.
usage: python gf_uifont_build.py <USRDIR> <out ui_font.bin>"""
import os, sys, struct, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gf_kbm_build as K

def main(usr, out):
    out = os.path.abspath(out); os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.environ.get('UI_FONT_TTF', '') in ('', '0'):
        try:
            import gf_uifont_game; gf_uifont_game.main(usr, out); return
        except (Exception, SystemExit) as e:
            print('game fonts not usable (%s) -- using installed fonts' % e)
    w, h, raw = K.find_atlas(usr)
    work = tempfile.mkdtemp(prefix='gf_uifont_')
    tex = os.path.join(work, 'buttons.tex'); tmp = os.path.join(work, 'ui_font.bin')
    with open(tex, 'wb') as f: f.write(b'GFTEX1\0\0' + struct.pack('<5I', w, h, 0x87, len(raw), K.CSUM) + raw)
    subprocess.check_call([sys.executable, os.path.join(HERE, 'gf_uifont.py'), tex, tmp], cwd=work)          # writes a preview in cwd
    subprocess.check_call([sys.executable, os.path.join(HERE, 'gf_uifont_serif.py'), tmp, out], cwd=work)
    print('overlay font ->', out)

if __name__ == '__main__':
    if len(sys.argv) != 3: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
