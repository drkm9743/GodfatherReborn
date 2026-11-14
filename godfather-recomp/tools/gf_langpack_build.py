"""gf_langpack_build.py -- build a language pack from another European disc of the game.

The port always runs on the BLES00043 (EUR, English) data. The French / Italian / German / Spanish discs
(BLES00044 / 45 / 46 / 47) carry the same executable and the same game assets; only their localisation differs.
This copies just that localisation into <langpack dir>/<code>/ (a folder mirroring USRDIR); at run time a file found
there replaces the base disc's file at the same path (ps3recomp runtime/syscalls/sys_fs.c, PS3_LANG_PACK /
langpack/<code> for the launcher's language):
  text/localetable.txt               which language the game loads
  */text/*.str                       the string tables (menus, HUD, subtitles)
  frontend/frontend.str              the menus, and the index of the frontend string tables
  godfather_v4/godfather_global_global.str, wb_mobface_final/wb_mobface_final_global.str
                                     the index of the in-game string tables (each table is found by an ID stored
                                     inside it, so the index must come from the same disc as the tables); apart from
                                     that index these two match BLES00043 except for build paths
  audiostreams/<XX>/**               the dubbed voices
  movies/*.vp6 that differ           the loading movie and the narrated training videos
  **/dlg/**                          per-language dialogue data
  **/missions/** that differ         mission scripts and cutscenes with their subtitles inside
Map, model and character files are never taken from the language disc.
usage: python gf_langpack_build.py <BLES00043 USRDIR> <language disc USRDIR> <langpack dir>  (writes <langpack dir>/<code>)"""
import os, sys, shutil

LANG_DISCS = {'BLES00044': 'fr', 'BLES00045': 'it', 'BLES00046': 'de', 'BLES00047': 'es'}
INDEXES = {'frontend/frontend.str', 'godfather_v4/godfather_global_global.str', 'wb_mobface_final/wb_mobface_final_global.str'}

def localetable(usr):
    p = os.path.join(usr, 'text', 'localetable.txt')
    if not os.path.isfile(p): sys.exit('no text/localetable.txt in ' + usr)
    return [l.strip() for l in open(p, 'rb').read().decode('latin1').splitlines()]

def same(a, b):
    if os.path.getsize(a) != os.path.getsize(b): return False
    with open(a, 'rb') as fa, open(b, 'rb') as fb:
        while True:
            x, y = fa.read(1 << 20), fb.read(1 << 20)
            if x != y: return False
            if not x: return True

def put(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try: os.link(src, dst)                       # same volume: no extra space
    except OSError: shutil.copy2(src, dst)

def main(base, lang, out):
    bt, lt = localetable(base), localetable(lang)
    if len(bt) < 3 or bt[1] != 'BLES00043': sys.exit('the base game must be BLES00043 (found %s)' % (bt[1] if len(bt) > 1 else '?'))
    if len(lt) < 3 or lt[1] not in LANG_DISCS: sys.exit('the language disc must be one of %s (found %s)' % (', '.join(LANG_DISCS), lt[1] if len(lt) > 1 else '?'))
    code = LANG_DISCS[lt[1]]
    if lt[2] != 'pal_' + code: print('note: %s says %s' % (lt[1], lt[2]))
    out = os.path.join(os.path.abspath(out), code)
    if os.path.isdir(out): shutil.rmtree(out)
    n = size = 0
    for root, dirs, files in os.walk(lang):
        for f in files:
            src = os.path.join(root, f); rel = os.path.relpath(src, lang).replace(os.sep, '/'); parts = rel.split('/')
            low = rel.lower(); bsrc = os.path.join(base, rel)
            take = (low == 'text/localetable.txt' or rel in INDEXES
                    or (len(parts) >= 2 and parts[-2].lower() == 'text' and low.endswith('.str'))
                    or (parts[0].lower() == 'audiostreams' and len(parts) > 2 and parts[1] == code.upper())
                    or '/dlg/' in '/' + low
                    or ((parts[0].lower() == 'movies' or '/missions/' in '/' + low) and not (os.path.isfile(bsrc) and same(src, bsrc))))
            if not take: continue
            put(src, os.path.join(out, rel)); n += 1; size += os.path.getsize(src)
    with open(os.path.join(out, 'langpack.txt'), 'w') as f:
        f.write('%s %s %d files\n' % (lt[1], code, n))
    print('language pack %s (%s): %d files, %.1f GB -> %s' % (code, lt[1], n, size / 2**30, out))

if __name__ == '__main__':
    if len(sys.argv) != 4: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
