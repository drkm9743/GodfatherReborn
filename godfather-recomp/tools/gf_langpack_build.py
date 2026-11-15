"""gf_langpack_build.py -- build a language pack from another European disc of the game.

The port always runs on the BLES00043 (EUR, English) data. The French / Italian / German / Spanish discs
(BLES00044 / 45 / 46 / 47) carry the same executable and the same game assets; only their localisation differs.
This copies just that localisation into <langpack dir>/<code>/ (a folder mirroring USRDIR); at run time a file found
there replaces the base disc's file at the same path (ps3recomp runtime/syscalls/sys_fs.c, PS3_LANG_PACK /
langpack/<code> for the launcher's language):
  text/localetable.txt               which language the game loads
  */text/*.str                       the string tables (menus, HUD, subtitles)
  frontend/frontend.str, godfather_v4/godfather_global_global.str, wb_mobface_final/wb_mobface_final_global.str
                                     the menus and the fonts (each disc's fonts only carry its own language's
                                     letters) and the index of the string tables
  <dir>/text/<BLES00043 table>.str   the BLES00043 world files (godfather_v4_global.str, the district *_global.str,
                                     kept from BLES00043 because they hold the world streaming tables) also list the
                                     English tables: those names get the language disc's same-role table, re-written
                                     with the English table's identity (tools/gf_refpack.py recompresses it)
  audiostreams/<XX>/**               the dubbed voices
  movies/*.vp6 that differ           the loading movie and the narrated training videos
  **/dlg/**                          per-language dialogue data
  **/missions/** that differ         mission scripts and cutscenes with their subtitles inside
Map, model and character files are never taken from the language disc.
usage: python gf_langpack_build.py <BLES00043 USRDIR> <language disc USRDIR> <langpack dir>  (writes <langpack dir>/<code>)"""
import os, re, sys, shutil, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gf_str, gf_refpack

LANG_DISCS = {'BLES00044': 'fr', 'BLES00045': 'it', 'BLES00046': 'de', 'BLES00047': 'es'}
TREF = re.compile(rb'(\w+)[\x5c/]text[\x5c/]([0-9A-F]{8})')

def localetable(usr):
    p = os.path.join(usr, 'text', 'localetable.txt')
    if not os.path.isfile(p): sys.exit('no text/localetable.txt in ' + usr)
    return [l.strip() for l in open(p, 'rb').read().decode('latin1').splitlines()]

def table_names(usr):
    """{text folder: [table ids in role order]} from the containers next to each text/ folder."""
    out = {}
    for d in sorted(os.listdir(usr)):
        dd = os.path.join(usr, d)
        if not os.path.isdir(os.path.join(dd, 'text')): continue
        for f in sorted(os.listdir(dd)):
            if not f.lower().endswith('.str'): continue
            for o, u in gf_str.chunks(os.path.join(dd, f)):
                for m in TREF.finditer(u):
                    if m.group(1).decode().lower() != d.lower(): continue
                    l = out.setdefault(d, [])
                    if m.group(2).decode() not in l: l.append(m.group(2).decode())
            if d in out: break                       # the first container that lists them is enough
    return out

def retag_table(src, base_tab):
    """The language disc's string table, re-written with the identity of the BLES00043 table it replaces: the
    resource name, ID, name hashes and source path (the block after the resource header; the game finds tables by these)."""
    d = open(src, 'rb').read(); b = open(base_tab, 'rb').read()
    for x, p in ((d, src), (b, base_tab)):
        if x[:4] != b'SToc' or x[0x800:0x802] != b'\x10\xfb': sys.exit('unexpected string table layout: ' + p)
    u = bytearray(gf_str.refpack(d, 0x800)); v = gf_str.refpack(b, 0x800)
    END = b'\x00\x00\x00\x04\x00\xbf\xbf\xbf'          # closes the name / ID / source-path block
    mu, mv = u.find(END, 0x40), v.find(END, 0x40)
    for x, m, p in ((u, mu, src), (v, mv, base_tab)):
        if x[0x10:0x14] != b'\x00\x00\x00\x14' or x[0x38:0x40] != b'\x00\x00\x00\x04LH2\x00' or not 0x40 < m < 0x100:
            sys.exit('unexpected string table resource: ' + p)
    if mu != mv: sys.exit('string table identity blocks differ in size: %s / %s' % (src, base_tab))
    u[0x10:mu] = v[0x10:mv]
    return gf_refpack.stoc_single(d, bytes(u))      # recompressed: the game decompresses the chunk in place

def same(a, b):
    if os.path.getsize(a) != os.path.getsize(b): return False
    with open(a, 'rb') as fa, open(b, 'rb') as fb:
        while True:
            x, y = fa.read(1 << 20), fb.read(1 << 20)
            if x != y: return False
            if not x: return True

def put(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(dst):
        if os.path.samefile(src, dst) or (os.path.getsize(dst) == os.path.getsize(src) and same(src, dst)): return
        os.remove(dst)
    try: os.link(src, dst)                       # same volume: no extra space
    except OSError: shutil.copy2(src, dst)

def main(base, lang, out):
    bt, lt = localetable(base), localetable(lang)
    if len(bt) < 3 or bt[1] != 'BLES00043': sys.exit('the base game must be BLES00043 (found %s)' % (bt[1] if len(bt) > 1 else '?'))
    if len(lt) < 3 or lt[1] not in LANG_DISCS: sys.exit('the language disc must be one of %s (found %s)' % (', '.join(LANG_DISCS), lt[1] if len(lt) > 1 else '?'))
    code = LANG_DISCS[lt[1]]
    out = os.path.join(os.path.abspath(out), code)
    want = set(); n = size = 0
    def add(src, rel):
        nonlocal n, size
        put(src, os.path.join(out, rel)); want.add(rel); n += 1; size += os.path.getsize(src)
    # string tables under the BLES00043 names and IDs
    bn, ln = table_names(base), table_names(lang)
    for d, names in bn.items():
        other = ln.get(d, [])
        if len(other) != len(names): sys.exit('%s: %d string tables on the base disc, %d on %s' % (d, len(names), len(other), lt[1]))
        for b, l in zip(names, other):
            rel = '%s/text/%s.str' % (d, b); dst = os.path.join(out, rel)
            data = retag_table(os.path.join(lang, d, 'text', l + '.str'), os.path.join(base, d, 'text', b + '.str'))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not (os.path.isfile(dst) and open(dst, 'rb').read() == data):
                if os.path.isfile(dst): os.remove(dst)
                with open(dst, 'wb') as f: f.write(data)
            want.add(rel); n += 1; size += len(data)
    FROM_LANG = {'frontend/frontend.str', 'godfather_v4/godfather_global_global.str', 'wb_mobface_final/wb_mobface_final_global.str'}
    for root, dirs, files in os.walk(lang):
        for f in files:
            src = os.path.join(root, f); rel = os.path.relpath(src, lang).replace(os.sep, '/'); parts = rel.split('/')
            low = rel.lower(); bsrc = os.path.join(base, rel)
            if rel in want: continue
            take = (low == 'text/localetable.txt' or rel in FROM_LANG
                    or (len(parts) >= 2 and parts[-2].lower() == 'text' and low.endswith('.str'))
                    or (parts[0].lower() == 'audiostreams' and len(parts) > 2 and parts[1] == code.upper())
                    or '/dlg/' in '/' + low
                    or ((parts[0].lower() == 'movies' or '/missions/' in '/' + low) and not (os.path.isfile(bsrc) and same(src, bsrc))))
            if take: add(src, rel)
    # files of an earlier pack layout that this one no longer has
    for root, dirs, files in os.walk(out):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), out).replace(os.sep, '/')
            if rel != 'langpack.txt' and rel not in want:
                try: os.remove(os.path.join(root, f))
                except OSError as e: print('could not remove %s: %s' % (rel, e))
    with open(os.path.join(out, 'langpack.txt'), 'w') as f:
        f.write('%s %s %d files\n' % (lt[1], code, n))
    print('language pack %s (%s): %d files, %.1f GB -> %s' % (code, lt[1], n, size / 2**30, out))

if __name__ == '__main__':
    if len(sys.argv) != 4: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
