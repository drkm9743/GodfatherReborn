"""GPS route line: build the city road network from the game's navigation graphs.
Reads every main road graph (GRAPH resources with flags 0x4) in USRDIR/godfather_v4, joins them through their
neighbour links and where nodes of two graphs coincide, and writes assets/gf_gps.bin (little-endian), read by gf_gps.c:
  header (struct GpsBlob in gf_gps.c) + node x[], z[] (s16, 0.25 m) + incoming-edge CSR (u16 start[n+1], u16 src[e]).
usage: python gf_gps_build.py <USRDIR> <out.bin>"""
import collections, glob, math, os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gf_str import refpack

RP, RAW, DATA_START = 0xB9F0B9EC, 0x0EAC15C8, 0x800
UNIT = 0.25          # metres per stored coordinate unit
STITCH = 4.0         # nodes of different graphs closer than this (m) are joined
MAX_NBR_LINK = 30.0  # neighbour links longer than this (m) are dropped


def stoc_chunks(data):
    """The chunks of an SToc file (RefPack or raw), decompressed."""
    if data[:4] != b"SToc":
        return
    count, table = data[8], struct.unpack(">I", data[0x10:0x14])[0]
    pos = DATA_START
    for i in range(count):
        _, comp, size, _, stored, _ = struct.unpack(">6I", data[table + 24 * i:table + 24 * i + 24])
        try:
            if comp == RP:
                yield refpack(data, pos)
            elif comp == RAW:
                yield data[pos:pos + size]
        except Exception:
            pass
        pos += stored


def graph_records(path):
    """GRAPH resources (type 0x716) in one .str file."""
    for c in stoc_chunks(open(path, "rb").read()):
        q = 0
        while q + 12 <= len(c):
            t, s, _ = struct.unpack("<III", c[q:q + 12])
            if s > len(c) - q - 12:
                break
            if t == 0x716 and b"GRAPH" in c[q + 12:q + 80]:
                yield c[q + 12:q + 12 + s]
            q += 12 + s


def parse_graph(rec):
    """EARS::Framework::Navigation::Graph (big-endian; offsets relative to the object, 4 bytes after the header)."""
    g = rec[struct.unpack(">I", rec[:4])[0] + 4:]
    U = lambda o, f=">I": struct.unpack(f, g[o:o + struct.calcsize(f)])[0]
    G = dict(nNode=U(12, ">H"), nConn=U(14, ">H"), aNode=U(32), aConn=U(36), nNeighConn=U(46, ">H"),
             nNeighbor=U(60, ">H"), flags=U(78, ">H"), guid=g[16:32].hex())
    G["nodes"] = [struct.unpack(">3f", g[G["aNode"] + 36 * i:G["aNode"] + 36 * i + 12]) for i in range(G["nNode"])]
    G["conns"] = [struct.unpack(">HH", g[G["aConn"] + 12 * i + 4:G["aConn"] + 12 * i + 8]) for i in range(G["nConn"])]
    aG, nG = U(40), U(44, ">H")
    G["guids"] = [g[aG + 16 * i:aG + 16 * i + 16].hex() for i in range(nG)]
    aNC, aNG = U(48), U(52)
    G["nbr_guid_idx"] = [U(aNG + 2 * i, ">H") for i in range(G["nNeighbor"])]
    G["nconns"] = [struct.unpack(">fHHHHHH", g[aNC + 16 * i:aNC + 16 * i + 16]) for i in range(G["nNeighConn"])]
    return G


def load_roads(usrdir):
    root = os.path.join(usrdir, "godfather_v4")
    files = sorted(p for p in glob.glob(os.path.join(root, "**", "*.str"), recursive=True)
                   if not any(k in p.replace("\\", "/").lower() for k in ("/dlg/", "igc_", "audio", "_shot", "mission")))
    gs = []
    for p in files:
        try:
            for rec in graph_records(p):
                G = parse_graph(rec)
                if G["flags"] == 0x4:            # the main road graph of a district
                    gs.append(G)
        except Exception:
            continue
    return gs


def build_city(gs):
    nodes, edges, base = [], set(), []
    for k, G in enumerate(gs):
        base.append(len(nodes))
        nodes += [(x, z, k) for x, _, z in G["nodes"]]
        edges |= {(base[k] + src, base[k] + dst) for dst, src in G["conns"]}   # (from, to)
    byguid = {G["guid"]: k for k, G in enumerate(gs)}
    for k, G in enumerate(gs):
        for _, dst, src, _, _, inb, _ in G["nconns"]:
            gi = G["nbr_guid_idx"][inb] if inb < len(G["nbr_guid_idx"]) else None
            k2 = byguid.get(G["guids"][gi]) if gi is not None and gi < len(G["guids"]) else None
            if k2 is not None and dst < gs[k2]["nNode"]:
                a, b = base[k] + src, base[k2] + dst
                if math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1]) < MAX_NBR_LINK:
                    edges.add((a, b))
    grid = collections.defaultdict(list)
    for i, (x, z, _) in enumerate(nodes):
        grid[(int(x // 8), int(z // 8))].append(i)
    for i, (x, z, k) in enumerate(nodes):
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for j in grid[(int(x // 8) + dx, int(z // 8) + dz)]:
                    if j > i and nodes[j][2] != k and math.hypot(nodes[j][0] - x, nodes[j][1] - z) < STITCH:
                        edges.add((i, j)); edges.add((j, i))
    return nodes, edges


def write_blob(nodes, edges, out):
    n = len(nodes)
    X = [int(round(x / UNIT)) for x, _, _ in nodes]
    Z = [int(round(z / UNIT)) for _, z, _ in nodes]
    assert all(-32768 <= v < 32768 for v in X + Z) and n < 65535
    inc = [[] for _ in range(n)]
    for a, b in edges:
        if a != b:
            inc[b].append(a)
    rstart, rsrc = [0], []
    for v in range(n):
        rsrc += sorted(set(inc[v])); rstart.append(len(rsrc))
    e = len(rsrc); assert e < 65536
    hdr = 16
    off_x = hdr; off_z = off_x + 2 * n; off_rs = off_z + 2 * n; off_rsrc = off_rs + 2 * (n + 1)
    blob = struct.pack("<4sIII", b"GPS2", n, e, 0)
    blob += struct.pack("<%dh" % n, *X) + struct.pack("<%dh" % n, *Z)
    blob += struct.pack("<%dH" % (n + 1), *rstart) + struct.pack("<%dH" % e, *rsrc)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "wb").write(blob)
    print("gps: %d road graphs -> %d nodes, %d edges, %d bytes -> %s" % (len(set(k for _, _, k in nodes)), n, e, len(blob), out))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    gs = load_roads(sys.argv[1])
    if not gs:
        sys.exit("gps: no road graphs found under %s" % sys.argv[1])
    write_blob(*build_city(gs), sys.argv[2])
