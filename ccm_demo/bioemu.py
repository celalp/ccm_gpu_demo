import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import mdtraj as md
import numpy as np
from bioemu2 import EmuSession, System, MSA, load_model, check_physicality, save_frames

from utils import get_protein_sequence, load, topology_from_cif, fpocket

ap = argparse.ArgumentParser()
ap.add_argument("--bioemu_output_folder", default="sampled")
ap.add_argument("--protein_fasta", default="vp35.fasta")
ap.add_argument("--msa_path", default="msa.a3m")
ap.add_argument("--model_cache_dir", default="models")
a = ap.parse_args()

# some parameters that we care about for finding pockets
MATCH_DIST = 4.0  # Å: pockets closer than this across frames count as the same site
MAX_RARITY = 0.20  # cryptic = present in < 20 % of frames ...
MIN_FRAMES = 2  # ... but in at least this many
MIN_VOL = 150.0  # Å^3 in its best frame

print("[" + datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "] " + "Getting protein sequence")
sequence = get_protein_sequence(a.protein_fasta)

# create the bioemu2 system
system = System.model_validate({
    "entities": [
        {
            "entity_type": "protein",
            "sequence": sequence,
            "asym_ids": ["A"],
        }
    ],
    "bonds": [],
})

msa = MSA({"A": Path(a.msa_path).read_text()})

print("[" + datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "] " + "Loading model")

emu = EmuSession(
    system,
    model=load_model(cache_dir=Path(a.model_cache_dir), device="cuda"),
    mode="equilibrium",
    msa=msa,
)

print("[" + datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "] " + "Sampling")

samples = emu.sample(num_samples=100)
physicality = check_physicality(samples, system=system)
save_frames(samples, a.bioemu_output_folder, physicality=physicality)

# go through the created samples and pick pockets
print("[" + datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "] " + "Finding pockets")

t = load(Path(a.bioemu_output_folder))
t = t.atom_slice(t.topology.select("protein"))
t.superpose(t[0], atom_indices=t.topology.select("name CA"))
idx = np.linspace(0, t.n_frames - 1, min(200, t.n_frames)).astype(int)
fdir = Path(a.bioemu_output_folder) / "frames";
fdir.mkdir(exist_ok=True)
paths = []

for k, atom in enumerate(t.topology.atoms, 1):
    atom.serial = k
for res in t.topology.residues:
    res.resSeq = int(res.resSeq)

for i in idx:
    p = fdir / f"f{i:05d}.pdb"
    t[int(i)].save_pdb(str(p))
    paths.append(p)

with ThreadPoolExecutor(8) as ex:
    per_frame = list(ex.map(fpocket, paths))

    # group pockets from different frames into sites by centre proximity
    sites = []
    for fi, pockets in enumerate(per_frame):
        for p in pockets:
            for s in sites:
                if np.linalg.norm(s["center"] - p["center"]) < MATCH_DIST:
                    s["hits"].append((fi, p));
                    break
            else:
                sites.append(dict(center=p["center"], hits=[(fi, p)]))

    cryptic = []
    for s in sites:
        n = len({fi for fi, _ in s["hits"]})
        best_fi, best = max(s["hits"], key=lambda h: h[1]["volume"])
        rarity = n / len(paths)
        if n >= MIN_FRAMES and rarity < MAX_RARITY and best["volume"] > MIN_VOL:
            cryptic.append(dict(n_frames=n, rarity=rarity, frame_i=best_fi, **best))
    cryptic.sort(key=lambda c: -c["volume"])

    odir = Path(a.bioemu_output_folder) / "pockets";
    odir.mkdir(exist_ok=True)
    out = []
    for k, c in enumerate(cryptic, 1):
        pdb = odir / f"pocket_{k}.pdb"
        shutil.copy(paths[c["frame_i"]], pdb)  # same coordinate frame as the centres
        resi = [f"{ch}:{r}" for ch, r in c["residues"]]
        (odir / f"pocket_{k}.resi.txt").write_text(" ".join(resi))
        out.append(dict(rank=k, pdb=str(pdb.resolve()), residues=resi,
                        center=[round(float(x), 3) for x in c["center"]],
                        box=[round(max(20.0, float(x) + 8), 1) for x in np.ptp(c["spheres"], axis=0)],
                        spheres=c["spheres"].round(3).tolist(),
                        volume=c["volume"], druggability=c["druggability"],
                        n_frames=c["n_frames"], rarity=round(c["rarity"], 3)))

    funnel = dict(frames=len(paths), pockets=sum(map(len, per_frame)),
                  sites=len(sites), cryptic=len(cryptic))
    (Path(a.bioemu_output_folder) / "pockets.json").write_text(json.dumps(dict(funnel=funnel, pockets=out), indent=2))

print("[" + datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "] " + "Done")
