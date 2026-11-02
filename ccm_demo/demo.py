# some of the commands that retun things to the interactive shell have been commented out

import json
import re

import pandas as pd
from tqdm import tqdm

from usearchmolecules.dataset import FingerprintedDataset, shape_mixed

from pathlib import Path
import subprocess
import tempfile



from benchmate.apis import UniProt
from benchmate.sequence import Sequence
from benchmate.alignment import MMSeqs

from utils import get_fraction_in_range, dock_smiles_with_vina, parse_vina_results, create_boltz2_yamls, load_boltz_affinities

# search uniprot for a proteins
uniprot=UniProt()
results=pd.DataFrame(uniprot.search("ebola vp35"))
#results

protein=uniprot.get_info("Q05127")
#protein.results.description
#len(protein.results["sequence"])

#protein.results["xref_types"]
#[item for item in protein.results["xrefs"] if item["type"]=="PDB"]

seq=Sequence(name="vpa35", sequence=protein.results.sequence, seq_type="protein")
seq.to_fasta("vpa35.fasta")
mutations=pd.DataFrame(protein.results["mutagenesis"])
#mutations

mmseqs=MMSeqs()


# get the msa we will use it with bioemy AND boltz2
search_results=mmseqs.search(seq, "swissprot/swissprot", "vp35.tsv", "vp35.a3m")

#run bioemu that's in a different conda env

bioemu_results=subprocess.run(["conda", "run", "-n", "bioemu", "python", "bioemu.py"], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

#this generates a json file with the pockets found in the bioemu generated samples
with open("sampled/pockets.json", "r") as file:
    data = json.load(file)

pk=data["pockets"]

df = pd.DataFrame([{
    "pocket": p["rank"],                                     # K in pocket_K.pdb / pocket_K.resi.txt
    "resi_file": f"pockets/pocket_{p['rank']}.resi.txt",
    "druggability": p["druggability"],
    "frac_frames": p["rarity"],                              # fraction of analyzed frames with this site
    "n_frames": p["n_frames"],
    "volume": p["volume"],
    "residues":p["residues"],
    "n_res": len(p["residues"]),
    "center": tuple(p["center"]),
    "radii":tuple(p["box"])
} for p in pk])
df["druggability"] = pd.to_numeric(df["druggability"], errors="coerce")

df=df[df["druggability"]>0.8]
df['p1_fraction'] = df['residues'].apply(lambda x: get_fraction_in_range(x, 90, 107))
df['p2_fraction'] = df['residues'].apply(lambda x: get_fraction_in_range(x, 305, 322))

#pick a pocket
df.sort_values(by="p1_fraction", ascending=False)
df.sort_values(by="p2_fraction", ascending=False)

#pocket=<this is the pocket number that we pick>
pocket=0

diffsbdd_results=subprocess.run(["conda", "run", "-n", "diffsbdd", "python", "diffsbdd.py"], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

with open(f"pocket_{str(pocket)}.smi", "r") as f:
    smiles=[]
    for line in f:
        line=line.strip()
        smiles.append(line)

#search for similar molecules in pubchem
data = FingerprintedDataset.open("USearchMolecules/data/pubchem", shape=shape_mixed)
#len(data) how many molecules are in the data

results=[]
for chem in smiles:
    r= data.search(chem, 20)
    s=[item[1] for item in r]
    results.extend(s)

center=df["center"][df["pocket"]==pocket].iloc[0]
size=df["radii"][df["pocket"]==pocket].iloc[0]
receptor_pdb=f"sampled/pockets/pocket_{pocket}.pdb"

docked_results=dock_smiles_with_vina(smiles_list=smiles, receptor_pdb=receptor_pdb, center=center, size=size)

vina_results=parse_vina_results("docked/docked", pattern="*_out.pdbqt")
#vina_results

top_5_smiles=vina_results["smiles"].iloc[0:5].tolist()

create_boltz2_yamls(protein.results["sequence"], "vp35.a3m", smiles_list=top_5_smiles, output_dir="boltz2_inputs", include_affinity=True)

boltz_results=subprocess.run(["conda", "run", "-n", "boltz2", "python", "ccm_demo/boltz2.py"], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

load_boltz_affinities(results_dir="boltz2_results/boltz_results_boltz2_inputs/predictions/")



