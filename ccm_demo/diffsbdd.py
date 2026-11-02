from datetime import datetime
import argparse
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--pdb_folder", default="sampled/pockets/")
ap.add_argument("--pocket", default=76)
ap.add_argument("--num_molecules", default=5)
ap.add_argument("--model_checkpoint", default="DiffSBDD/models/moad_fullatom_cond.ckpt")
a = ap.parse_args()

print("[" + datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "] " + "Getting residues")

with open(f"{a.pdb_folder}/pocket_{str(a.pocket)}.resi.txt", "r") as f:
    lines=[]
    for line in f:
        line=line.strip()
        lines.append(line)
    lines="".join(lines)

pdb=f"{a.pdb_folder}/pocket_{str(a.pocket)}.pdb"

print("[" + datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "] " + "Generating ligands")

command=["python", "DiffSBDD/generate_ligands.py", a.model_checkpoint, "--pdbfile", pdb,
        "--outfile", f"pocket_{str(a.pocket)}.sdf", "--nsamples", str(a.num_molecules),
        "--resi_list", lines]

#run diffsbdd
subprocess.run(command, check=False, text=True)

#convert the sdf file to smiles
subprocess.run(["obabel", f"pocket_{str(a.pocket)}.sdf", "-O", f"pocket_{str(a.pocket)}.smi"])

print("[" + datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "] " + f"Done the results are in pocket_{str(a.pocket)}.smi")