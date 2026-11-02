import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
import yaml

import mdtraj as md
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

def get_protein_sequence(fasta):
    sequence = []
    with open(fasta, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                continue
            else:
                sequence.append(line)

    sequence = "".join(sequence)
    return sequence


def topology_from_cif(path):
    """Minimal _atom_site reader, used only if mdtraj can't read the CIF."""
    lines = Path(path).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("_atom_site."))
    hdr = []
    while lines[i].startswith("_atom_site."):
        hdr.append(lines[i].strip()[len("_atom_site."):])
        i += 1

    col = {h: n for n, h in enumerate(hdr)}
    top, asym0, key0 = md.Topology(), None, None
    for l in lines[i:]:
        if not l.strip() or l[0] in "#_" or l.startswith("loop_"):
            break
        f = shlex.split(l)
        asym, comp = f[col["label_asym_id"]], f[col["label_comp_id"]]
        raw = f[col["label_seq_id"]]
        seq = int(raw) if raw.lstrip("-").isdigit() else 1  # "." for ligands
        if asym != asym0:
            chain, asym0, key0 = top.add_chain(), asym, None
        if (seq, comp) != key0:
            res, key0 = top.add_residue(comp, chain, resSeq=seq), (seq, comp)
        sym = f[col["type_symbol"]].capitalize()
        top.add_atom(f[col["label_atom_id"]], md.element.get_by_symbol(sym), res)
    return top


def load(run):
    xtc, cif = str(run / "samples.xtc"), str(run / "topology.cif")
    try:
        return md.load(xtc, top=cif)
    except Exception as e:
        print(f"mdtraj could not read the CIF ({type(e).__name__}); using fallback parser")
        return md.load_xtc(xtc, top=topology_from_cif(cif))


def fpocket(pdb):
    subprocess.run(["fpocket", "-f", str(pdb)], check=True, capture_output=True)
    out = pdb.with_name(pdb.stem + "_out")
    info = (out / f"{pdb.stem}_info.txt").read_text()
    parts = re.split(r"Pocket\s+(\d+)\s*:", info)
    pockets = []
    for n, blk in zip(parts[1::2], parts[2::2]):
        vol = float(re.search(r"Volume\s*:\s*([\d.]+)", blk).group(1))
        drug = re.search(r"Druggability Score\s*:\s*([-\d.]+)", blk)
        atm = [l for l in (out / "pockets" / f"pocket{n}_atm.pdb").read_text().splitlines()
               if l.startswith(("ATOM", "HETATM"))]
        vert = [l.split()[-5:-2] for l in (out / "pockets" / f"pocket{n}_vert.pqr").read_text().splitlines()
                if l.startswith("ATOM")]
        sph = np.array(vert, dtype=float)  # alpha-sphere centres
        pockets.append(dict(volume=vol, druggability=float(drug.group(1)) if drug else None,
                            center=sph.mean(0), spheres=sph,
                            residues=sorted({(l[21], int(l[22:26])) for l in atm})))
    shutil.rmtree(out)  # removing the pockets we will not really need them
    return pockets


def get_fraction_in_range(residue_list, min_val=90, max_val=100):
    if not isinstance(residue_list, list) or len(residue_list) == 0:
        return 0.0

    count_in_range = 0
    valid_count = 0

    for item in residue_list:
        match = re.search(r':(\d+)', item)
        if match:
            valid_count += 1
            num = int(match.group(1))
            if min_val <= num <= max_val:
                count_in_range += 1

    return count_in_range / valid_count if valid_count > 0 else 0.0


def dock_smiles_with_vina(smiles_list, receptor_pdb, center,
                          size, output_dir="docked",
                          vina_bin="vina", exhaustiveness=1, num_modes=1, num_cores=8):
    """
    Dock a list of SMILES against a receptor using AutoDock Vina.
    smiles_list : list[str]
    receptor_pdb : str
    center : tuple(float,float,float)
        (center_x, center_y, center_z)
    size : tuple(float,float,float)
        (size_x, size_y, size_z)
    output_dir : str
    vina_bin : str
    exhaustiveness : int
    num_modes : int
    num_cores : int
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    lig_dir = output_dir / "ligands"
    dock_dir = output_dir / "docked"

    lig_dir.mkdir(exist_ok=True)
    dock_dir.mkdir(exist_ok=True)

    receptor_pdbqt = output_dir / "receptor.pdbqt"

    subprocess.run(["mk_prepare_receptor.py", "-i", str(receptor_pdb), "-p", str(receptor_pdbqt), ], check=True)
    center_x, center_y, center_z = center
    size_x, size_y, size_z = size

    for idx, mol in tqdm(enumerate(smiles_list)):
        mol = Chem.MolFromSmiles(smiles_list[idx])
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42, )
        AllChem.MMFFOptimizeMolecule(mol)
        sdf_file = lig_dir / f"lig_{idx:03d}.sdf"
        pdbqt_file = lig_dir / f"lig_{idx:03d}.pdbqt"
        writer = Chem.SDWriter(str(sdf_file))
        writer.write(mol)
        writer.close()
        subprocess.run(["mk_prepare_ligand.py", "-i", str(sdf_file), "-o", str(pdbqt_file), ],
                       check=True, )

    ligands = sorted(Path("docked/ligands").glob("*.pdbqt"))

    command = [vina_bin, "--receptor", receptor_pdbqt, "--batch", *map(str, ligands),
               "--center_x", center_x, "--center_y", center_y, "--center_z", center_z,
               "--size_x", size_x, "--size_y", size_y, "--size_z", size_z,
               "--exhaustiveness", exhaustiveness, "--num_modes", num_modes,
               "--dir", output_dir, "--cpu", str(num_cores)]


    Path("docked/docked").mkdir(parents=True, exist_ok=True)

    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,  stderr=subprocess.STDOUT,)
    return result

def parse_vina_results(pdbqt_dir, pattern="*_out.pdbqt"):
    rows = []
    for pdbqt_file in sorted(Path(pdbqt_dir).glob(pattern)):
        pose = None
        with pdbqt_file.open() as f:
            for line in f:
                line = line.strip()
                if line.startswith("MODEL"):
                    pose = int(line.split()[1])
                elif line.startswith("REMARK VINA RESULT:"):
                    values = line.split(":")[1].split()
                    rows.append({"ligand": pdbqt_file.stem.replace("_out", ""),
                        "pose": pose or 1, "vina_score": float(values[0]),
                        "rmsd_lb": float(values[1]), "rmsd_ub": float(values[2]),
                        "pose_file": str(pdbqt_file),})

                elif line.startswith("REMARK INTER + INTRA:"):
                    rows[-1]["inter_plus_intra"] = float(
                        line.split(":")[1]
                    )
                elif line.startswith("REMARK INTER:"):
                    rows[-1]["inter"] = float(line.split(":")[1])
                elif line.startswith("REMARK INTRA:"):
                    rows[-1]["intra"] = float(line.split(":")[1])
                elif line.startswith("REMARK UNBOUND:"):
                    rows[-1]["unbound"] = float(line.split(":")[1])
                elif line.startswith("REMARK SMILES "):
                    if not line.startswith("REMARK SMILES IDX"):
                        rows[-1]["smiles"] = line.removeprefix(
                            "REMARK SMILES "
                        ).strip()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    ligand_index = df["ligand"].str.extract(r"(\d+)", expand=False,)
    df["index"] = pd.to_numeric(ligand_index, errors="coerce")

    columns = ["index","ligand", "smiles",  "pose", "vina_score", "rmsd_lb", "rmsd_ub", "inter_plus_intra",
               "inter", "intra", "unbound", "pose_file",]

    df = df.reindex(columns=columns)
    return df.sort_values(["vina_score", "ligand", "pose"], na_position="last",).reset_index(drop=True)


def create_boltz2_yamls(protein_sequence,msa_file, smiles_list, output_dir="boltz2_inputs",
                        include_affinity=True,):
    """
    protein_sequence : str
        Single-chain protein sequence.
    msa_file : str or Path
        Path to the custom A3M file.
    smiles_list : list[str]
        Ligand SMILES strings.
    output_dir : str or Path
        Directory for generated YAML files.
    include_affinity : bool
        Include Boltz-2 affinity prediction.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    msa_file = Path(msa_file).resolve()

    if not msa_file.is_file():
        raise FileNotFoundError(f"MSA file not found: {msa_file}")

    protein_sequence = "".join(protein_sequence.split()).upper()

    valid_residues = set("ACDEFGHIKLMNPQRSTVWYX")
    invalid = sorted(set(protein_sequence) - valid_residues)

    if invalid:
        raise ValueError(f"Invalid protein-sequence characters: {invalid}" )

    if not smiles_list:
        raise ValueError("smiles_list is empty")

    yaml_files = []

    for index, smiles in enumerate(smiles_list):
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError(f"Invalid SMILES at index {index}: {smiles!r}")

        document = {"version": 1,
                "sequences": [{
                    "protein": {
                    "id": "A",
                    "sequence": protein_sequence,
                    "msa": str(msa_file),
                }},
                {"ligand": {
                    "id": "B",
                    "smiles": smiles.strip()
                }},
            ],}

        if include_affinity:
            document["properties"] = [
                {"affinity": {"binder": "B",}}]

        yaml_file = output_dir / f"ligand_{index:03d}.yaml"

        with yaml_file.open("w") as handle:
            yaml.safe_dump(document, handle, sort_keys=False,  default_flow_style=False,)
        yaml_files.append(yaml_file)
    return yaml_files

def load_boltz_affinities(results_dir):
    results_dir = Path(results_dir)

    records = []

    for json_path in results_dir.rglob("affinity_*.json"):
        with json_path.open() as f:
            data = json.load(f)

        # Typical filename: affinity_<input_name>.json
        input_name = json_path.stem.removeprefix("affinity_")

        record = {
            "input_name": input_name,
            "affinity_pred_value": data.get("affinity_pred_value"),
            "affinity_probability_binary": data.get(
                "affinity_probability_binary"
            ),
            "json_path": str(json_path),
        }

        # Preserve any additional scalar fields
        for key, value in data.items():
            if key not in record and not isinstance(value, (dict, list)):
                record[key] = value

        records.append(record)

    if not records:
        return pd.DataFrame(
            columns=[
                "input_name",
                "affinity_pred_value",
                "affinity_probability_binary",
                "json_path",
            ]
        )

    df = pd.DataFrame(records)

    return df.sort_values(
        "affinity_pred_value",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)