import argparse
from pathlib import Path
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--yaml_dir", default="boltz2_inputs")
ap.add_argument("--output_dir", default="boltz2_results")
ap.add_argument("--cache", default="../boltz_cache")
a = ap.parse_args()


def run_boltz2_batch(yaml_dir, output_dir="boltz2_results",  devices=1,  diffusion_samples=1,):
    yaml_dir = Path(yaml_dir)
    output_dir = Path(output_dir)

    yaml_files = list(yaml_dir.glob("*.yaml")) + list(yaml_dir.glob("*.yml"))
    if not yaml_files:
        raise ValueError(f"No YAML files found in {yaml_dir}")

    command = ["boltz", "predict", str(yaml_dir), "--out_dir",  str(output_dir),
               "--model",  "boltz2", "--devices", str(devices), "--diffusion_samples",
               str(diffusion_samples), "--output_format", "mmcif",
    ]
    results=subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return results

run_boltz2_batch(a.yaml_dir, a.output_dir, devices=1, diffusion_samples=1)