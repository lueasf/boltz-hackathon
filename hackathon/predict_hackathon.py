# predict_hackathon.py
import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any, List, Optional

import torch
torch.backends.cuda.matmul.fp32_precision = 'high'

import yaml
from hackathon_api import Datapoint, Protein, SmallMolecule
import random, math, glob, json

# ---------------------------------------------------------------------------
# ---- Participants should modify these four functions ----------------------
# ---------------------------------------------------------------------------

def _iter_atoms_from_pdb(pdb_path):
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    chain = line[21].strip()
                    b = float(line[60:66].strip()) if line[60:66].strip() else None
                    yield (chain, x, y, z, b)
                except Exception:
                    continue


def _d(a, b): return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2+(a[2]-b[2])**2)


def _interface_score(pdb_path, chains_ab=("H", "L"), chain_ag="A", cutoff=5.0):
    ab, ag, bvals = [], [], []
    for chain, x, y, z, b in _iter_atoms_from_pdb(pdb_path):
        if chain in chains_ab: ab.append((x,y,z,b))
        elif chain == chain_ag: ag.append((x,y,z,b))
    if not ab or not ag: return -1e9
    contacts=0
    for x1,y1,z1,b1 in ab:
        for x2,y2,z2,b2 in ag:
            if _d((x1,y1,z1),(x2,y2,z2))<=cutoff:
                contacts+=1
                if b1 is not None: bvals.append(b1)
                if b2 is not None: bvals.append(b2)
    mean_b = sum(bvals)/len(bvals) if bvals else 0.0
    return contacts + 0.5*mean_b


def _subsample_msa(msa_file: Path, out_file: Path, max_lines: int, rng: random.Random):
    lines = msa_file.read_text().splitlines()
    if any(l.startswith(">") for l in lines):  # FASTA-like
        records, cur = [], []
        for l in lines:
            if l.startswith(">"):
                if cur: records.append(cur); cur=[l]
                else: cur=[l]
            else:
                cur.append(l)
        if cur: records.append(cur)
        rng.shuffle(records)
        records = records[:max_lines]
        flat = []
        for r in records: flat.extend(r)
        out_file.write_text("\n".join(flat) + "\n")
    else:
        rng.shuffle(lines)
        out_file.write_text("\n".join(lines[:max_lines]) + "\n")



def prepare_protein_complex(datapoint_id: str, proteins, input_dict: dict, msa_dir: Optional[Path] = None):
    rng = random.Random(hash(datapoint_id) & 0xFFFFFFFF)
    configs = []

    config_params = [
      {"subsample_msa": False, "num_msa": None, "step_scale": 0.6, "diffusion_samples": 8, "recycling_steps": 8},
    ]

    for params in config_params:
        cli = [
            "--devices", "1",
            "--output_format", "pdb",
            "--diffusion_samples", str(params["diffusion_samples"]),
            "--recycling_steps", str(params["recycling_steps"]),
            "--seed", str(rng.randint(1, 10_000_000)),
            "--override",  # <<< force la re-génération même si un run existe
        ]
        if params["step_scale"] is not None:
            cli += ["--step_scale", str(params["step_scale"])]
        if params["subsample_msa"]:
            cli += ["--subsample_msa", "--num_subsampled_msa", str(params["num_msa"])]
        configs.append((input_dict, cli))
    return configs

def post_process_protein_complex(datapoint, input_dicts: list[dict[str, Any]], cli_args_list: list[list[str]], prediction_dirs: list[Path]) -> list[Path]:
    def interface_contacts_and_b(pdb_path: Path, cutoff: float = 5.0):
        ab, ag, bvals = [], [], []
        for chain, x, y, z, b in _iter_atoms_from_pdb(pdb_path):
            if chain in ("H","L"): ab.append((x,y,z,b))
            elif chain == "A": ag.append((x,y,z,b))
        if not ab or not ag: 
            return 0, 0.0
        contacts = 0
        for x1,y1,z1,b1 in ab:
            for x2,y2,z2,b2 in ag:
                if _d((x1,y1,z1),(x2,y2,z2)) <= cutoff:
                    contacts += 1
                    if b1 is not None: bvals.append(b1)
                    if b2 is not None: bvals.append(b2)
        mean_b = (sum(bvals)/len(bvals)) if bvals else 0.0
        return contacts, mean_b

    candidates = []
    for pred_dir in prediction_dirs:
        # Essayer de charger une métrique de confiance si disponible (très permissif)
        conf_map = {}
        for js in pred_dir.glob("*.json"):
            try:
                data = json.loads(js.read_text())
            except Exception:
                continue
            # Cherche des clés plausibles
            if isinstance(data, dict):
                # Exemple: {"models": [{"name":"model_0","confidence":87.1,...}, ...]}
                if "models" in data and isinstance(data["models"], list):
                    for m in data["models"]:
                        name = str(m.get("name",""))
                        c = m.get("confidence") or m.get("confidence_score") or m.get("plddt")
                        if name and isinstance(c, (int,float)):
                            conf_map[name] = float(c)
                # Exemple simple: {"confidence_score": 83.2}
                if "confidence_score" in data:
                    # On appliquera à tous si on ne trouve rien de mieux
                    conf_map["__global__"] = float(data["confidence_score"])

        for pdb in pred_dir.glob("*.pdb"):
            fname = pdb.stem  # ex: "model_0"
            conf = conf_map.get(fname, conf_map.get("__global__", 0.0))
            contacts, mean_b = interface_contacts_and_b(pdb, cutoff=6.0)
            # Filtre doux pour éviter les interfaces quasi-nulles
            if contacts < 5:
                conf *= 0.1
            # Score final (simple et robuste)
            score = 0.7 * conf + 0.3 * contacts + 0.05 * mean_b
            candidates.append((score, pdb))

    if not candidates:
        return []
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in candidates[:5]]




def prepare_protein_ligand(datapoint_id: str, protein: Protein, ligands: list[SmallMolecule], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:
    """
    Prepare input dict and CLI args for a protein-ligand prediction.
    You can return multiple configurations to run by returning a list of (input_dict, cli_args) tuples.
    Args:
        datapoint_id: The unique identifier for this datapoint
        protein: The protein sequence
        ligands: A list of a single small molecule ligand object 
        input_dict: Prefilled input dict
        msa_dir: Directory containing MSA files (for computing relative paths)
    Returns:
        List of tuples of (final input dict that will get exported as YAML, list of CLI args). Each tuple represents a separate configuration to run.
    """
    # Please note:
    # `protein` is a single-chain target protein sequence with id A
    # `ligands` contains a single small molecule ligand object with unknown binding sites
    # you can modify input_dict to change the input yaml file going into the prediction, e.g.
    # ```
    # input_dict["constraints"] = [{
    #   "contact": {
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME], 
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME]
    #   }
    # }]
    # ```
    #
    # will add contact constraints to the input_dict

    # Example: predict 5 structures
    cli_args = ["--diffusion_samples", "5"]
    return [(input_dict, cli_args)]


def post_process_protein_ligand(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path]) -> List[Path]:
    """
    Return ranked model files for protein-ligand submission.
    Args:
        datapoint: The original datapoint object
        input_dicts: List of input dictionaries used for predictions (one per config)
        cli_args_list: List of command line arguments used for predictions (one per config)
        prediction_dirs: List of directories containing prediction results (one per config)
    Returns: 
        Sorted pdb file paths that should be used as your submission.
    """
    # Collect all PDBs from all configurations
    all_pdbs = []
    for prediction_dir in prediction_dirs:
        config_pdbs = sorted(prediction_dir.glob(f"{datapoint.datapoint_id}_config_*_model_*.pdb"))
        all_pdbs.extend(config_pdbs)
    
    # Sort all PDBs and return their paths
    all_pdbs = sorted(all_pdbs)
    return all_pdbs

# -----------------------------------------------------------------------------
# ---- End of participant section ---------------------------------------------
# -----------------------------------------------------------------------------


DEFAULT_OUT_DIR = Path("predictions")
DEFAULT_SUBMISSION_DIR = Path("submission")
DEFAULT_INPUTS_DIR = Path("inputs")

ap = argparse.ArgumentParser(
    description="Hackathon scaffold for Boltz predictions",
    epilog="Examples:\n"
            "  Single datapoint: python predict_hackathon.py --input-json examples/specs/example_protein_ligand.json --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate\n"
            "  Multiple datapoints: python predict_hackathon.py --input-jsonl examples/test_dataset.jsonl --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate",
    formatter_class=argparse.RawDescriptionHelpFormatter
)

input_group = ap.add_mutually_exclusive_group(required=True)
input_group.add_argument("--input-json", type=str,
                        help="Path to JSON datapoint for a single datapoint")
input_group.add_argument("--input-jsonl", type=str,
                        help="Path to JSONL file with multiple datapoint definitions")

ap.add_argument("--msa-dir", type=Path,
                help="Directory containing MSA files (for computing relative paths in YAML)")
ap.add_argument("--submission-dir", type=Path, required=False, default=DEFAULT_SUBMISSION_DIR,
                help="Directory to place final submissions")
ap.add_argument("--intermediate-dir", type=Path, required=False, default=Path("hackathon_intermediate"),
                help="Directory to place generated input YAML files and predictions")
ap.add_argument("--group-id", type=str, required=False, default=None,
                help="Group ID to set for submission directory (sets group rw access if specified)")
ap.add_argument("--result-folder", type=Path, required=False, default=None,
                help="Directory to save evaluation results. If set, will automatically run evaluation after predictions.")

args = ap.parse_args()


def _prefill_input_dict(datapoint_id: str, proteins: Iterable[Protein], ligands: Optional[list[SmallMolecule]] = None, msa_dir: Optional[Path] = None) -> dict:
    """
    Prepare input dict for Boltz YAML.
    """
    seqs = []
    for p in proteins:
        if msa_dir and p.msa:
            if Path(p.msa).is_absolute():
                msa_full_path = Path(p.msa)
            else:
                msa_full_path = msa_dir / p.msa
            try:
                msa_relative_path = os.path.relpath(msa_full_path, Path.cwd())
            except ValueError:
                msa_relative_path = str(msa_full_path)
        else:
            msa_relative_path = p.msa
        entry = {
            "protein": {
                "id": p.id,
                "sequence": p.sequence,
                "msa": msa_relative_path
            }
        }
        seqs.append(entry)
    if ligands:
        def _format_ligand(ligand: SmallMolecule) -> dict:
            output =  {
                "ligand": {
                    "id": ligand.id,
                    "smiles": ligand.smiles
                }
            }
            return output
        
        for ligand in ligands:
            seqs.append(_format_ligand(ligand))
    doc = {
        "version": 1,
        "sequences": seqs,
    }
    return doc


def _run_boltz_and_collect(datapoint) -> None:
    """
    New flow: prepare input dict, write yaml, run boltz, post-process, copy submissions.
    """
    out_dir = args.intermediate_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    subdir = args.submission_dir / datapoint.datapoint_id
    subdir.mkdir(parents=True, exist_ok=True)

    # Prepare input dict and CLI args
    base_input_dict = _prefill_input_dict(datapoint.datapoint_id, datapoint.proteins, datapoint.ligands, args.msa_dir)

    if datapoint.task_type == "protein_complex":
        configs = prepare_protein_complex(datapoint.datapoint_id, datapoint.proteins, base_input_dict, args.msa_dir)
    elif datapoint.task_type == "protein_ligand":
        configs = prepare_protein_ligand(datapoint.datapoint_id, datapoint.proteins[0], datapoint.ligands, base_input_dict, args.msa_dir)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    # Run boltz for each configuration
    all_input_dicts = []
    all_cli_args = []
    all_pred_subfolders = []
    
    input_dir = args.intermediate_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    for config_idx, (input_dict, cli_args) in enumerate(configs):
        # Write input YAML with config index suffix
        yaml_path = input_dir / f"{datapoint.datapoint_id}_config_{config_idx}.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(input_dict, f, sort_keys=False)

        # Run boltz
        cache = os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz"))
        fixed = [
            "boltz", "predict", str(yaml_path),
            "--devices", "1",
            "--out_dir", str(out_dir),
            "--cache", cache,
            "--no_kernels",
            "--output_format", "pdb",
        ]
        cmd = fixed + cli_args
        print(f"Running config {config_idx}:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        # Compute prediction subfolder for this config
        pred_subfolder = out_dir / f"boltz_results_{datapoint.datapoint_id}_config_{config_idx}" / "predictions" / f"{datapoint.datapoint_id}_config_{config_idx}"
        
        all_input_dicts.append(input_dict)
        all_cli_args.append(cli_args)
        all_pred_subfolders.append(pred_subfolder)

    # Post-process and copy submissions
    if datapoint.task_type == "protein_complex":
        ranked_files = post_process_protein_complex(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)
    elif datapoint.task_type == "protein_ligand":
        ranked_files = post_process_protein_ligand(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    if not ranked_files:
        raise FileNotFoundError(f"No model files found for {datapoint.datapoint_id}")

    for i, file_path in enumerate(ranked_files[:5]):
        target = subdir / (f"model_{i}.pdb" if file_path.suffix == ".pdb" else f"model_{i}{file_path.suffix}")
        shutil.copy2(file_path, target)
        print(f"Saved: {target}")

    if args.group_id:
        try:
            subprocess.run(["chgrp", "-R", args.group_id, str(subdir)], check=True)
            subprocess.run(["chmod", "-R", "g+rw", str(subdir)], check=True)
        except Exception as e:
            print(f"WARNING: Failed to set group ownership or permissions: {e}")


def _load_datapoint(path: Path):
    """Load JSON datapoint file."""
    with open(path) as f:
        return Datapoint.from_json(f.read())


def _run_evaluation(input_file: str, task_type: str, submission_dir: Path, result_folder: Path):
    """
    Run the appropriate evaluation script based on task type.
    
    Args:
        input_file: Path to the input JSON or JSONL file
        task_type: Either "protein_complex" or "protein_ligand"
        submission_dir: Directory containing prediction submissions
        result_folder: Directory to save evaluation results
    """
    script_dir = Path(__file__).parent
    
    if task_type == "protein_complex":
        eval_script = script_dir / "evaluate_abag.py"
        cmd = [
            "python", str(eval_script),
            "--dataset-file", input_file,
            "--submission-folder", str(submission_dir),
            "--result-folder", str(result_folder)
        ]
    elif task_type == "protein_ligand":
        eval_script = script_dir / "evaluate_asos.py"
        cmd = [
            "python", str(eval_script),
            "--dataset-file", input_file,
            "--submission-folder", str(submission_dir),
            "--result-folder", str(result_folder)
        ]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    print(f"\n{'=' * 80}")
    print(f"Running evaluation for {task_type}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")
    
    subprocess.run(cmd, check=True)
    print(f"\nEvaluation complete. Results saved to {result_folder}")


def _process_jsonl(jsonl_path: str, msa_dir: Optional[Path] = None):

    
    """Process multiple datapoints from a JSONL file."""
    print(f"Processing JSONL file: {jsonl_path}")

    for line_num, line in enumerate(Path(jsonl_path).read_text().splitlines(), 1):
        if not line.strip():
            continue

        print(f"\n--- Processing line {line_num} ---")

        try:
            datapoint = Datapoint.from_json(line)
            _run_boltz_and_collect(datapoint)

        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON on line {line_num}: {e}")
            continue
        except Exception as e:
            print(f"ERROR: Failed to process datapoint on line {line_num}: {e}")
            raise e
            continue

            
def _process_json(json_path: str, msa_dir: Optional[Path] = None):

    
    """Process a single datapoint from a JSON file."""
    print(f"Processing JSON file: {json_path}")

    try:
        datapoint = _load_datapoint(Path(json_path))
        _run_boltz_and_collect(datapoint)
    except Exception as e:
        print(f"ERROR: Failed to process datapoint: {e}")
        raise

        
def main():

    
    """Main entry point for the hackathon scaffold."""
    # Determine task type from first datapoint for evaluation
    task_type = None
    input_file = None
    
    if args.input_json:
        input_file = args.input_json
        _process_json(args.input_json, args.msa_dir)
        # Get task type from the single datapoint
        try:
            datapoint = _load_datapoint(Path(args.input_json))
            task_type = datapoint.task_type
        except Exception as e:
            print(f"WARNING: Could not determine task type: {e}")
    elif args.input_jsonl:
        input_file = args.input_jsonl
        _process_jsonl(args.input_jsonl, args.msa_dir)
        # Get task type from first datapoint in JSONL
        try:
            with open(args.input_jsonl) as f:
                first_line = f.readline().strip()
                if first_line:
                    first_datapoint = Datapoint.from_json(first_line)
                    task_type = first_datapoint.task_type
        except Exception as e:
            print(f"WARNING: Could not determine task type: {e}")
    
    # Run evaluation if result folder is specified and task type was determined
    if args.result_folder and task_type and input_file:
        try:
            _run_evaluation(input_file, task_type, args.submission_dir, args.result_folder)
        except Exception as e:
            print(f"WARNING: Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

