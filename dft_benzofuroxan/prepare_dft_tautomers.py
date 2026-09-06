#!/usr/bin/env python3
"""
Generate 3D lowest-energy conformers and ORCA input files for
benzofuroxan-5-carboxylic acid tautomers (1-oxide vs 3-oxide).
"""
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

TAUTOMERS = {
    "tautomer_1_oxide": {
        "name": "1-oxido-2,1,3-benzoxadiazol-1-ium-5-carboxylic acid (1-oxide)",
        "smiles": "O=C(O)c1ccc2c(c1)no[n+]2[O-]",
        "charge": 0,
        "mult": 1,
    },
    "tautomer_3_oxide": {
        "name": "3-oxido-2,1,3-benzoxadiazol-3-ium-5-carboxylic acid (3-oxide)",
        "smiles": "O=C(O)c1ccc2no[n+]([O-])c2c1",
        "charge": 0,
        "mult": 1,
    }
}

ORCA_TEMPLATE = """! {method} {basis} {dispersion} {rijcosx} CPCM(Water) Opt Freq
%pal nprocs {nprocs} end
%maxcore {maxcore}

* xyz {charge} {mult}
{xyz_coords}*
"""

def generate_conformer(smiles: str, num_confs: int = 50, seed: int = 42):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    if not cids:
        raise RuntimeError(f"Embedding failed for SMILES: {smiles}")
    
    # MMFF optimization
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    energies = [r[1] for r in res]
    best_idx = energies.index(min(energies))
    return mol, best_idx, min(energies)

def mol_to_xyz_block(mol, conf_id: int) -> str:
    conf = mol.GetConformer(conf_id)
    lines = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():<2} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
    return "\n".join(lines) + "\n"

def main():
    out_dir = Path("/home/diego/PoliScreen/dft_benzofuroxan")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # DFT settings
    method = "B3LYP"
    basis = "def2-SVP"
    dispersion = "D4"
    rijcosx = "def2/J RIJCOSX"
    nprocs = 6   # 6 cores keeps laptop responsive and cool
    maxcore = 1500 # 1.5 GB per core
    
    print("=" * 65)
    print("  PREPARING BENZOFUROXAN TAUTOMER DFT CALCULATIONS")
    print("=" * 65)
    
    for key, data in TAUTOMERS.items():
        print(f"\nProcessing {data['name']}...")
        mol, best_idx, e_mmff = generate_conformer(data["smiles"])
        xyz_coords = mol_to_xyz_block(mol, best_idx)
        
        # Save XYZ file
        xyz_path = out_dir / f"{key}.xyz"
        num_atoms = mol.GetNumAtoms()
        with open(xyz_path, "w") as f:
            f.write(f"{num_atoms}\n{data['name']} - lowest MMFF conf {best_idx} ({e_mmff:.2f} kcal/mol)\n")
            f.write(xyz_coords)
        print(f"  [+] Saved XYZ: {xyz_path.name} (MMFF E: {e_mmff:.2f} kcal/mol)")
        
        # Save ORCA INP file
        inp_path = out_dir / f"{key}.inp"
        inp_content = ORCA_TEMPLATE.format(
            method=method,
            basis=basis,
            dispersion=dispersion,
            rijcosx=rijcosx,
            nprocs=nprocs,
            maxcore=maxcore,
            charge=data["charge"],
            mult=data["mult"],
            xyz_coords=xyz_coords
        )
        with open(inp_path, "w") as f:
            f.write(inp_content)
        print(f"  [+] Saved ORCA input: {inp_path.name}")
        
    print("\n" + "=" * 65)
    print("All input files prepared successfully in:")
    print(f"  {out_dir}")
    print("=" * 65)

if __name__ == "__main__":
    main()
