import argparse
import re
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
from meeko.writer import PDBQTWriterLegacy


# чистим названия файлов для извлечения названий молекул
def safe_filename(name, fallback="mol"):
    if not name:
        name = fallback

    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)

    return name


# получаем имя из sdf файла
def get_name_sdf(mol, idx):
    if mol.HasProp("name") and mol.GetProp("name").strip():
        return safe_filename(mol.GetProp("name"), f"sdf_{idx:04d}")

    if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
        return safe_filename(mol.GetProp("_Name"), f"sdf_{idx:04d}")

    return f"sdf_{idx:04d}"


# есть ли 3д структура
def has_3d(mol):
    return mol.GetNumConformers() > 0 and mol.GetConformer().Is3D()


# добавляем Н и строим 3д если нет
def ensure_h_3d(mol, seed):
    mol = Chem.AddHs(mol)

    if not has_3d(mol):
        params = AllChem.ETKDGv3()
        if seed is not None:
            params.randomSeed = seed

        status = AllChem.EmbedMolecule(mol, params)

        if status != 0:
            raise RuntimeError("не удалось построить 3D координаты")

        AllChem.UFFOptimizeMolecule(mol, maxIters=200)

    return mol


# получаем pdbqt
def rdkit_to_pdbqt(mol):

    preparator = MoleculePreparation(  # доп звдвние, больше 5 атомов - гибкие
        rigid_macrocycles=False,
        min_ring_size=6
    )
    setups = preparator.prepare(mol)
    setup = setups[0]

    result = PDBQTWriterLegacy.write_string(setup)
    pdbqt_str = result[0]

    return pdbqt_str


# читаем sdf и записываем pdbqt
def convert_sdf(sdf_path, out_dir, seed):
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)

    for i, mol in enumerate(suppl):
        if mol is None:
            print(f"не удалось прочитать молекулу {i}")
            continue

        try:
            mol = ensure_h_3d(mol, seed=seed)

            name = get_name_sdf(mol, i)

            pdbqt_str = rdkit_to_pdbqt(mol)

            out_file = out_dir / f"{name}.pdbqt"
            with open(out_file, "w") as f:
                f.write(pdbqt_str)

            print(f"ок {out_file}")

        except Exception as e:
            print(f"ошибка mol {i}: {e}")


# читаем smiles и записывает pdbqt
def convert_smi(smi_path, out_dir, seed):
    with open(smi_path) as f:

        for i, line in enumerate(f):
            if not line.strip():
                continue

            parts = line.strip().split()
            smi = parts[0]
            name = parts[1] if len(parts) > 1 else f"mol_{i}"
            name = safe_filename(name)

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"не удалось прочитать smiles молекулы {i}")
                continue

            try:
                mol = ensure_h_3d(mol, seed=seed)

                pdbqt_str = rdkit_to_pdbqt(mol)

                out_file = out_dir / f"{name}.pdbqt"
                with open(out_file, "w") as f:
                    f.write(pdbqt_str)

                print(f"ок {out_file}")

            except Exception as e:
                print(f"ошибка mol {i}: {e}")


def main():
    parser = argparse.ArgumentParser(description="SDF/SMI - PDBQT")
    parser.add_argument("-i", "--input", required=True,
                        help="входной файл (.sdf / .smi)")
    parser.add_argument("-o", "--out", required=True, help="папка для PDBQT")
    parser.add_argument("--seed", type=int, help="seed для 3D")

    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() == ".sdf":
        convert_sdf(in_path, out_dir, args.seed)
    elif in_path.suffix.lower() in [".smi", ".smiles"]:
        convert_smi(in_path, out_dir, args.seed)
    else:
        raise ValueError("поддерживаются только .sdf и .smi")


if __name__ == "__main__":
    main()
