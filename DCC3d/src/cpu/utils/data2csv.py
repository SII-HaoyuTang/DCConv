import os  # 添加os模块

import pandas as pd
from numpy.ma.core import indices
from tqdm import tqdm


def read_folder(folder_path="./133660_curatedQM9_outof_133885"):  # 修改函数名和参数
    file_name = []

    mol_len = []
    atom_coords_x = []  # in angstrom
    atom_coords_y = []
    atom_coords_z = []
    atom_mass = []  # in relative atom mass
    atom_valence_electrons = []
    atom_radius = []  # in angstrom
    atom_mulliken_charge = []  # Mulliken partial charges (in e) on atoms
    molecule_id = []
    molecule_id_per_atom = []
    U = []
    # 获取文件夹中的所有文件
    all_files = os.listdir(folder_path)

    # 遍历文件夹中的所有文件
    for filename in tqdm(all_files):
        file_path = os.path.join(folder_path, filename)

        # 跳过目录，只处理文件
        if not os.path.isfile(file_path):
            continue

        file_name.append(filename)

        with open(file_path, "r") as file_obj:  # 直接打开文件
            # 读取文件内容
            content = file_obj.read()

            text_content = content

            # 分割内容为行
            lines = text_content.strip().split("\n")

            atom_coords_str = ""

            prop = lines[1].split()
            molecule_id.append(prop[1])
            U.append(prop[13])

            mol_len.append(lines[0])

            for line in lines[2:-4]:
                # atom_coords_lines.append(line)
                para = line.split("\t")
                atom_coords_x.append(para[1])
                atom_coords_y.append(para[2])
                atom_coords_z.append(para[3])
                atom_mulliken_charge.append(para[4])
                molecule_id_per_atom.append(prop[1])
                if para[0] == "H":
                    atom_mass.append(1.008)
                    atom_valence_electrons.append(1)
                    atom_radius.append(0.37)
                elif para[0] == "C":
                    atom_mass.append(12.011)
                    atom_valence_electrons.append(4)
                    atom_radius.append(0.77)
                elif para[0] == "N":
                    atom_mass.append(14.007)
                    atom_valence_electrons.append(5)
                    atom_radius.append(0.75)
                elif para[0] == "O":
                    atom_mass.append(15.999)
                    atom_valence_electrons.append(6)
                    atom_radius.append(0.74)
                elif para[0] == "F":
                    atom_mass.append(18.998)
                    atom_valence_electrons.append(7)
                    atom_radius.append(0.71)
                else:
                    raise ValueError("atom_type_error")

    return (
        file_name,
        mol_len,
        atom_coords_x,
        atom_coords_y,
        atom_coords_z,
        atom_mass,
        atom_valence_electrons,
        atom_radius,
        atom_mulliken_charge,
        molecule_id,
        molecule_id_per_atom,
        U,
    )


def main():
    # 调用时使用文件夹路径
    (
        file_name,
        mol_len,
        atom_coords_x,
        atom_coords_y,
        atom_coords_z,
        atom_mass,
        atom_valence_electrons,
        atom_radius,
        atom_mulliken_charge,
        molecule_id,
        molecule_id_per_atom,
        U,
    ) = read_folder()  # 修改为文件夹路径

    data = pd.DataFrame()
    indics = pd.DataFrame()

    data["molecule_id"] = molecule_id_per_atom
    data["x"] = atom_coords_x
    data["y"] = atom_coords_y
    data["z"] = atom_coords_z
    data["atom_mass"] = atom_mass
    data["atom_valence_electrons"] = atom_valence_electrons
    data["atom_radius"] = atom_radius
    data["atom_mulliken_charge"] = atom_mulliken_charge

    indics["molecule_id"] = molecule_id
    indics["num_atoms"] = mol_len
    indics["internal_energy"] = U  # Internal energy at 298.15K

    data.to_csv("../../../../data/qm9.csv", index=False)
    indics.to_csv("../../../../data/qm9_indices.csv", index=False)


if __name__ == "__main__":
    main()
