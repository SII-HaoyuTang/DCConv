import pandas as pd
from tqdm import tqdm
import os  # 添加os模块


def read_folder(folder_path="./133660_curatedQM9_outof_133885"):  # 修改函数名和参数
    file_name = []

    mol_len = []
    atom_coords = []

    A = []
    B = []
    C = []
    miu = []
    alpha = []
    homo = []
    lumo = []
    gap = []
    R2 = []
    zpve = []
    Uo = []
    U = []
    H = []
    G = []
    Cv = []

    smiles = []
    vibrationalfrequence = []
    inchi = []

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
            # for n in lines:
            #     print(n)
            #     print('..')

            atom_coords_str = ""

            for line in lines[2:-4]:
                # atom_coords_lines.append(line)
                atom_coords_str = atom_coords_str + line + "\n"

            atom_coords.append(atom_coords_str)

            mol_len.append(lines[0])

            prop = lines[1].split()

            if prop[0] == 'gdb':
                A.append(prop[2])
                B.append(prop[3])
                C.append(prop[4])
                miu.append(prop[5])
                alpha.append(prop[6])
                homo.append(prop[7])
                lumo.append(prop[8])
                gap.append(prop[9])
                R2.append(prop[10])
                zpve.append(prop[11])
                Uo.append(prop[12])
                U.append(prop[13])
                H.append(prop[14])
                G.append(prop[15])
                Cv.append(prop[16])
            else:
                A.append(prop[1])
                B.append(prop[2])
                C.append(prop[3])
                miu.append(prop[4])
                alpha.append(prop[5])
                homo.append(prop[6])
                lumo.append(prop[7])
                gap.append(prop[8])
                R2.append(prop[9])
                zpve.append(prop[10])
                Uo.append(prop[11])
                U.append(prop[12])
                H.append(prop[13])
                G.append(prop[14])
                Cv.append(prop[15])

            vibrationalfrequence.append(lines[-3])
            smiles.append(lines[-2])  # .split()[0]
            inchi.append(lines[-1])

            # print('atom_coords_lines:',atom_coords_str)
            # print('smiles',smiles)

    return (
        file_name,
        mol_len,
        atom_coords,
        A,
        B,
        C,
        miu,
        alpha,
        homo,
        lumo,
        gap,
        R2,
        zpve,
        Uo,
        U,
        H,
        G,
        Cv,
        vibrationalfrequence,
        smiles,
        inchi,
    )


# 调用时使用文件夹路径
(
    file_name,
    mol_len,
    atom_coords,
    A,
    B,
    C,
    miu,
    alpha,
    homo,
    lumo,
    gap,
    R2,
    zpve,
    Uo,
    U,
    H,
    G,
    Cv,
    vibrationalfrequence,
    smiles,
    inchi,
) = read_folder()  # 修改为文件夹路径

data = pd.DataFrame()

data["file_name"] = file_name
data["mol_len"] = mol_len
data["atom_coords"] = atom_coords
data["A"] = A
data["B"] = B
data["C"] = C
data["miu"] = miu
data["alpha"] = alpha
data["homo"] = homo
data["lumo"] = lumo
data["gap"] = gap
data["R2"] = R2
data["zpve"] = zpve
data["Uo"] = Uo
data["U"] = U
data["H"] = H
data["G"] = G
data["Cv"] = Cv
data["smiles"] = smiles
data["vibrationalfrequence"] = vibrationalfrequence
data["inchi"] = inchi

print(data)
data.to_json("../../../../data/qm9.json")
