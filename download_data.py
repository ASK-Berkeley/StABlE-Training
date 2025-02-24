"""
Downloads all datasets used in "Stability-Aware Training of Machine Learning Force Fields
with Differentiable Boltzmann Estimators" (https://arxiv.org/abs/2402.13984).

1. Aspirin from MD17
2. Ac-Ala3-NHMe from MD22
3. Water from Forces are Not Enough (https://arxiv.org/abs/2210.07237)

"""

import os
import subprocess
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download_path",
        type=str,
        default="./DATAPATH",
        help="Path at which to download data.",
    )
    args = parser.parse_args()

    # Aspirin 1k (MD17)
    command = [
        "python",
        "data/md17.py",
        "--molecule",
        "aspirin",
        "--data_path",
        os.path.join(args.download_path, "md17"),
        "--db_path",
        os.path.join(args.download_path, "md17"),
        "--size",
        "1k",
    ]

    subprocess.run(command, check=True)

    # Aspirin (MD17) 1k Contiguous
    command = [
        "python",
        "data/md17.py",
        "--molecule",
        "aspirin",
        "--data_path",
        os.path.join(args.download_path, "contiguous-md17"),
        "--db_path",
        os.path.join(args.download_path, "contiguous-md17"),
        "--size",
        "1k",
        "--contiguous",
    ]
    subprocess.run(command, check=True)

    # Aspirin 10k (MD17)
    command = [
        "python",
        "data/md17.py",
        "--molecule",
        "aspirin",
        "--data_path",
        os.path.join(args.download_path, "md17"),
        "--db_path",
        os.path.join(args.download_path, "md17"),
        "--size",
        "10k",
    ]

    subprocess.run(command, check=True)

    # Aspirin (MD17) 10k Contiguous
    command = [
        "python",
        "data/md17.py",
        "--molecule",
        "aspirin",
        "--data_path",
        os.path.join(args.download_path, "contiguous-md17"),
        "--db_path",
        os.path.join(args.download_path, "contiguous-md17"),
        "--size",
        "10k",
        "--contiguous",
    ]
    subprocess.run(command, check=True)

    # ac-Ala3-NHMe (MD22) 25%
    command = [
        "python",
        "data/md22.py",
        "--molecule",
        "ac_Ala3_NHMe",
        "--data_path",
        os.path.join(args.download_path, "md22"),
        "--db_path",
        os.path.join(args.download_path, "md22"),
        "--size",
        "25percent",
    ]

    subprocess.run(command, check=True)

    # ac-Ala3-NHMe (MD22) 25% Contiguous
    command = [
        "python",
        "data/md22.py",
        "--molecule",
        "ac_Ala3_NHMe",
        "--data_path",
        os.path.join(args.download_path, "contiguous-md22"),
        "--db_path",
        os.path.join(args.download_path, "contiguous-md22"),
        "--size",
        "25percent",
        "--contiguous",
    ]
    subprocess.run(command, check=True)

    # ac-Ala3-NHMe (MD22) 100%
    command = [
        "python",
        "data/md22.py",
        "--molecule",
        "ac_Ala3_NHMe",
        "--data_path",
        os.path.join(args.download_path, "md22"),
        "--db_path",
        os.path.join(args.download_path, "md22"),
        "--size",
        "100percent",
    ]

    subprocess.run(command, check=True)

    # ac-Ala3-NHMe (MD22) 100% Contiguous
    command = [
        "python",
        "data/md22.py",
        "--molecule",
        "ac_Ala3_NHMe",
        "--data_path",
        os.path.join(args.download_path, "contiguous-md22"),
        "--db_path",
        os.path.join(args.download_path, "contiguous-md22"),
        "--size",
        "100percent",
        "--contiguous",
    ]
    subprocess.run(command, check=True)

    # Water 1k
    command = [
        "python",
        "data/water.py",
        "--data_path",
        os.path.join(args.download_path, "water"),
        "--db_path",
        os.path.join(args.download_path, "water"),
        "--size",
        "1k",
    ]

    subprocess.run(command, check=True)

    # Water 1k Contiguous
    command = [
        "python",
        "data/water.py",
        "--data_path",
        os.path.join(args.download_path, "contiguous-water"),
        "--db_path",
        os.path.join(args.download_path, "contiguous-water"),
        "--size",
        "1k",
        "--contiguous",
    ]
    subprocess.run(command, check=True)

    # Water 10k
    command = [
        "python",
        "data/water.py",
        "--data_path",
        os.path.join(args.download_path, "water"),
        "--db_path",
        os.path.join(args.download_path, "water"),
        "--size",
        "10k",
    ]

    subprocess.run(command, check=True)

    # Water 10k Contiguous
    command = [
        "python",
        "data/water.py",
        "--data_path",
        os.path.join(args.download_path, "contiguous-water"),
        "--db_path",
        os.path.join(args.download_path, "contiguous-water"),
        "--size",
        "10k",
        "--contiguous",
    ]
    subprocess.run(command, check=True)
