import argparse
from multiprocessing import Pool
from credit.gefs import download_gefs_run, process_member
from functools import partial


def main():
    parser = argparse.ArgumentParser(
        description="Initialize CREDIT models with GEFS data"
    )
    parser.add_argument(
        "-d", "--date", help="Initialization date in YYYY-MM-DD HHMM format."
    )
    parser.add_argument(
        "-p", "--path", help="Path to where raw GEFS data will be downloaded."
    )
    parser.add_argument(
        "-o", "--out", help="Path to where processed GEFS data will be saved."
    )
    parser.add_argument(
        "-w", "--weights", help="Path to ESMF_RegridWeightGen regrid weight file."
    )
    parser.add_argument(
        "-m",
        "--members",
        type=int,
        default=30,
        help="Number of GEFS perturbation members to download.",
    )
    parser.add_argument(
        "-n", "--nprocs", type=int, default=1, help="Number of processes to use."
    )
    parser.add_argument(
        "-v",
        "--variables",
        type=str,
        default="",
        help="Variables to use separated by commas.",
    )
    args = parser.parse_args()
    init_date_str = args.date
    download_path = args.path
    out_path = args.out
    weight_file = args.weights
    n_pert_members = args.members
    variables = args.variables.split(",")
    download_gefs_run(init_date_str, out_path, n_pert_members)
    member_names = ["c0"] + [f"p{m:02d}" for m in range(n_pert_members)]
    with Pool(args.nprocs) as pool:
        pool.map(
            partial(
                process_member,
                member_path=download_path,
                out_path=out_path,
                init_date_str=init_date_str,
                variables=variables,
                weight_file=weight_file,
            ),
            member_names,
        )
    return


if __name__ == "__main__":
    main()
