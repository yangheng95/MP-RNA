import argparse

from omnigenome import AutoBench

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="rgb")
    parser.add_argument(
        "--gfm", type=str, default="anonymous8/OmniGenome-52M"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--overwrite", type=bool, default=False)

    args = parser.parse_args()
    bench_root = args.root

    bench = AutoBench(
        bench_root=bench_root, model_name_or_path=args.gfm, overwrite=args.overwrite
    )
    bench.run(autocast=False, batch_size=args.batch_size)
