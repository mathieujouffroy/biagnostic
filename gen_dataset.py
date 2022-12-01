import os
import json
import multiprocessing
from train_framework.utils import parse_args, set_seed, set_logging
from train_framework.dataloader import BratsDatasetGenerator

def main():

    args = parse_args()
    set_seed(args)

    brats_generator = BratsDatasetGenerator(args)
    brats_generator.print_info()

    if not os.path.exists(f"{args.ds_path}subvolumes"):
        os.makedirs(f"{args.ds_path}subvolumes")

    brats_generator.gen_subvol_coords()

    print(f"\nN_CPU: {multiprocessing.cpu_count()}\n")
    for id_lst in [brats_generator.train_ids, brats_generator.val_ids, brats_generator.test_ids]:
        print(f"id_lst:{id_lst}")
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pool.map(brats_generator.generate_sub_volume, id_lst)
        print("\n--set done --")

    split_sets = dict()
    split_sets['train'] = []
    split_sets['val'] = []
    split_sets['test'] = []
    for filename in os.listdir(f"{args.ds_path}subvolumes"):
        idx = int(filename.split('_')[1])
        if idx in brats_generator.train_ids:
            split_sets['train'].append(filename)
        if idx in brats_generator.val_ids:
            split_sets['val'].append(filename)
        if idx in brats_generator.test_ids:
            split_sets['test'].append(filename)

    with open(f"{args.ds_path}split_sets.json", 'w') as f:
        json.dump(split_sets, f, indent=4)


if __name__ == "__main__":
    main()