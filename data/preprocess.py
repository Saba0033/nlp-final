"""Run all preprocess steps, or one at a time:

  python data/preprocess_squad.py
  python data/preprocess_wiki.py
  python data/preprocess_jm.py
"""

import argparse

import preprocess_jm
import preprocess_squad
import preprocess_wiki
from preprocess_utils import load_cfg, save_splits, split_by_groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad", action="store_true", help="only squad")
    parser.add_argument("--wiki", action="store_true", help="only wiki")
    parser.add_argument("--jm", action="store_true", help="only j&m demo corpus")
    args = parser.parse_args()

    run_all = not (args.squad or args.wiki or args.jm)
    cfg = load_cfg()
    seed = cfg["data"]["seed"]

    if run_all or args.wiki:
        print("wikipedia")
        wiki_groups = preprocess_wiki.build_wiki(cfg)
        save_splits(cfg["data"]["wiki_dir"], *split_by_groups(wiki_groups, seed))

    if run_all or args.squad:
        print("squad")
        squad_groups = preprocess_squad.build_squad(cfg)
        save_splits(cfg["data"]["squad_dir"], *split_by_groups(squad_groups, seed))

    if run_all or args.jm:
        print("demo corpus")
        preprocess_jm.build_jm(cfg["data"]["corpus_path"])

    print("done")


if __name__ == "__main__":
    main()
