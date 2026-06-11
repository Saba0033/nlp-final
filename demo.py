import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    print(args.query)
    print("dosent work yet")


if __name__ == "__main__":
    main()
