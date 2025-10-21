from engine import train_one_epoch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training Prototype Network")
    return parser.parse_args()

def main(args):
  train_one_epoch()


if __name__ == '__main__':
    args = parse_args()
    main(args)