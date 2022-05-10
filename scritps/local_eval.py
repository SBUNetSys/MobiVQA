#!/usr/bin/env python3
# coding: utf-8
import argparse
import json

from vqa.train.metrics import compute_vqa_accuracy


def main(args):
    pred_file = args.pred_file
    gt_file = args.gt_file
    predictions = json.load(open(pred_file))
    gt_data = [json.loads(line) for line in open(gt_file)]
    accuracy = compute_vqa_accuracy(gt_data, predictions)
    print(f'{accuracy=:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_file', type=str)
    parser.add_argument('-g', '--gt_file', type=str)
    main(parser.parse_args())
