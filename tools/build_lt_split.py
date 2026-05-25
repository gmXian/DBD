import argparse
import csv
from collections import Counter


def get_split(n):
    if n > 100:
        return 'many'
    if n >= 20:
        return 'medium'
    return 'few'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_txt', type=str, required=True)
    parser.add_argument('--out_csv', type=str, required=True)
    args = parser.parse_args()

    counter = Counter()
    with open(args.train_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, label = line.split()
            counter[int(label)] += 1

    rows = []
    split_counter = Counter()
    for class_id in range(1000):
        n = int(counter[class_id])
        split = get_split(n)
        split_counter[split] += 1
        rows.append({'class_id': class_id, 'train_count': n, 'split': split})

    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['class_id', 'train_count', 'split'])
        writer.writeheader()
        writer.writerows(rows)

    print(f'Saved {args.out_csv}')
    print('split counts:', dict(split_counter))
    print('total train images:', sum(counter.values()))


if __name__ == '__main__':
    main()
