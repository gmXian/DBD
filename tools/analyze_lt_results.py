import argparse
import os

import pandas as pd


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def summarize(df, prefix):
    total = int(df[f'{prefix}_total'].sum())
    correct = int(df[f'{prefix}_correct'].sum())
    macro = float(df[f'{prefix}_acc'].mean()) if len(df) else 0.0
    worst = float(df[f'{prefix}_acc'].min()) if len(df) else 0.0
    return {
        'micro_acc': safe_div(correct, total),
        'macro_acc': macro,
        'worst_acc': worst,
        'total': total,
        'correct': correct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_csv', type=str, required=True)
    parser.add_argument('--adv_csv', type=str, required=True)
    parser.add_argument('--split_csv', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    clean = pd.read_csv(args.clean_csv)
    adv = pd.read_csv(args.adv_csv)
    split = pd.read_csv(args.split_csv)

    clean = clean.rename(columns={
        'test_count': 'clean_total',
        'vanilla_correct': 'clean_vanilla_correct',
        'dbd_correct': 'clean_dbd_correct',
        'vanilla_acc': 'clean_vanilla_acc',
        'dbd_acc': 'clean_dbd_acc',
        'db_score_mean': 'clean_db_score_mean',
        'db_score_std': 'clean_db_score_std',
        'high_db_ratio': 'clean_high_db_ratio',
    })
    adv = adv.rename(columns={
        'test_count': 'adv_total',
        'vanilla_correct': 'adv_vanilla_correct',
        'dbd_correct': 'adv_dbd_correct',
        'vanilla_acc': 'adv_vanilla_acc',
        'dbd_acc': 'adv_dbd_acc',
        'db_score_mean': 'adv_db_score_mean',
        'db_score_std': 'adv_db_score_std',
        'high_db_ratio': 'adv_high_db_ratio',
    })

    keep_clean = ['class_id', 'class_name', 'clean_total', 'clean_vanilla_correct', 'clean_dbd_correct', 'clean_vanilla_acc', 'clean_dbd_acc', 'clean_db_score_mean', 'clean_db_score_std', 'clean_high_db_ratio']
    keep_adv = ['class_id', 'adv_total', 'adv_vanilla_correct', 'adv_dbd_correct', 'adv_vanilla_acc', 'adv_dbd_acc', 'adv_db_score_mean', 'adv_db_score_std', 'adv_high_db_ratio']
    merged = clean[keep_clean].merge(adv[keep_adv], on='class_id', how='inner')
    merged = merged.merge(split, on='class_id', how='left')
    merged.to_csv(os.path.join(args.out_dir, 'classwise_merged.csv'), index=False)

    rows = []
    for group_name, group_df in [('all', merged)] + list(merged.groupby('split')):
        row = {
            'split': group_name,
            'num_classes': len(group_df),
            'clean_num_samples': int(group_df['clean_total'].sum()),
            'adv_num_samples': int(group_df['adv_total'].sum()),
            'clean_vanilla_micro': safe_div(group_df['clean_vanilla_correct'].sum(), group_df['clean_total'].sum()),
            'clean_dbd_micro': safe_div(group_df['clean_dbd_correct'].sum(), group_df['clean_total'].sum()),
            'adv_vanilla_micro': safe_div(group_df['adv_vanilla_correct'].sum(), group_df['adv_total'].sum()),
            'adv_dbd_micro': safe_div(group_df['adv_dbd_correct'].sum(), group_df['adv_total'].sum()),
            'clean_vanilla_macro': float(group_df['clean_vanilla_acc'].mean()),
            'clean_dbd_macro': float(group_df['clean_dbd_acc'].mean()),
            'adv_vanilla_macro': float(group_df['adv_vanilla_acc'].mean()),
            'adv_dbd_macro': float(group_df['adv_dbd_acc'].mean()),
            'adv_dbd_worst': float(group_df['adv_dbd_acc'].min()),
            'clean_db_score_mean': float(group_df['clean_db_score_mean'].mean()),
            'adv_db_score_mean': float(group_df['adv_db_score_mean'].mean()),
            'db_score_gap': float(group_df['adv_db_score_mean'].mean() - group_df['clean_db_score_mean'].mean()),
            'clean_high_db_ratio': float(group_df['clean_high_db_ratio'].mean()),
            'adv_high_db_ratio': float(group_df['adv_high_db_ratio'].mean()),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, 'lt_summary.csv'), index=False)
    print(summary.to_string(index=False))
    print(f"Saved merged class-wise results and summary to {args.out_dir}")


if __name__ == '__main__':
    main()
