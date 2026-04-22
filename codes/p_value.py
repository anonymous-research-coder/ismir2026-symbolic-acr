#!/usr/bin/env python3
# Example usage: python p_value.py

from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
INPUT_DIR = Path('')
SUMMARY_PVALUES_CSV = INPUT_DIR / 'summary_p_values.csv'
SUMMARY_COMPARISONS = [('model_a', 'model_b'), ('model_a', 'model_c'), ('model_a', 'model_e'), ('model_a', 'model_d')]
SUMMARY_METRICS = ['precision', 'recall', 'f1', 'norm_similarity', 'edit_distance']
MAIN_MODEL = 'model_a'
PAIRWISE_RESULTS_CSV = INPUT_DIR / 'pvalues_pairwise.csv'
MAIN_MODEL_RESULTS_CSV = INPUT_DIR / f'pvalues_{MAIN_MODEL}_vs_others.csv'
SUMMARY_TXT = INPUT_DIR / 'pvalues_summary.txt'
METRICS_HIGHER_BETTER = ['precision', 'recall', 'f1', 'exact_match_rate', 'norm_similarity']
METRICS_LOWER_BETTER = ['edit_distance']
ALL_METRICS = METRICS_HIGHER_BETTER + METRICS_LOWER_BETTER

def make_summary_pvalues_csv(pairwise_df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for method_a, method_b in SUMMARY_COMPARISONS:
        sub = pairwise_df[(pairwise_df['method_a'] == method_a) & (pairwise_df['method_b'] == method_b) & pairwise_df['metric'].isin(SUMMARY_METRICS)].copy()
        if sub.empty:
            sub = pairwise_df[(pairwise_df['method_a'] == method_b) & (pairwise_df['method_b'] == method_a) & pairwise_df['metric'].isin(SUMMARY_METRICS)].copy()
            if not sub.empty:
                for _, row in sub.iterrows():
                    rows.append({'model_1': method_a, 'model_2': method_b, 'metric': row['metric'], 'mean_model_1': row['mean_b'], 'mean_model_2': row['mean_a'], 'delta_model_1_minus_model_2': row['mean_b'] - row['mean_a'], 'better_model': row['better_method'], 'n_songs': int(row['n_songs']), 'ttest_p': row['ttest_p'], 'ttest_sig': row['ttest_sig'], 'wilcoxon_p': row['wilcoxon_p'], 'wilcoxon_sig': row['wilcoxon_sig']})
            continue
        for _, row in sub.iterrows():
            rows.append({'model_1': method_a, 'model_2': method_b, 'metric': row['metric'], 'mean_model_1': row['mean_a'], 'mean_model_2': row['mean_b'], 'delta_model_1_minus_model_2': row['delta_a_minus_b'], 'better_model': row['better_method'], 'n_songs': int(row['n_songs']), 'ttest_p': row['ttest_p'], 'ttest_sig': row['ttest_sig'], 'wilcoxon_p': row['wilcoxon_p'], 'wilcoxon_sig': row['wilcoxon_sig']})
    out_df = pd.DataFrame(rows)
    metric_order = {'precision': 0, 'recall': 1, 'f1': 2, 'norm_similarity': 3, 'edit_distance': 4}
    comp_order = {('mine', 'predictions_p_txt'): 0, ('mine', 'predictions_c_txt'): 1, ('mine', 'augmentednet'): 2, ('mine', 'mine_pop'): 3}
    out_df['comp_order'] = out_df.apply(lambda r: comp_order.get((r['model_1'], r['model_2']), 999), axis=1)
    out_df['metric_order'] = out_df['metric'].map(metric_order)
    out_df = out_df.sort_values(['comp_order', 'metric_order']).drop(columns=['comp_order', 'metric_order'])
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')

def load_per_song_csvs(input_dir: Path) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory does not exist: {input_dir}')
    for csv_path in sorted(input_dir.glob('*_per_song.csv')):
        method_name = csv_path.name.replace('_per_song.csv', '')
        df = pd.read_csv(csv_path)
        if 'status' not in df.columns or 'song' not in df.columns:
            print(f'[SKIP] Missing required columns in {csv_path.name}')
            continue
        df = df[df['status'] == 'OK'].copy()
        missing_metrics = [m for m in ALL_METRICS if m not in df.columns]
        if missing_metrics:
            print(f'[SKIP] Missing metrics in {csv_path.name}: {missing_metrics}')
            continue
        for metric in ALL_METRICS:
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
        df = df.dropna(subset=['song'] + ALL_METRICS).copy()
        df = df.drop_duplicates(subset=['song']).copy()
        result[method_name] = df
    if not result:
        raise RuntimeError(f'No valid *_per_song.csv files found in: {input_dir}')
    return result

def significance_stars(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'

def safe_wilcoxon(a: pd.Series, b: pd.Series) -> Tuple[float | None, float | None]:
    try:
        res = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        return (float(res.statistic), float(res.pvalue))
    except Exception:
        return (None, None)

def compare_two_methods(method_a: str, df_a: pd.DataFrame, method_b: str, df_b: pd.DataFrame) -> List[Dict]:
    merged = pd.merge(df_a[['song'] + ALL_METRICS], df_b[['song'] + ALL_METRICS], on='song', suffixes=('_a', '_b'), how='inner')
    rows: List[Dict] = []
    if merged.empty:
        for metric in ALL_METRICS:
            rows.append({'method_a': method_a, 'method_b': method_b, 'metric': metric, 'n_songs': 0, 'mean_a': None, 'mean_b': None, 'delta_a_minus_b': None, 'better_method': '', 'ttest_stat': None, 'ttest_p': None, 'ttest_sig': '', 'wilcoxon_stat': None, 'wilcoxon_p': None, 'wilcoxon_sig': ''})
        return rows
    for metric in ALL_METRICS:
        a = merged[f'{metric}_a'].astype(float)
        b = merged[f'{metric}_b'].astype(float)
        mean_a = float(a.mean())
        mean_b = float(b.mean())
        delta = mean_a - mean_b
        if metric in METRICS_HIGHER_BETTER:
            better_method = method_a if mean_a > mean_b else method_b if mean_b > mean_a else 'tie'
        else:
            better_method = method_a if mean_a < mean_b else method_b if mean_b < mean_a else 'tie'
        try:
            t_res = ttest_rel(a, b, nan_policy='omit')
            t_stat = float(t_res.statistic)
            t_p = float(t_res.pvalue)
        except Exception:
            t_stat, t_p = (None, None)
        w_stat, w_p = safe_wilcoxon(a, b)
        rows.append({'method_a': method_a, 'method_b': method_b, 'metric': metric, 'n_songs': int(len(merged)), 'mean_a': mean_a, 'mean_b': mean_b, 'delta_a_minus_b': delta, 'better_method': better_method, 'ttest_stat': t_stat, 'ttest_p': t_p, 'ttest_sig': significance_stars(t_p), 'wilcoxon_stat': w_stat, 'wilcoxon_p': w_p, 'wilcoxon_sig': significance_stars(w_p)})
    return rows

def format_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return 'NA'
    if p < 0.001:
        return '<0.001'
    return f'{p:.4f}'

def write_summary_txt(out_path: Path, pairwise_df: pd.DataFrame, main_model: str) -> None:
    lines: List[str] = []
    lines.append('Statistical significance summary')
    lines.append('=' * 80)
    lines.append('')
    lines.append(f'Input folder: {INPUT_DIR}')
    lines.append(f'Main model: {main_model}')
    lines.append('Tests: paired t-test and Wilcoxon signed-rank')
    lines.append('Significance: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant')
    lines.append('')
    if main_model not in set(pairwise_df['method_a']).union(set(pairwise_df['method_b'])):
        lines.append(f"Main model '{main_model}' not found in comparisons.")
        out_path.write_text('\n'.join(lines), encoding='utf-8')
        return
    main_rows = pairwise_df[(pairwise_df['method_a'] == main_model) | (pairwise_df['method_b'] == main_model)].copy()
    metrics_order = ALL_METRICS
    other_methods = sorted(set(main_rows['method_a']).union(set(main_rows['method_b'])) - {main_model})
    for other in other_methods:
        lines.append(f'{main_model} vs {other}')
        lines.append('-' * 80)
        sub = main_rows[(main_rows['method_a'] == main_model) & (main_rows['method_b'] == other) | (main_rows['method_a'] == other) & (main_rows['method_b'] == main_model)].copy()
        if sub.empty:
            lines.append('  No overlapping songs.')
            lines.append('')
            continue
        for metric in metrics_order:
            row = sub[sub['metric'] == metric]
            if row.empty:
                continue
            row = row.iloc[0]
            if row['method_a'] == main_model:
                mean_main = row['mean_a']
                mean_other = row['mean_b']
                delta = row['delta_a_minus_b']
            else:
                mean_main = row['mean_b']
                mean_other = row['mean_a']
                delta = -row['delta_a_minus_b']
            lines.append(f"  {metric}: {main_model}={mean_main:.6f}, {other}={mean_other:.6f}, delta={delta:+.6f}, t-test p={format_p(row['ttest_p'])} {row['ttest_sig']}, Wilcoxon p={format_p(row['wilcoxon_p'])} {row['wilcoxon_sig']}, n={int(row['n_songs'])}")
        lines.append('')
    out_path.write_text('\n'.join(lines), encoding='utf-8')

def main() -> None:
    print(f'[INFO] Loading per-song CSV files from:\n  {INPUT_DIR}\n')
    method_dfs = load_per_song_csvs(INPUT_DIR)
    print('[INFO] Methods found:')
    for method_name, df in method_dfs.items():
        print(f'  - {method_name}: {len(df)} OK songs')
    print()
    methods = sorted(method_dfs.keys())
    pairwise_rows: List[Dict] = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method_a = methods[i]
            method_b = methods[j]
            print(f'[INFO] Comparing {method_a} vs {method_b}')
            rows = compare_two_methods(method_a=method_a, df_a=method_dfs[method_a], method_b=method_b, df_b=method_dfs[method_b])
            pairwise_rows.extend(rows)
    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df['metric_order'] = pairwise_df['metric'].map({m: i for i, m in enumerate(ALL_METRICS)})
    pairwise_df = pairwise_df.sort_values(by=['method_a', 'method_b', 'metric_order']).drop(columns=['metric_order'])
    pairwise_df.to_csv(PAIRWISE_RESULTS_CSV, index=False, encoding='utf-8-sig')
    print(f'\n[SAVED] Pairwise results: {PAIRWISE_RESULTS_CSV}')
    main_df = pairwise_df[(pairwise_df['method_a'] == MAIN_MODEL) | (pairwise_df['method_b'] == MAIN_MODEL)].copy()
    main_df.to_csv(MAIN_MODEL_RESULTS_CSV, index=False, encoding='utf-8-sig')
    print(f'[SAVED] Main model results: {MAIN_MODEL_RESULTS_CSV}')
    write_summary_txt(SUMMARY_TXT, pairwise_df, MAIN_MODEL)
    print(f'[SAVED] Summary text: {SUMMARY_TXT}')
    make_summary_pvalues_csv(pairwise_df, SUMMARY_PVALUES_CSV)
    print(f'[SAVED] Summary p-values CSV: {SUMMARY_PVALUES_CSV}')
    print('\n[DONE] Statistical testing completed.')
if __name__ == '__main__':
    main()
