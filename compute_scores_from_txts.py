import os
import csv
import math
from typing import Dict, List, Tuple, Optional
INPUT_ROOT = ''
OUTPUT_ROOT = ''
GROUND_TRUTH_FOLDER = 'groundtruth'
TXT_EXTS = {'.txt'}
Event = Tuple[int, float, str]
Point = Tuple[int, float]

def point_key(p: Point) -> Tuple[int, float]:
    return (p[0], p[1])

def point_leq(a: Point, b: Point) -> bool:
    return (a[0], a[1]) <= (b[0], b[1])

def point_lt(a: Point, b: Point) -> bool:
    return (a[0], a[1]) < (b[0], b[1])

def point_min(points: List[Point]) -> Point:
    return min(points, key=point_key)

def safe_float(x: str) -> float:
    return float(x.strip())

def beat_str(v: float) -> str:
    if abs(v - round(v)) < 1e-09:
        return str(int(round(v)))
    return f'{v:.6f}'.rstrip('0').rstrip('.')

def parse_chord_txt(path: str) -> List[Event]:
    events: List[Event] = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line == 'NO_CHORDS_FOUND':
                continue
            parts = line.split('\t')
            data = {}
            for part in parts:
                if '=' not in part:
                    continue
                k, v = part.split('=', 1)
                data[k.strip()] = v.strip()
            if 'measure' not in data or 'beat' not in data or 'chord' not in data:
                continue
            measure = int(data['measure'])
            beat = safe_float(data['beat'])
            chord = data['chord']
            events.append((measure, beat, chord))
    events.sort(key=lambda x: (x[0], x[1]))
    return events

def crop_events_to_point(events: List[Event], end_point: Point) -> List[Event]:
    return [ev for ev in events if point_leq((ev[0], ev[1]), end_point)]

def get_last_point(events: List[Event]) -> Optional[Point]:
    if not events:
        return None
    last = events[-1]
    return (last[0], last[1])

def list_immediate_subfolders(root: str) -> List[str]:
    out = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isdir(full):
            out.append(name)
    out.sort()
    return out

def discover_txts_recursive(root: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in TXT_EXTS:
                continue
            abs_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(abs_path, root)
            mapping[rel_path] = abs_path
    return mapping

def union_sample_points(cropped_events_by_folder: Dict[str, List[Event]], crop_end: Point) -> List[Point]:
    pts = set()
    for events in cropped_events_by_folder.values():
        for m, b, _ in events:
            p = (m, b)
            if point_leq(p, crop_end):
                pts.add(p)
    pts.add(crop_end)
    out = sorted(pts, key=point_key)
    return out

def active_label_at_point(events: List[Event], p: Point) -> str:
    active = 'N'
    for m, b, chord in events:
        if point_leq((m, b), p):
            active = chord
        else:
            break
    return active

def label_sequence_on_grid(events: List[Event], sample_points: List[Point]) -> List[str]:
    return [active_label_at_point(events, p) for p in sample_points]

def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    n = len(seq1)
    m = len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        a = seq1[i - 1]
        for j in range(1, m + 1):
            b = seq2[j - 1]
            cost = 0 if a == b else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]

def compute_weighted_prf(y_true: List[str], y_pred: List[str]) -> Tuple[float, float, float, float]:
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0)
    labels = sorted(set(y_true) | set(y_pred))
    total_support = 0
    weighted_precision_sum = 0.0
    weighted_recall_sum = 0.0
    weighted_f1_sum = 0.0
    exact_matches = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            exact_matches += 1
    for label in labels:
        tp = 0
        fp = 0
        fn = 0
        support = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == label:
                support += 1
            if yt == label and yp == label:
                tp += 1
            elif yt != label and yp == label:
                fp += 1
            elif yt == label and yp != label:
                fn += 1
        if support == 0:
            continue
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        total_support += support
        weighted_precision_sum += precision * support
        weighted_recall_sum += recall * support
        weighted_f1_sum += f1 * support
    if total_support == 0:
        return (0.0, 0.0, 0.0, 0.0)
    weighted_precision = weighted_precision_sum / total_support
    weighted_recall = weighted_recall_sum / total_support
    weighted_f1 = weighted_f1_sum / total_support
    exact_match_rate = exact_matches / n
    return (weighted_precision, weighted_recall, weighted_f1, exact_match_rate)

def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    precision, recall, f1, exact_match_rate = compute_weighted_prf(y_true, y_pred)
    edit_distance = levenshtein_distance(y_true, y_pred)
    max_len = max(len(y_true), len(y_pred), 1)
    norm_similarity = 1.0 - edit_distance / max_len
    return {'precision': precision, 'recall': recall, 'f1': f1, 'exact_match_rate': exact_match_rate, 'edit_distance': float(edit_distance), 'norm_similarity': norm_similarity}

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def evaluate_method_against_gt(method_name: str, gt_map: Dict[str, str], method_map: Dict[str, str], all_maps_by_folder: Dict[str, Dict[str, str]]) -> Tuple[List[Dict], Dict]:
    per_song_rows: List[Dict] = []
    gt_song_keys = sorted(gt_map.keys())
    matched_songs = 0
    missing_songs = 0
    metric_sums = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'exact_match_rate': 0.0, 'edit_distance': 0.0, 'norm_similarity': 0.0}
    for rel_path in gt_song_keys:
        gt_path = gt_map[rel_path]
        pred_path = method_map.get(rel_path)
        if pred_path is None:
            missing_songs += 1
            per_song_rows.append({'method': method_name, 'song': rel_path, 'status': 'MISSING', 'crop_measure': '', 'crop_beat': '', 'num_sample_points': 0, 'precision': '', 'recall': '', 'f1': '', 'exact_match_rate': '', 'edit_distance': '', 'norm_similarity': ''})
            continue
        available_events_by_folder: Dict[str, List[Event]] = {}
        available_last_points: List[Point] = []
        for folder_name, folder_map in all_maps_by_folder.items():
            path = folder_map.get(rel_path)
            if path is None:
                continue
            events = parse_chord_txt(path)
            if not events:
                continue
            last_pt = get_last_point(events)
            if last_pt is None:
                continue
            available_events_by_folder[folder_name] = events
            available_last_points.append(last_pt)
        if GROUND_TRUTH_FOLDER not in available_events_by_folder or method_name not in available_events_by_folder:
            missing_songs += 1
            per_song_rows.append({'method': method_name, 'song': rel_path, 'status': 'EMPTY_OR_MISSING', 'crop_measure': '', 'crop_beat': '', 'num_sample_points': 0, 'precision': '', 'recall': '', 'f1': '', 'exact_match_rate': '', 'edit_distance': '', 'norm_similarity': ''})
            continue
        crop_end = point_min(available_last_points)
        cropped_events_by_folder: Dict[str, List[Event]] = {}
        for folder_name, events in available_events_by_folder.items():
            cropped = crop_events_to_point(events, crop_end)
            cropped_events_by_folder[folder_name] = cropped
        sample_points = union_sample_points(cropped_events_by_folder, crop_end)
        gt_seq = label_sequence_on_grid(cropped_events_by_folder[GROUND_TRUTH_FOLDER], sample_points)
        pred_seq = label_sequence_on_grid(cropped_events_by_folder[method_name], sample_points)
        metrics = compute_metrics(gt_seq, pred_seq)
        matched_songs += 1
        for k in metric_sums:
            metric_sums[k] += metrics[k]
        per_song_rows.append({'method': method_name, 'song': rel_path, 'status': 'OK', 'crop_measure': crop_end[0], 'crop_beat': beat_str(crop_end[1]), 'num_sample_points': len(sample_points), 'precision': metrics['precision'], 'recall': metrics['recall'], 'f1': metrics['f1'], 'exact_match_rate': metrics['exact_match_rate'], 'edit_distance': metrics['edit_distance'], 'norm_similarity': metrics['norm_similarity']})
    if matched_songs > 0:
        summary = {'test_tag': method_name, 'matched_songs': matched_songs, 'missing_songs': missing_songs, 'precision': metric_sums['precision'] / matched_songs, 'recall': metric_sums['recall'] / matched_songs, 'f1': metric_sums['f1'] / matched_songs, 'exact_match_rate': metric_sums['exact_match_rate'] / matched_songs, 'avg_edit_distance': metric_sums['edit_distance'] / matched_songs, 'norm_similarity': metric_sums['norm_similarity'] / matched_songs}
    else:
        summary = {'test_tag': method_name, 'matched_songs': 0, 'missing_songs': missing_songs, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'exact_match_rate': 0.0, 'avg_edit_distance': 0.0, 'norm_similarity': 0.0}
    return (per_song_rows, summary)

def main() -> None:
    ensure_dir(OUTPUT_ROOT)
    if not os.path.isdir(INPUT_ROOT):
        raise SystemExit(f'Input root does not exist: {INPUT_ROOT}')
    subfolders = list_immediate_subfolders(INPUT_ROOT)
    if GROUND_TRUTH_FOLDER not in subfolders:
        raise SystemExit(f"Ground truth folder '{GROUND_TRUTH_FOLDER}' not found inside: {INPUT_ROOT}")
    method_folders = [x for x in subfolders if x != GROUND_TRUTH_FOLDER]
    if not method_folders:
        raise SystemExit('No prediction subfolders found to evaluate.')
    print('Input root:', INPUT_ROOT)
    print('Output root:', OUTPUT_ROOT)
    print('Ground truth:', GROUND_TRUTH_FOLDER)
    print('Methods:', method_folders)
    all_maps_by_folder: Dict[str, Dict[str, str]] = {}
    for folder_name in subfolders:
        folder_root = os.path.join(INPUT_ROOT, folder_name)
        mapping = discover_txts_recursive(folder_root)
        all_maps_by_folder[folder_name] = mapping
        print(f"Discovered {len(mapping)} txt files in '{folder_name}'")
    gt_map = all_maps_by_folder[GROUND_TRUTH_FOLDER]
    summary_rows: List[Dict] = []
    for method_name in method_folders:
        print(f'\nEvaluating: {method_name}')
        method_map = all_maps_by_folder[method_name]
        per_song_rows, summary = evaluate_method_against_gt(method_name=method_name, gt_map=gt_map, method_map=method_map, all_maps_by_folder=all_maps_by_folder)
        summary_rows.append(summary)
        per_song_csv = os.path.join(OUTPUT_ROOT, f'{method_name}_per_song.csv')
        write_csv(per_song_csv, per_song_rows, fieldnames=['method', 'song', 'status', 'crop_measure', 'crop_beat', 'num_sample_points', 'precision', 'recall', 'f1', 'exact_match_rate', 'edit_distance', 'norm_similarity'])
        print(f'Saved per-song results: {per_song_csv}')
    summary_rows.sort(key=lambda x: x['f1'], reverse=True)
    summary_csv = os.path.join(OUTPUT_ROOT, 'summary.csv')
    write_csv(summary_csv, summary_rows, fieldnames=['test_tag', 'matched_songs', 'missing_songs', 'precision', 'recall', 'f1', 'exact_match_rate', 'avg_edit_distance', 'norm_similarity'])
    print(f'\nSaved summary: {summary_csv}')
    print('\nRanking by F1:')
    for i, row in enumerate(summary_rows, 1):
        print(f"{i}. {row['test_tag']} | F1={row['f1']:.6f} | P={row['precision']:.6f} | R={row['recall']:.6f} | EM={row['exact_match_rate']:.6f}")
if __name__ == '__main__':
    main()
