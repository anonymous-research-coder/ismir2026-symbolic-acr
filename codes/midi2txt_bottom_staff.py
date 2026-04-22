# Example usage: python midi2txt_bottom_staff.py -i "INPUT_MIDI_FOLDER" -o "OUTPUT_TXT_FOLDER" -w 8

import os
import argparse
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

import mir_eval
from music21 import converter, stream, meter, key, tempo, chord as m21chord, note


INPUT_ROOT = r""
OUTPUT_ROOT = r""
WORKERS = cpu_count()
MIDI_EXTS = {".mid", ".midi"}

PITCH_CLASS_TO_NAME_SHARP = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}


def deepcopy_stream(obj):
    try:
        return obj.__deepcopy__({})
    except Exception:
        import copy
        return copy.deepcopy(obj)


def collect_globals(src):
    out = []

    for ts in src.recurse().getElementsByClass(meter.TimeSignature):
        if ts.offset <= 0.01:
            out.append(deepcopy_stream(ts))
            break

    for ks in src.recurse().getElementsByClass(key.KeySignature):
        if ks.offset <= 0.01:
            out.append(deepcopy_stream(ks))
            break

    for mm in src.recurse().getElementsByClass(tempo.MetronomeMark):
        if mm.offset <= 0.01:
            out.append(deepcopy_stream(mm))
            break

    return out


def keep_target_part_from_score(src):
    if isinstance(src, stream.Part):
        target_part = deepcopy_stream(src)
        new_score = stream.Score()
        new_score.insert(0, target_part)
        return new_score

    if not isinstance(src, stream.Score):
        sc = stream.Score()
        sc.insert(0, src)
        src = sc

    parts = src.parts

    if len(parts) == 0:
        raise ValueError("Score has no parts.")

    if len(parts) == 1:
        target_part = deepcopy_stream(parts[0])
    else:
        target_part = deepcopy_stream(parts[1])

    new_score = stream.Score()

    for g in collect_globals(src):
        new_score.insert(0, g)

    inst = target_part.getInstrument(returnDefault=False)
    if inst is not None:
        new_score.insert(0, deepcopy_stream(inst))

    new_score.insert(0, target_part)

    return new_score


def get_unique_pitch_classes(ch):
    return sorted(set(p.pitchClass for p in ch.pitches))


def safe_root_pitch_class(ch):
    try:
        r = ch.root()
        if r is not None:
            return r.pitchClass
    except Exception:
        pass

    try:
        b = ch.bass()
        if b is not None:
            return b.pitchClass
    except Exception:
        pass

    return None


def validate_or_fallback_label(label):
    try:
        mir_eval.chord.encode(label)
        return label
    except Exception:
        return "N"


def infer_chord_label(ch):
    pcs = get_unique_pitch_classes(ch)

    if not pcs:
        return "N"

    if len(pcs) == 1:
        return validate_or_fallback_label(f"{PITCH_CLASS_TO_NAME_SHARP[pcs[0]]}:1")

    root_pc = safe_root_pitch_class(ch)
    if root_pc is None:
        return "N"

    intervals = sorted({(pc - root_pc) % 12 for pc in pcs})
    root_name = PITCH_CLASS_TO_NAME_SHARP[root_pc]

    patterns = {
        frozenset({0, 4, 7}): "maj",
        frozenset({0, 3, 7}): "min",
        frozenset({0, 3, 6}): "dim",
        frozenset({0, 4, 8}): "aug",
        frozenset({0, 7}): "5",
        frozenset({0, 2, 7}): "sus2",
        frozenset({0, 5, 7}): "sus4",
        frozenset({0, 4, 7, 10}): "7",
        frozenset({0, 4, 7, 11}): "maj7",
        frozenset({0, 3, 7, 10}): "min7",
        frozenset({0, 3, 7, 11}): "minmaj7",
        frozenset({0, 3, 6, 9}): "dim7",
        frozenset({0, 3, 6, 10}): "hdim7",
    }

    q = patterns.get(frozenset(intervals))

    if q is not None:
        return validate_or_fallback_label(f"{root_name}:{q}")

    if {0, 4, 7}.issubset(intervals):
        return validate_or_fallback_label(f"{root_name}:maj")

    if {0, 3, 7}.issubset(intervals):
        return validate_or_fallback_label(f"{root_name}:min")

    return "N"


def extract_chords(score):
    bottom_score = keep_target_part_from_score(score)
    part = bottom_score.parts[0]
    chordified = part.chordify()

    events = []
    prev = None

    for ch in chordified.recurse().getElementsByClass(m21chord.Chord):
        if not ch.pitches:
            continue

        label = infer_chord_label(ch)
        measure = ch.measureNumber if ch.measureNumber is not None else -1
        beat = float(ch.beat)
        sig = (measure, beat, label)

        if sig == prev:
            continue

        events.append(sig)
        prev = sig

    return events


def beat_str(v):
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.6f}".rstrip("0").rstrip(".")


def write_txt(events, out_path):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf8") as f:
        if not events:
            f.write("NO_CHORDS_FOUND\n")
            return

        for m, b, c in events:
            f.write(f"measure={m}\tbeat={beat_str(b)}\tchord={c}\n")


def process_file(in_path, out_txt):
    try:
        score = converter.parse(in_path)
        events = extract_chords(score)
        write_txt(events, out_txt)
        return "OK", f"{os.path.basename(in_path)} ({len(events)} chords)"
    except Exception as e:
        return "SKIP", f"{os.path.basename(in_path)}: {e}"


def find_midis(root):
    files = []

    for dirpath, _, names in os.walk(root):
        for n in names:
            if os.path.splitext(n)[1].lower() in MIDI_EXTS:
                files.append(os.path.join(dirpath, n))

    return files


def build_out_path(in_root, out_root, in_path):
    rel = os.path.relpath(in_path, in_root)
    rel = os.path.splitext(rel)[0]
    return os.path.join(out_root, rel + ".txt")


def process_root(input_root, output_root, workers):
    os.makedirs(output_root, exist_ok=True)

    files = find_midis(input_root)

    if not files:
        print("No MIDI files found.")
        return

    tasks = [(f, build_out_path(input_root, output_root, f)) for f in files]
    total = len(tasks)

    print("Files discovered:", total)
    print("Workers:", workers)

    ok = 0
    skip = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_file, in_path, out_path) for in_path, out_path in tasks]

        for i, future in enumerate(as_completed(futures), 1):
            status, msg = future.result()

            if status == "OK":
                ok += 1
                print(f"[{i}/{total}] OK   {msg}")
            else:
                skip += 1
                print(f"[{i}/{total}] SKIP {msg}")

    print("\nFinished")
    print("OK:", ok)
    print("Skipped:", skip)
    print("Total:", total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=INPUT_ROOT)
    parser.add_argument("-o", "--output", default=OUTPUT_ROOT)
    parser.add_argument("-w", "--workers", type=int, default=WORKERS)

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        raise SystemExit("Input path not folder")

    workers = max(args.workers, 1)
    process_root(args.input, args.output, workers)


if __name__ == "__main__":
    main()