from tqdm import tqdm
from transformers import PreTrainedTokenizer
from music21 import converter, harmony, pitch, note, interval, stream, meter, chord, duration
import mir_eval
from copy import deepcopy
import numpy as np
import os
import json
import ast
import random
from music_utils import detect_key, get_transposition_interval, transpose_score
INT_TO_ROOT_SHARP = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}
MIR_QUALITIES = mir_eval.chord.QUALITIES
EXT_MIR_QUALITIES = deepcopy(MIR_QUALITIES)
for k in list(MIR_QUALITIES.keys()) + ['7(b9)', '7(#9)', '7(#11)', '7(b13)']:
    _, semitone_bitmap, _ = mir_eval.chord.encode('C' + (len(k) > 0) * ':' + k, reduce_extended_chords=True)
    EXT_MIR_QUALITIES[k] = semitone_bitmap

class CSGridMLMTokenizer(PreTrainedTokenizer):

    def __init__(self, quantization='16th', fixed_length=None, vocab=None, special_tokens=None, use_pc_roll=True, use_full_range_melody=True, intertwine_bar_info=True, trim_start=False, **kwargs):
        self.quantization = quantization
        self.fixed_length = fixed_length
        self.intertwine_bar_info = intertwine_bar_info
        self.trim_start = trim_start
        self.use_pc_roll = use_pc_roll
        self.use_full_range_melody = use_full_range_melody
        self.pianoroll_dim = 88 * int(use_full_range_melody) + 12 * int(use_pc_roll) + int(intertwine_bar_info)
        self.no_chord = '<nc>'
        self.nc_token = '<nc>'
        self.csl_token = '<s>'
        self.bar_token = '<bar>'
        self.vocab = {'<unk>': 0, '<pad>': 1, '<s>': 2, '</s>': 3, '<nc>': 4, '<mask>': 5, '<bar>': 6}
        self.update_ids_to_tokens()
        if vocab is not None:
            self.vocab = vocab
            self.update_ids_to_tokens()
        self.special_tokens = special_tokens if special_tokens is not None else {}
        self._added_tokens_encoder = {}
        super().__init__(unk_token='<unk>', pad_token='<pad>', bos_token='<s>', eos_token='</s>', mask_token='<mask>', **kwargs)
        self._refresh_special_token_ids()
        self.time_quantization = []
        self.time_signatures = []
        max_quarters = 10
        subdivisions = [0, 0.16, 0.25, 0.33, 0.5, 0.66, 0.75, 0.83]
        for quarter in range(max_quarters):
            for subdivision in subdivisions:
                quant_time = round(quarter + subdivision, 3)
                self.time_quantization.append(quant_time)
        self.time_signatures = self.infer_time_signatures_from_quantization(self.time_quantization, max_quarters)
        if vocab is None:
            chromatic_roots = []
            for i in range(12):
                pitch_obj = pitch.Pitch(i)
                if '-' in pitch_obj.name:
                    pitch_obj = pitch_obj.getEnharmonic()
                chromatic_roots.append(pitch_obj.name)
            qualities = list(EXT_MIR_QUALITIES.keys())
            for root in chromatic_roots:
                for quality in qualities:
                    chord_token = root + (len(quality) > 0) * ':' + quality
                    if chord_token not in self.vocab:
                        self.vocab[chord_token] = len(self.vocab)
            self.update_ids_to_tokens()
        self.total_vocab_size = len(self.vocab)
        self._refresh_special_token_ids()

    def _refresh_special_token_ids(self):
        self.unk_token_id = self.vocab.get('<unk>', 0)
        self.pad_token_id = self.vocab.get('<pad>', 1)
        self.bos_token_id = self.vocab.get('<s>', 2)
        self.eos_token_id = self.vocab.get('</s>', 3)
        self.nc_token_id = self.vocab.get('<nc>', 4)
        self.mask_token_id = self.vocab.get('<mask>', 5)
        self.bar_token_id = self.vocab.get('<bar>', 6)

    def _to_float(self, x):
        return float(x)

    def _q_floor(self, x, ql_per_quantum):
        return int(np.floor(float(x) / float(ql_per_quantum)))

    def _q_round(self, x, ql_per_quantum):
        return int(np.round(float(x) / float(ql_per_quantum)))

    def _q_ceil(self, x, ql_per_quantum):
        return int(np.ceil(float(x) / float(ql_per_quantum)))

    def infer_time_signatures_from_quantization(self, time_quantization, max_quarters=10):
        inferred_time_signatures = set()
        for measure_length in range(1, max_quarters + 1):
            measure_tokens = [t for t in time_quantization if int(t) < measure_length]
            inferred_time_signatures.add((measure_length, 4))
            for numerator in range(1, measure_length * 2 + 1):
                eighth_duration = 0.5
                valid_onsets = [i * eighth_duration for i in range(numerator)]
                if all((any((abs(t - onset) < 0.01 for t in measure_tokens)) for onset in valid_onsets)):
                    inferred_time_signatures.add((numerator, 8))
        quarter_signatures = {num for num, denom in inferred_time_signatures if denom == 4}
        cleaned_signatures = []
        for num, denom in inferred_time_signatures:
            if denom == 4:
                cleaned_signatures.append((num, denom))
            elif denom == 8 and num / 2 not in quarter_signatures:
                cleaned_signatures.append((num, denom))
        return sorted(cleaned_signatures)

    def update_ids_to_tokens(self):
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    def get_vocab(self):
        return getattr(self, 'vocab', {})

    def __len__(self):
        return len(self.get_vocab())

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get('<unk>', 0))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, '<unk>')

    def convert_tokens_to_ids(self, tokens):
        unk_id = self.vocab.get('<unk>', 0)
        if isinstance(tokens, str):
            return self.vocab.get(tokens, unk_id)
        return [self.vocab.get(token, unk_id) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.ids_to_tokens.get(ids, '<unk>')
        return [self.ids_to_tokens.get(i, '<unk>') for i in ids]

    def normalize_root_to_sharps(self, root):
        special_mapping = {'C-': 'B', 'D-': 'C#', 'E-': 'D#', 'F-': 'E', 'E#': 'F', 'G-': 'F#', 'A-': 'G#', 'B-': 'A#', 'B#': 'C', 'C##': 'D', 'D##': 'E', 'E##': 'F#', 'F##': 'G', 'G##': 'A', 'A##': 'B', 'B##': 'C#', 'C--': 'A#', 'D--': 'C', 'E--': 'D', 'F--': 'D#', 'G--': 'F', 'A--': 'G', 'B--': 'A'}
        if root in special_mapping:
            return special_mapping[root]
        pitch_obj = pitch.Pitch(root)
        return pitch_obj.name

    def get_closest_mir_eval_symbol(self, chord_symbol):
        ti = interval.Interval(chord_symbol.root(), pitch.Pitch('C'))
        tc = chord_symbol.transpose(ti)
        b = np.zeros(12)
        b[tc.pitchClasses] = 1
        similarity_max = -1
        key_max = '<unk>'
        for k in EXT_MIR_QUALITIES.keys():
            tmp_similarity = np.sum(b == EXT_MIR_QUALITIES[k])
            if similarity_max < tmp_similarity:
                similarity_max = tmp_similarity
                key_max = k
        return key_max

    def normalize_chord_symbol(self, chord_symbol):
        if hasattr(chord_symbol, 'root') and chord_symbol.root() is not None:
            root_name = chord_symbol.root().name
        else:
            return ('<unk>', '')
        root = self.normalize_root_to_sharps(root_name)
        quality = self.get_closest_mir_eval_symbol(chord_symbol)
        return (f'{root}', f'{quality}')

    def handle_chord_symbol(self, h):
        root_token, type_token = self.normalize_chord_symbol(h)
        chord_token = root_token + (len(type_token) > 0) * ':' + type_token
        if chord_token in self.vocab:
            chord_token_id = self.vocab[chord_token]
        else:
            chord_token = '<unk>'
            chord_token_id = self.vocab['<unk>']
        return (chord_token, chord_token_id)

    def decode_chord_symbol(self, harmony_tokens):
        raise NotImplementedError()

    def fit(self, corpus):
        pass

    def transform(self, corpus, add_start_harmony_token=True):
        tokens = []
        ids = []
        for file_path in tqdm(corpus, desc='Processing Files'):
            encoded = self.encode(file_path)
            harmony_tokens = encoded['harmony_tokens']
            harmony_ids = encoded['harmony_ids']
            tokens.append(harmony_tokens)
            ids.append(harmony_ids)
        return {'tokens': tokens, 'ids': ids}

    def fit_transform(self, corpus, add_start_harmony_token=True):
        self.fit(corpus)
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)

    def __call__(self, corpus, add_start_harmony_token=True):
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)

    def randomize_score(self, score, note_remove_pct=0.0, chord_remove_pct=0.0, note_change_pct=0.0):
        part = score.parts[0]
        notes = [n for n in part.notes if isinstance(n, note.Note)]
        chords = [c for c in part.notes if isinstance(c, harmony.ChordSymbol)]
        num_notes_remove = int(len(notes) * note_remove_pct)
        if num_notes_remove > 0 and len(notes) > 0:
            notes_to_remove = random.sample(notes, min(num_notes_remove, len(notes)))
            for n in notes_to_remove:
                part.remove(n)
        num_chords_remove = int(len(chords) * chord_remove_pct)
        if num_chords_remove > 0 and len(chords) > 0:
            chords_to_remove = random.sample(chords, min(num_chords_remove, len(chords)))
            for c in chords_to_remove:
                part.remove(c)
        num_notes_change = int(len(notes) * note_change_pct / 2)
        if num_notes_change > 0 and len(notes) > 0:
            notes_to_change = random.sample(notes, min(num_notes_change, len(notes)))
            for n in notes_to_change:
                shift_semitones = np.random.randint(-3, 3)
                n.transpose(shift_semitones, inPlace=True)
        return score

    def pitch_class_from_chord_token(self, chord_token):
        if chord_token in [self.nc_token, self.pad_token, self.bar_token]:
            return np.zeros(12, dtype=int)
        if chord_token not in self.vocab:
            return np.zeros(12, dtype=int)
        try:
            root, quality = chord_token.split(':')
        except ValueError:
            root = chord_token
            quality = ''
        if quality not in EXT_MIR_QUALITIES:
            return np.zeros(12, dtype=int)
        root_pitch = pitch.Pitch(root)
        root_pc = root_pitch.pitchClass
        quality_bitmap = EXT_MIR_QUALITIES[quality]
        pc_profile = np.roll(quality_bitmap, root_pc)
        return pc_profile

    def to_category(self, x, thresholds):
        if x <= thresholds[0]:
            return [1, 0, 0, 0]
        elif x < thresholds[1]:
            return [0, 1, 0, 0]
        else:
            return [0, 0, 1, 0]

    def compute_harmonic_rhythm_density(self, chord_token_ids):
        bars = []
        current_bar = []
        for tok in chord_token_ids:
            if tok == self.bar_token_id:
                if current_bar:
                    bars.append(current_bar)
                current_bar = []
            else:
                current_bar.append(tok)
        if current_bar:
            bars.append(current_bar)
        chord_counts = []
        for bar in bars:
            valid_chords = [t for t in bar if t not in (self.nc_token_id, self.pad_token_id)]
            if not valid_chords:
                continue
            changes = 1
            for prev, curr in zip(valid_chords, valid_chords[1:]):
                if curr != prev:
                    changes += 1
            chord_counts.append(changes)
        if not chord_counts:
            return (0.0, [0, 0, 0, 1])
        hrd = sum(chord_counts) / len(chord_counts)
        return (hrd, self.to_category(hrd, [1.0001, 1.5556]))

    def compute_harmonic_complexity(self, chord_tokens):
        if not chord_tokens:
            return (0.0, [0, 0, 0, 1])
        pitch_class_sum = np.zeros(12, dtype=float)
        for chord_token in chord_tokens:
            pc_profile = self.pitch_class_from_chord_token(chord_token)
            pitch_class_sum += pc_profile
        if np.sum(pitch_class_sum) == 0:
            return (0.0, [0, 0, 0, 1])
        pitch_class_dist = pitch_class_sum / np.sum(pitch_class_sum)
        entropy = -np.sum(pitch_class_dist * np.log(pitch_class_dist + 1e-12))
        return (float(entropy), self.to_category(entropy, [1.8225, 1.9254]))

    def encode(self, file_path, filler_token='<nc>', keep_durations=False, normalize_tonality=False):
        file_ext = file_path.split('.')[-1].lower()
        if file_ext in ['xml', 'mxl', 'musicxml']:
            return self.encode_musicXML(file_path, filler_token=filler_token, keep_durations=keep_durations, normalize_tonality=normalize_tonality)
        elif file_ext in ['mid', 'midi']:
            return self.encode_MIDI(file_path, filler_token=filler_token, keep_durations=keep_durations, normalize_tonality=normalize_tonality)
        else:
            raise ValueError(f'ERROR: unknown file extension: {file_ext}')

    def encode_musicXML(self, file_path, filler_token='<nc>', keep_durations=False, normalize_tonality=False):
        score = converter.parse(file_path)
        back_interval = None
        if normalize_tonality:
            original_key = detect_key(score)
            to_c_or_a_interval = get_transposition_interval(original_key)
            score = transpose_score(score, to_c_or_a_interval)
            back_interval = to_c_or_a_interval.reverse()
        time_signature = score.recurse().getElementsByClass(meter.TimeSignature).first()
        ts_num_list = [0] * 14
        ts_den_list = [0, 0]
        ts_num_list[int(min(max(time_signature.numerator - 2, 0), 13))] = 1
        ts_den_list[int(time_signature.denominator == 4)] = 1
        melody_part = score.parts[0].flatten()
        chords_part = None
        if len(score.parts) > 1:
            chords_part = score.parts[1].chordify().flatten()
        if self.quantization == '16th':
            ql_per_quantum = 1 / 4
        elif self.quantization == '8th':
            ql_per_quantum = 1 / 2
        elif self.quantization == '4th':
            ql_per_quantum = 1
        elif self.quantization == '32nd':
            ql_per_quantum = 1 / 8
        else:
            ql_per_quantum = 1 / 4
        first_chord_offset = None
        skip_steps = 0
        if self.trim_start:
            if chords_part is None:
                for el in melody_part.recurse().getElementsByClass(harmony.ChordSymbol):
                    first_chord_offset = el.offset
                    break
            else:
                for el in chords_part:
                    first_chord_offset = el.offset
                    break
            measure_start_offset = 0.0
            if first_chord_offset is not None:
                for meas in melody_part.getElementsByClass(stream.Measure):
                    meas_offset = self._to_float(meas.offset)
                    meas_dur = self._to_float(meas.duration.quarterLength)
                    if meas_offset <= self._to_float(first_chord_offset) < meas_offset + meas_dur:
                        measure_start_offset = meas_offset
                        break
            skip_steps = self._q_round(measure_start_offset, ql_per_quantum)
        total_duration_q = self._to_float(melody_part.highestTime)
        total_steps = int(np.ceil(total_duration_q / float(ql_per_quantum)))
        chord_tokens = [None] * total_steps
        chord_token_ids = [self.pad_token_id] * total_steps
        if chords_part is None:
            for el in melody_part.recurse().getElementsByClass(harmony.ChordSymbol):
                start = self._q_floor(el.offset, ql_per_quantum)
                if 0 <= start < len(chord_tokens):
                    chord_tokens[start], chord_token_ids[start] = self.handle_chord_symbol(el)
                if keep_durations:
                    end = self._q_ceil(self._to_float(el.offset) + self._to_float(el.duration.quarterLength), ql_per_quantum) + 1
                    if end < len(chord_tokens):
                        chord_tokens[end] = '<nc>'
                        chord_token_ids[end] = self.vocab['<nc>']
        else:
            for el in chords_part.recurse().getElementsByClass(chord.Chord):
                start = self._q_round(el.offset, ql_per_quantum)
                if 0 <= start < len(chord_tokens):
                    chord_tokens[start], chord_token_ids[start] = self.handle_chord_symbol(el)
                if keep_durations:
                    end = self._q_round(self._to_float(el.offset) + self._to_float(el.duration.quarterLength), ql_per_quantum) + 1
                    if end < len(chord_tokens):
                        chord_tokens[end] = '<nc>'
                        chord_token_ids[end] = self.vocab['<nc>']
        for i in range(1, len(chord_tokens)):
            if chord_tokens[i] is None:
                chord_tokens[i] = chord_tokens[i - 1]
                chord_token_ids[i] = chord_token_ids[i - 1]
        for i in range(len(chord_tokens)):
            if chord_tokens[i] is None:
                chord_tokens[i] = filler_token
                chord_token_ids[i] = self.vocab[filler_token]
        pitch_range = list(range(21, 109))
        raw_pianoroll = np.zeros((total_steps, len(pitch_range)), dtype=np.uint8)
        for el in melody_part.notesAndRests:
            start = self._q_floor(el.offset, ql_per_quantum)
            dur_steps = self._q_ceil(el.quarterLength, ql_per_quantum)
            if isinstance(el, note.Note):
                midi = el.pitch.midi
                if midi in pitch_range:
                    idx = pitch_range.index(midi)
                    raw_pianoroll[start:start + dur_steps, idx] = 1
            elif isinstance(el, chord.Chord):
                for p in el.pitches:
                    midi = p.midi
                    if midi in pitch_range:
                        idx = pitch_range.index(midi)
                        raw_pianoroll[start:start + dur_steps, idx] = 1
        n_steps = raw_pianoroll.shape[0]
        if self.use_pc_roll:
            pitch_classes = np.zeros((n_steps, 12), dtype=np.uint8)
            for i in range(n_steps):
                pitch_indices = np.where(raw_pianoroll[i] > 0)[0]
                for idx in pitch_indices:
                    midi = pitch_range[idx]
                    pitch_classes[i, midi % 12] = 1
            if self.use_full_range_melody:
                full_pianoroll = np.hstack([pitch_classes, raw_pianoroll])
            else:
                full_pianoroll = pitch_classes
        else:
            full_pianoroll = raw_pianoroll
        if self.intertwine_bar_info:
            num_steps = full_pianoroll.shape[0]
            bar_column = np.zeros((num_steps, 1), dtype=np.float32)
            full_pianoroll = np.hstack([full_pianoroll, bar_column])
            insertion_indices = []
            for meas in score.parts[0].getElementsByClass(stream.Measure):
                bar_step = self._q_round(meas.offset, ql_per_quantum)
                if 0 <= bar_step <= full_pianoroll.shape[0]:
                    insertion_indices.append(bar_step)
            insertion_indices = sorted(set(insertion_indices))
            pianoroll_ext = []
            step = 0
            for i in range(full_pianoroll.shape[0]):
                if i in insertion_indices:
                    bar_row = np.zeros((1, full_pianoroll.shape[1]), dtype=np.float32)
                    bar_row[0, -1] = 1.0
                    pianoroll_ext.append(bar_row)
                if step < full_pianoroll.shape[0]:
                    pianoroll_ext.append(full_pianoroll[step:step + 1, :])
                    step += 1
            full_pianoroll = np.vstack(pianoroll_ext)
            chord_tokens_ext = []
            chord_token_ids_ext = []
            step = 0
            for i in range(len(chord_tokens)):
                if i in insertion_indices:
                    chord_tokens_ext.append(self.bar_token)
                    chord_token_ids_ext.append(self.bar_token_id)
                if step < len(chord_tokens):
                    chord_tokens_ext.append(chord_tokens[step])
                    chord_token_ids_ext.append(chord_token_ids[step])
                    step += 1
            chord_tokens = chord_tokens_ext
            chord_token_ids = chord_token_ids_ext
        if self.trim_start:
            chord_tokens = chord_tokens[skip_steps:]
            chord_token_ids = chord_token_ids[skip_steps:]
        n_steps = len(chord_tokens)
        if self.fixed_length is not None:
            if n_steps >= self.fixed_length:
                full_pianoroll = full_pianoroll[:self.fixed_length]
                chord_tokens = chord_tokens[:self.fixed_length]
                chord_token_ids = chord_token_ids[:self.fixed_length]
                attention_mask = [1] * self.fixed_length
            else:
                pad_len = self.fixed_length - n_steps
                pad_pr = np.zeros((pad_len, full_pianoroll.shape[1]), dtype=np.uint8)
                pad_ch = ['<pad>'] * pad_len
                pad_ch_ids = [self.vocab['<pad>']] * pad_len
                full_pianoroll = np.vstack([full_pianoroll, pad_pr])
                chord_tokens += pad_ch
                chord_token_ids += pad_ch_ids
                attention_mask = [1] * n_steps + [0] * pad_len
        else:
            attention_mask = [1] * n_steps
        dens_result = self.compute_harmonic_rhythm_density(chord_token_ids)
        if isinstance(dens_result, tuple):
            h_rhythm, r_cat = dens_result
        else:
            h_rhythm, r_cat = (dens_result, None)
        comp_result = self.compute_harmonic_complexity(chord_tokens)
        if isinstance(comp_result, tuple):
            h_complexity, c_cat = comp_result
        else:
            h_complexity, c_cat = (comp_result, None)
        default_cat = [0, 0, 0, 1]
        if not isinstance(r_cat, (list, tuple)) or len(r_cat) != 4:
            r_cat = default_cat
        if not isinstance(c_cat, (list, tuple)) or len(c_cat) != 4:
            c_cat = default_cat
        h_density_complexity = list(r_cat) + list(c_cat)
        return {'harmony_tokens': chord_tokens, 'harmony_ids': chord_token_ids, 'pianoroll': full_pianoroll, 'time_signature': ts_num_list + ts_den_list, 'attention_mask': attention_mask, 'skip_steps': skip_steps, 'ql_per_quantum': ql_per_quantum, 'back_interval': back_interval, 'harmonic_rhythm_density': h_rhythm, 'harmonic_complexity': h_complexity, 'h_density_complexity': h_density_complexity}

    def encode_MIDI(self, file_path, filler_token='<nc>', keep_durations=False, normalize_tonality=False):
        score = converter.parse(file_path)
        back_interval = None
        if normalize_tonality:
            original_key = detect_key(score)
            to_c_or_a_interval = get_transposition_interval(original_key)
            score = transpose_score(score, to_c_or_a_interval)
            back_interval = to_c_or_a_interval.reverse()
        time_signature = score.recurse().getElementsByClass(meter.TimeSignature).first()
        ts_num_list = [0] * 14
        ts_den_list = [0, 0]
        ts_num_list[int(min(max(time_signature.numerator - 2, 0), 13))] = 1
        ts_den_list[int(time_signature.denominator == 4)] = 1
        melody_part = score.parts[0].flatten()
        chords_part = None
        if len(score.parts) > 1:
            chords_part = score.parts[1].chordify().flatten()
        if self.quantization == '16th':
            ql_per_quantum = 1 / 4
        elif self.quantization == '8th':
            ql_per_quantum = 1 / 2
        elif self.quantization == '4th':
            ql_per_quantum = 1
        elif self.quantization == '32nd':
            ql_per_quantum = 1 / 8
        else:
            ql_per_quantum = 1 / 4
        first_chord_offset = None
        skip_steps = 0
        if self.trim_start:
            if chords_part is not None:
                for el in chords_part.recurse().getElementsByClass(chord.Chord):
                    first_chord_offset = el.offset
                    break
            measure_start_offset = 0.0
            if first_chord_offset is not None:
                for meas in melody_part.getElementsByClass(stream.Measure):
                    meas_offset = self._to_float(meas.offset)
                    meas_dur = self._to_float(meas.duration.quarterLength)
                    if meas_offset <= self._to_float(first_chord_offset) < meas_offset + meas_dur:
                        measure_start_offset = meas_offset
                        break
            skip_steps = self._q_round(measure_start_offset, ql_per_quantum)
        total_duration_q = self._to_float(melody_part.highestTime)
        total_steps = int(np.ceil(total_duration_q / float(ql_per_quantum)))
        chord_tokens = [None] * total_steps
        chord_token_ids = [self.pad_token_id] * total_steps
        if chords_part is not None:
            for el in chords_part.recurse().getElementsByClass(chord.Chord):
                start = self._q_floor(el.offset, ql_per_quantum)
                if 0 <= start < len(chord_tokens):
                    chord_tokens[start], chord_token_ids[start] = self.handle_chord_symbol(el)
                if keep_durations:
                    end = self._q_ceil(self._to_float(el.offset) + self._to_float(el.duration.quarterLength), ql_per_quantum)
                    if end < len(chord_tokens) and chord_tokens[end] is None:
                        chord_tokens[end] = '<nc>'
                        chord_token_ids[end] = self.vocab['<nc>']
            for i in range(1, len(chord_tokens)):
                if chord_tokens[i] is None:
                    chord_tokens[i] = chord_tokens[i - 1]
                    chord_token_ids[i] = chord_token_ids[i - 1]
            for i in range(len(chord_tokens)):
                if chord_tokens[i] is None:
                    chord_tokens[i] = filler_token
                    chord_token_ids[i] = self.vocab[filler_token]
        else:
            for i in range(len(chord_tokens)):
                chord_tokens[i] = filler_token
                chord_token_ids[i] = self.vocab[filler_token]
        pitch_range = list(range(21, 109))
        raw_pianoroll = np.zeros((total_steps, len(pitch_range)), dtype=np.uint8)
        for el in melody_part.notesAndRests:
            start = self._q_floor(el.offset, ql_per_quantum)
            dur_steps = self._q_ceil(el.quarterLength, ql_per_quantum)
            if isinstance(el, note.Note):
                midi = el.pitch.midi
                if midi in pitch_range:
                    idx = pitch_range.index(midi)
                    raw_pianoroll[start:start + dur_steps, idx] = 1
            elif isinstance(el, chord.Chord):
                for p in el.pitches:
                    midi = p.midi
                    if midi in pitch_range:
                        idx = pitch_range.index(midi)
                        raw_pianoroll[start:start + dur_steps, idx] = 1
        n_steps = len(raw_pianoroll)
        if self.use_pc_roll:
            pitch_classes = np.zeros((n_steps, 12), dtype=np.uint8)
            for i in range(n_steps):
                pitch_indices = np.where(raw_pianoroll[i] > 0)[0]
                for idx in pitch_indices:
                    midi = pitch_range[idx]
                    pitch_classes[i, midi % 12] = 1
            if self.use_full_range_melody:
                full_pianoroll = np.hstack([pitch_classes, raw_pianoroll])
            else:
                full_pianoroll = pitch_classes
        else:
            full_pianoroll = raw_pianoroll
        if self.intertwine_bar_info:
            num_steps = full_pianoroll.shape[0]
            bar_column = np.zeros((num_steps, 1), dtype=np.float32)
            full_pianoroll = np.hstack([full_pianoroll, bar_column])
            insertion_indices = []
            for meas in score.parts[0].getElementsByClass(stream.Measure):
                bar_step = self._q_round(meas.offset, ql_per_quantum)
                if 0 <= bar_step <= full_pianoroll.shape[0]:
                    insertion_indices.append(bar_step)
            insertion_indices = sorted(set(insertion_indices))
            pianoroll_ext = []
            step = 0
            for i in range(full_pianoroll.shape[0]):
                if i in insertion_indices:
                    bar_row = np.zeros((1, full_pianoroll.shape[1]), dtype=np.float32)
                    bar_row[0, -1] = 1.0
                    pianoroll_ext.append(bar_row)
                if step < full_pianoroll.shape[0]:
                    pianoroll_ext.append(full_pianoroll[step:step + 1, :])
                    step += 1
            full_pianoroll = np.vstack(pianoroll_ext)
            chord_tokens_ext = []
            chord_token_ids_ext = []
            step = 0
            for i in range(len(chord_tokens)):
                if i in insertion_indices:
                    chord_tokens_ext.append(self.bar_token)
                    chord_token_ids_ext.append(self.bar_token_id)
                if step < len(chord_tokens):
                    chord_tokens_ext.append(chord_tokens[step])
                    chord_token_ids_ext.append(chord_token_ids[step])
                    step += 1
            chord_tokens = chord_tokens_ext
            chord_token_ids = chord_token_ids_ext
        if self.trim_start:
            chord_tokens = chord_tokens[skip_steps:]
            chord_token_ids = chord_token_ids[skip_steps:]
        n_steps = len(chord_tokens)
        if self.fixed_length is not None:
            if n_steps >= self.fixed_length:
                full_pianoroll = full_pianoroll[:self.fixed_length]
                chord_tokens = chord_tokens[:self.fixed_length]
                chord_token_ids = chord_token_ids[:self.fixed_length]
                attention_mask = [1] * self.fixed_length
            else:
                pad_len = self.fixed_length - n_steps
                pad_pr = np.zeros((pad_len, full_pianoroll.shape[1]), dtype=np.uint8)
                pad_ch = ['<pad>'] * pad_len
                pad_ch_ids = [self.vocab['<pad>']] * pad_len
                full_pianoroll = np.vstack([full_pianoroll, pad_pr])
                chord_tokens += pad_ch
                chord_token_ids += pad_ch_ids
                attention_mask = [1] * n_steps + [0] * pad_len
        else:
            attention_mask = [1] * n_steps
        h_rhythm, r_cat = self.compute_harmonic_rhythm_density(chord_token_ids)
        h_complexity, c_cat = self.compute_harmonic_complexity(chord_tokens)
        return {'harmony_tokens': chord_tokens, 'harmony_ids': chord_token_ids, 'pianoroll': full_pianoroll, 'time_signature': ts_num_list + ts_den_list, 'attention_mask': attention_mask, 'skip_steps': skip_steps, 'melody_part': melody_part, 'ql_per_quantum': ql_per_quantum, 'back_interval': back_interval, 'harmonic_rhythm_density': h_rhythm, 'harmonic_complexity': h_complexity, 'h_density_complexity': r_cat + c_cat}

    def save_vocabulary(self, save_directory, filename_prefix=None):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        vocab_filename = 'vocab.json' if filename_prefix is None else f'{filename_prefix}-vocab.json'
        vocab_file = os.path.join(save_directory, vocab_filename)
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        return (vocab_file,)

    def save_pretrained(self, save_directory, **kwargs):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        vocab_file = os.path.join(save_directory, 'vocab.json')
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        config_file = os.path.join(save_directory, 'tokenizer_config.json')
        config = {'special_tokens': self.special_tokens, 'quantization': self.quantization, 'fixed_length': self.fixed_length, 'use_pc_roll': self.use_pc_roll, 'use_full_range_melody': self.use_full_range_melody, 'intertwine_bar_info': self.intertwine_bar_info, 'trim_start': self.trim_start}
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return (save_directory,)

    @classmethod
    def from_pretrained(cls, load_directory, **kwargs):
        vocab_file = os.path.join(load_directory, 'vocab.json')
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        config_file = os.path.join(load_directory, 'tokenizer_config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        special_tokens = config.get('special_tokens', {})
        quantization = config.get('quantization', '16th')
        fixed_length = config.get('fixed_length', None)
        use_pc_roll = config.get('use_pc_roll', True)
        use_full_range_melody = config.get('use_full_range_melody', True)
        intertwine_bar_info = config.get('intertwine_bar_info', True)
        trim_start = config.get('trim_start', False)
        return cls(quantization=quantization, fixed_length=fixed_length, vocab=vocab, special_tokens=special_tokens, use_pc_roll=use_pc_roll, use_full_range_melody=use_full_range_melody, intertwine_bar_info=intertwine_bar_info, trim_start=trim_start, **kwargs)
