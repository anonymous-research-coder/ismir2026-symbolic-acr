import os
from pathlib import Path
from copy import deepcopy

import mir_eval
import numpy as np
import torch
from music21 import chord, duration, harmony, key, meter, metadata, note, stream, tempo

from models import SEModular
from music_utils import transpose_score

models_dict = {
    "SE": SEModular,
    "SE_v0": SEModular,
}


def remove_conflicting_rests(flat_part):
    cleaned = stream.Part()
    all_notes = [el for el in flat_part if isinstance(el, note.Note)]
    note_offsets = [n.offset for n in all_notes]

    for i, n in enumerate(all_notes):
        if n.duration.quarterLength == 0:
            if i < len(all_notes) - 1:
                n.duration = duration.Duration(note_offsets[i + 1] - note_offsets[i])
            else:
                n.duration = duration.Duration(0.5)

    for el in flat_part:
        if isinstance(el, note.Rest) and el.offset in note_offsets:
            continue
        cleaned.insert(el.offset, el)

    return cleaned


def random_progressive_generate(
    model,
    melody_grid,
    conditioning_vec,
    num_stages,
    mask_token_id,
    temperature=1.0,
    strategy="topk",
    token_strategy="argmax",
    nucleus_p=0.9,
    pad_token_id=None,
    nc_token_id=None,
    force_fill=True,
    chord_constraints=None,
):
    device = melody_grid.device
    seq_len = melody_grid.shape[1]
    tokens_per_stage = int(seq_len / num_stages + 0.5)

    visible_harmony = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)

    if chord_constraints is not None:
        idxs = torch.logical_and(chord_constraints != nc_token_id, chord_constraints != pad_token_id)
        visible_harmony[idxs] = chord_constraints[idxs]

    if force_fill:
        active = (melody_grid != 0).any(dim=-1).squeeze(0)
        try:
            last_active_index = active.nonzero(as_tuple=True)[0].max().item()
        except Exception:
            last_active_index = -1
    else:
        last_active_index = -1

    while (visible_harmony == mask_token_id).any():
        with torch.no_grad():
            stage = int((visible_harmony == mask_token_id).sum().item() / visible_harmony.numel() * num_stages)
            stage = max(min(stage, num_stages - 1), 0)

            if conditioning_vec is None:
                logits = model(
                    melody_grid=melody_grid.to(model.device),
                    harmony_tokens=visible_harmony.to(model.device),
                    stage_indices=torch.LongTensor([stage]).to(model.device),
                )
            else:
                logits = model(
                    melody_grid=melody_grid.to(model.device),
                    conditioning_vec=conditioning_vec.to(model.device),
                    harmony_tokens=visible_harmony.to(model.device),
                    stage_indices=torch.LongTensor([stage]).to(model.device),
                )

        if force_fill and pad_token_id is not None and nc_token_id is not None:
            for i in range(seq_len):
                if i <= last_active_index:
                    logits[0, i, pad_token_id] = float("-inf")
                    logits[0, i, nc_token_id] = float("-inf")
                else:
                    logits[0, i, :] = float("-inf")
                    logits[0, i, pad_token_id] = 1.0

        probs = torch.softmax(logits / temperature, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)

        masked_positions = (visible_harmony == mask_token_id).squeeze(0).nonzero(as_tuple=True)[0]
        if masked_positions.numel() == 0:
            break

        topk = min(tokens_per_stage, masked_positions.size(0))

        if strategy == "topk":
            masked_confidences = confidences[0, masked_positions]
            topk_indices = torch.topk(masked_confidences, k=topk).indices
            selected_positions = masked_positions[topk_indices.to(device)]
        elif strategy == "sample":
            masked_confidences = confidences[0, masked_positions]
            p = masked_confidences / masked_confidences.sum()
            selected_positions = masked_positions[torch.multinomial(p, topk, replacement=False).to(device)]
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        for idx in selected_positions:
            if token_strategy == "argmax":
                visible_harmony[0, idx] = predictions[0, idx]
            elif token_strategy == "nucleus":
                probs_pos = probs[0, idx]
                sorted_probs, sorted_idx = torch.sort(probs_pos, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                nucleus_mask = cumulative <= nucleus_p
                nucleus_mask[0] = True
                nucleus_probs = sorted_probs[nucleus_mask]
                nucleus_idx = sorted_idx[nucleus_mask]
                nucleus_probs = nucleus_probs / nucleus_probs.sum()
                sampled = torch.multinomial(nucleus_probs, 1).item()
                visible_harmony[0, idx] = nucleus_idx[sampled].item()
            else:
                raise ValueError(f"Unsupported token_strategy: {token_strategy}")

    return visible_harmony


def greedy_token_by_token_generate(
    model,
    melody_grid,
    conditioning_vec,
    num_stages,
    mask_token_id,
    bar_token_id,
    temperature=1.0,
    pad_token_id=None,
    nc_token_id=None,
    force_fill=True,
    chord_constraints=None,
    max_steps=None,
):
    device = melody_grid.device
    seq_len = melody_grid.shape[1]

    visible_harmony = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)

    if chord_constraints is not None:
        idxs = torch.logical_and(chord_constraints != nc_token_id, chord_constraints != pad_token_id)
        visible_harmony[idxs] = chord_constraints[idxs]

    if force_fill:
        active = (melody_grid != 0).any(dim=-1).squeeze(0)
        try:
            last_active_index = active.nonzero(as_tuple=True)[0].max().item()
        except Exception:
            last_active_index = -1
    else:
        last_active_index = -1

    prev_logits = None
    avg_diffs = []
    total_tokens = visible_harmony.numel()
    step = 0

    while (visible_harmony == mask_token_id).any():
        if max_steps is not None and step >= max_steps:
            break

        num_masked = (visible_harmony == mask_token_id).sum().item()
        num_unmasked = total_tokens - num_masked
        s = max(round(num_unmasked / total_tokens * num_stages) - 1, 0)

        with torch.no_grad():
            logits = model(
                melody_grid=melody_grid.to(model.device),
                conditioning_vec=conditioning_vec.to(model.device),
                harmony_tokens=visible_harmony.to(model.device),
                stage_indices=torch.LongTensor([s]).to(model.device),
            )

        if force_fill and pad_token_id is not None and nc_token_id is not None:
            for i in range(seq_len):
                if i <= last_active_index:
                    logits[0, i, pad_token_id] = float("-inf")
                    logits[0, i, nc_token_id] = float("-inf")
                else:
                    logits[0, i, :] = float("-inf")
                    logits[0, i, pad_token_id] = 1.0

        probs = torch.softmax(logits / temperature, dim=-1)
        confidences, _ = torch.max(probs, dim=-1)

        if prev_logits is not None:
            masked_positions = (visible_harmony == mask_token_id).squeeze(0).nonzero(as_tuple=True)[0]
            if masked_positions.numel() > 0:
                prev_p = torch.softmax(prev_logits[0, masked_positions] / temperature, dim=-1)
                curr_p = torch.softmax(logits[0, masked_positions] / temperature, dim=-1)
                mad = torch.mean(torch.abs(prev_p - curr_p)).item()
                avg_diffs.append(mad)

        prev_logits = logits.clone()

        masked_positions = (visible_harmony == mask_token_id).squeeze(0).nonzero(as_tuple=True)[0]
        if masked_positions.numel() == 0:
            break

        masked_logits = logits[0, masked_positions] / temperature
        masked_probs = torch.softmax(masked_logits, dim=-1)
        masked_confidences, _ = torch.max(masked_probs, dim=-1)

        best_pos_idx = torch.argmax(masked_confidences).item()
        pos = masked_positions[best_pos_idx].item()

        k = min(10, masked_logits.size(-1))
        topk_logits, topk_indices = torch.topk(masked_logits[best_pos_idx], k)
        topk_probs = torch.softmax(topk_logits, dim=-1)
        sampled_idx = torch.multinomial(topk_probs, 1).item()
        sampled_token = topk_indices[sampled_idx].item()

        visible_harmony[0, pos] = sampled_token
        step += 1

    return visible_harmony, avg_diffs


def beam_token_by_token_generate(
    model,
    melody_grid,
    mask_token_id,
    bar_token_id,
    temperature=1.0,
    pad_token_id=None,
    nc_token_id=None,
    force_fill=True,
    chord_constraints=None,
    beam_size=5,
    top_k=5,
    unmasking_order="random",
):
    device = melody_grid.device
    seq_len = melody_grid.shape[1]

    init_visible_harmony = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)

    if chord_constraints is not None:
        idxs = torch.logical_and(chord_constraints != nc_token_id, chord_constraints != pad_token_id)
        init_visible_harmony[idxs] = chord_constraints[idxs]

    if force_fill:
        active = (melody_grid != 0).any(dim=-1).squeeze(0)
        try:
            last_active_index = active.nonzero(as_tuple=True)[0].max().item()
        except Exception:
            last_active_index = -1
    else:
        last_active_index = -1

    beams = [(init_visible_harmony.clone(), 0.0, [], None)]

    while any((beam[0] == mask_token_id).any() for beam in beams):
        candidates = []

        for visible_harmony, score, avg_diffs, prev_logits in beams:
            with torch.no_grad():
                logits = model(
                    melody_grid=melody_grid.to(model.device),
                    harmony_tokens=visible_harmony.to(model.device),
                )

            masked_positions = (visible_harmony == mask_token_id).squeeze(0).nonzero(as_tuple=True)[0]

            if masked_positions.numel() == 0:
                candidates.append((visible_harmony, score, avg_diffs, logits.clone()))
                continue

            probs = torch.softmax(logits[0, masked_positions] / temperature, dim=-1)
            entropies = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)

            if unmasking_order == "random":
                pos = masked_positions[torch.randint(0, masked_positions.numel(), (1,))].item()
            elif unmasking_order == "uncertain":
                pos = masked_positions[torch.argmax(entropies)].item()
            elif unmasking_order == "certain":
                pos = masked_positions[torch.argmin(entropies)].item()
            elif unmasking_order == "start":
                pos = masked_positions[0].item()
            elif unmasking_order == "end":
                pos = masked_positions[-1].item()
            else:
                pos = masked_positions[torch.randint(0, masked_positions.numel(), (1,))].item()

            if force_fill and pad_token_id is not None and nc_token_id is not None:
                for i in range(seq_len):
                    if i <= last_active_index:
                        logits[0, i, pad_token_id] = float("-inf")
                        logits[0, i, nc_token_id] = float("-inf")
                    else:
                        logits[0, i, :] = float("-inf")
                        logits[0, i, pad_token_id] = 1.0

            if prev_logits is not None:
                masked_positions_now = (visible_harmony == mask_token_id).squeeze(0).nonzero(as_tuple=True)[0]
                if masked_positions_now.numel() > 0:
                    prev_p = torch.softmax(prev_logits[0, masked_positions_now] / temperature, dim=-1)
                    curr_p = torch.softmax(logits[0, masked_positions_now] / temperature, dim=-1)
                    mad = torch.mean(torch.abs(prev_p - curr_p)).item()
                    avg_diffs = avg_diffs + [mad]

            masked_logits = logits[0, pos] / temperature
            topk_logits, topk_indices = torch.topk(masked_logits, min(top_k, masked_logits.size(-1)))
            topk_probs = torch.softmax(topk_logits, dim=-1)

            for j in range(topk_indices.size(0)):
                token = topk_indices[j].item()
                token_prob = topk_probs[j].item()
                new_harmony = visible_harmony.clone()
                new_harmony[0, pos] = token
                new_score = score + torch.log(torch.tensor(token_prob + 1e-12)).item()
                candidates.append((new_harmony, new_score, avg_diffs, logits.clone()))

        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

    best_harmony, _, best_avg_diffs, _ = beams[0]
    return best_harmony, best_avg_diffs


def nucleus_token_by_token_generate(
    model,
    melody_grid,
    mask_token_id,
    temperature=1.0,
    pad_token_id=None,
    nc_token_id=None,
    force_fill=True,
    chord_constraints=None,
    p=0.9,
    unmasking_order="random",
    num_stages=None,
    conditioning_vec=None,
):
    device = melody_grid.device
    seq_len = melody_grid.shape[1]

    visible_harmony = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)

    if chord_constraints is not None:
        idxs = torch.logical_and(chord_constraints != nc_token_id, chord_constraints != pad_token_id)
        visible_harmony[idxs] = chord_constraints[idxs]

    if force_fill:
        active = (melody_grid != 0).any(dim=-1).squeeze(0)
        try:
            last_active_index = active.nonzero(as_tuple=True)[0].max().item()
        except Exception:
            last_active_index = -1
    else:
        last_active_index = -1

    while (visible_harmony == mask_token_id).any():
        with torch.no_grad():
            if num_stages is None:
                stage = 0
            else:
                stage = int((visible_harmony == mask_token_id).sum().item() / visible_harmony.numel() * num_stages)
                stage = max(min(stage, num_stages - 1), 0)

            if conditioning_vec is None:
                logits = model(
                    melody_grid=melody_grid.to(model.device),
                    harmony_tokens=visible_harmony.to(model.device),
                    stage_indices=torch.LongTensor([stage]).to(model.device),
                )
            else:
                logits = model(
                    melody_grid=melody_grid.to(model.device),
                    conditioning_vec=conditioning_vec.to(model.device),
                    harmony_tokens=visible_harmony.to(model.device),
                    stage_indices=torch.LongTensor([stage]).to(model.device),
                )

        masked_positions = (visible_harmony == mask_token_id).squeeze(0).nonzero(as_tuple=True)[0]
        if masked_positions.numel() == 0:
            break

        probs = torch.softmax(logits[0, masked_positions] / temperature, dim=-1)
        entropies = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)

        if unmasking_order == "random":
            pos = masked_positions[torch.randint(0, masked_positions.numel(), (1,))].item()
        elif unmasking_order == "uncertain":
            pos = masked_positions[torch.argmax(entropies)].item()
        elif unmasking_order == "certain":
            pos = masked_positions[torch.argmin(entropies)].item()
        elif unmasking_order == "start":
            pos = masked_positions[0].item()
        elif unmasking_order == "end":
            pos = masked_positions[-1].item()
        else:
            pos = masked_positions[torch.randint(0, masked_positions.numel(), (1,))].item()

        if force_fill and pad_token_id is not None and nc_token_id is not None:
            for i in range(seq_len):
                if i <= last_active_index:
                    logits[0, i, pad_token_id] = float("-inf")
                    logits[0, i, nc_token_id] = float("-inf")
                else:
                    logits[0, i, :] = float("-inf")
                    logits[0, i, pad_token_id] = 1.0

        logits_pos = logits[0, pos] / temperature
        logits_pos[mask_token_id] = logits_pos.min().item() / 100
        probs_pos = torch.softmax(logits_pos, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs_pos, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus_mask = cumulative_probs <= p
        nucleus_mask[0] = True

        nucleus_probs = sorted_probs[nucleus_mask]
        nucleus_idx = sorted_idx[nucleus_mask]
        nucleus_probs = nucleus_probs / nucleus_probs.sum()

        sampled_idx = torch.multinomial(nucleus_probs, 1).item()
        token = nucleus_idx[sampled_idx].item()
        visible_harmony[0, pos] = token

    return visible_harmony


def structured_progressive_generate(
    model,
    melody_grid,
    conditioning_vec,
    num_stages,
    mask_token_id,
    temperature=1.0,
    strategy="topk",
    nucleus_p=0.9,
    pad_token_id=None,
    nc_token_id=None,
    force_fill=True,
    chord_constraints=None,
):
    device = melody_grid.device
    seq_len = melody_grid.shape[1]

    visible_harmony = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)

    if chord_constraints is not None:
        idxs = torch.logical_and(chord_constraints != nc_token_id, chord_constraints != pad_token_id)
        visible_harmony[idxs] = chord_constraints[idxs]

    if force_fill:
        active = (melody_grid != 0).any(dim=-1).squeeze(0)
        try:
            last_active_index = active.nonzero(as_tuple=True)[0].max().item()
        except Exception:
            last_active_index = -1
    else:
        last_active_index = -1

    computed_stages = int(np.ceil(np.log2(seq_len)))

    for stage in range(computed_stages):
        if not (visible_harmony == mask_token_id).any():
            break

        spacing_target = max(1, 2 ** (computed_stages - stage - 1))
        candidate_positions = torch.arange(0, seq_len, spacing_target, device=device)
        masked_positions = (visible_harmony[0] == mask_token_id).nonzero(as_tuple=True)[0]
        positions_to_predict = [pos for pos in candidate_positions if pos in masked_positions]

        if not positions_to_predict:
            continue

        with torch.no_grad():
            if conditioning_vec is None:
                logits = model(
                    melody_grid=melody_grid.to(model.device),
                    harmony_tokens=visible_harmony.to(model.device),
                    stage_indices=torch.LongTensor([stage]).to(model.device),
                )
            else:
                logits = model(
                    melody_grid=melody_grid.to(model.device),
                    conditioning_vec=conditioning_vec.to(model.device),
                    harmony_tokens=visible_harmony.to(model.device),
                    stage_indices=torch.LongTensor([stage]).to(model.device),
                )

        if force_fill and pad_token_id is not None and nc_token_id is not None:
            for i in range(seq_len):
                if i <= last_active_index:
                    logits[0, i, pad_token_id] = float("-inf")
                    logits[0, i, nc_token_id] = float("-inf")
                else:
                    logits[0, i, :] = float("-inf")
                    logits[0, i, pad_token_id] = 1.0

        probs = torch.softmax(logits / temperature, dim=-1)
        _, predictions = torch.max(probs, dim=-1)

        if strategy == "topk":
            for pos in positions_to_predict:
                visible_harmony[0, pos] = predictions[0, pos]
        elif strategy == "sample":
            for pos in positions_to_predict:
                prob_dist = probs[0, pos]
                sampled_token = torch.multinomial(prob_dist, num_samples=1)
                visible_harmony[0, pos] = sampled_token
        elif strategy == "nucleus":
            for pos in positions_to_predict:
                prob_dist = probs[0, pos]
                sorted_probs, sorted_idx = torch.sort(prob_dist, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                cutoff = cumulative_probs > nucleus_p
                if cutoff.any():
                    cutoff_index = cutoff.nonzero(as_tuple=True)[0][0].item()
                    sorted_probs = sorted_probs[: cutoff_index + 1]
                    sorted_idx = sorted_idx[: cutoff_index + 1]
                sorted_probs = sorted_probs / sorted_probs.sum()
                sampled_token = sorted_idx[torch.multinomial(sorted_probs, num_samples=1)]
                visible_harmony[0, pos] = sampled_token
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    return visible_harmony


def overlay_generated_harmony(melody_part, generated_chords, ql_per_16th, skip_steps):
    filtered_part = stream.Part()

    for el in melody_part.recurse().getElementsByClass(
        (key.KeySignature, meter.TimeSignature, tempo.MetronomeMark)
    ):
        if el.offset < 64:
            filtered_part.insert(el.offset, el)

    for el in melody_part.flatten().notesAndRests:
        if el.offset < 64:
            filtered_part.insert(el.offset, el)

    filtered_part = remove_conflicting_rests(filtered_part)
    melody_part = filtered_part

    for el in melody_part.recurse().getElementsByClass(harmony.ChordSymbol):
        melody_part.remove(el)

    melody_measures = melody_part.makeMeasures()
    chords_part = deepcopy(melody_measures)

    for measure in chords_part.getElementsByClass(stream.Measure):
        for el in list(measure):
            if isinstance(el, (note.Note, note.Rest, chord.Chord, harmony.ChordSymbol)):
                measure.remove(el)
        full_rest = note.Rest()
        full_rest.quarterLength = measure.barDuration.quarterLength
        measure.insert(0.0, full_rest)

    last_chord_symbol = None
    num_bar_tokens = 0

    for i, mir_chord in enumerate(generated_chords):
        if mir_chord in ("<pad>", "<nc>"):
            continue
        if mir_chord == last_chord_symbol:
            continue
        if mir_chord == "<bar>":
            num_bar_tokens += 1
            continue

        offset = (i + skip_steps - num_bar_tokens) * ql_per_16th

        try:
            r, t, _ = mir_eval.chord.encode(mir_chord, reduce_extended_chords=True)
            pcs = r + np.where(t > 0)[0] + 48
            c = chord.Chord(pcs.tolist())
        except Exception as e:
            print(f"Skipping invalid chord {mir_chord} at step {i}: {e}")
            continue

        bar = None
        bars = list(chords_part.getElementsByClass(stream.Measure))
        for b in reversed(bars):
            if b.offset <= offset:
                bar = b
                break

        if bar is None:
            continue

        bar_start = bar.offset
        bar_end = bar_start + bar.barDuration.quarterLength
        max_duration = bar_end - offset
        c.quarterLength = min(c.quarterLength, max_duration)

        for el in bar.getElementsByOffset(0.0):
            if isinstance(el, note.Rest):
                bar.remove(el)

        bar.insert(offset - bar_start, c)
        last_chord_symbol = mir_chord

    for m in chords_part.getElementsByClass(stream.Measure):
        bar_offset = m.offset
        bar_duration = m.barDuration.quarterLength
        has_chord = any(isinstance(el, chord.Chord) and el.offset == 0.0 for el in m)

        if not has_chord:
            prev_chords = []
            for curr_bar in chords_part.recurse().getElementsByClass(stream.Measure):
                for el in curr_bar.recurse().getElementsByClass(chord.Chord):
                    if curr_bar.offset + el.offset < bar_offset:
                        prev_chords.append(el)

            if prev_chords:
                for el in m.getElementsByOffset(0.0):
                    if isinstance(el, note.Rest):
                        m.remove(el)
                prev_chord = prev_chords[-1]
                m.insert(0.0, deepcopy(prev_chord))
        else:
            for el in m.getElementsByOffset(0.0):
                if isinstance(el, note.Rest):
                    m.remove(el)
            for el in m.notes:
                if isinstance(el, chord.Chord):
                    max_duration = bar_duration - el.offset
                    if el.quarterLength > max_duration:
                        el.quarterLength = max_duration

    score = stream.Score()
    score.insert(0, chords_part)
    return score


def save_harmonized_score(score, title="Harmonized Piece", out_path="harmonized.xml"):
    score.metadata = metadata.Metadata()
    score.metadata.title = title

    if out_path.endswith((".xml", ".mxl", ".musicxml")):
        score.write("musicxml", fp=out_path)
    elif out_path.endswith((".mid", ".midi")):
        score.write("midi", fp=out_path)
    else:
        print("unknown file format for file:", out_path)


def load_SE_Modular(
    d_model=512,
    nhead=8,
    num_layers=8,
    curriculum_type="f2f",
    subfolder=None,
    device_name="cuda:0",
    tokenizer=None,
    grid_length=80,
    nvis=None,
    condition_dim=None,
    unmasking_stages=None,
    trainable_pos_emb=False,
    version="SE",
    model_path=None,
):
    if device_name == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        print("Selected device not available:", device_name)
        device = torch.device("cpu")

    model = models_dict[version](
        chord_vocab_size=len(tokenizer.vocab),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        device=device,
        grid_length=grid_length,
        pianoroll_dim=tokenizer.pianoroll_dim,
        condition_dim=condition_dim,
        unmasking_stages=unmasking_stages,
        trainable_pos_emb=trainable_pos_emb,
    )

    if model_path is None:
        model_path = "saved_models/" + version + "/" + subfolder + "/" + curriculum_type
        if nvis is not None:
            model_path += "_nvis" + str(nvis)
        model_path += ".pt"

    print("model_path:", model_path)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model


def generate_files_with_base2(
    model,
    tokenizer,
    input_f,
    mxl_folder,
    midi_folder,
    name_suffix,
    use_constraints=False,
    intertwine_bar_info=False,
    normalize_tonality=False,
    temperature=1.0,
    p=0.9,
    unmasking_order="None",
    num_stages=10,
    use_conditions=False,
    create_gen=True,
    create_real=False,
):
    pad_token_id = tokenizer.pad_token_id
    nc_token_id = tokenizer.nc_token_id

    input_encoded = tokenizer.encode(
        input_f,
        keep_durations=True,
        normalize_tonality=normalize_tonality,
    )

    harmony_real = torch.LongTensor(input_encoded["harmony_ids"]).reshape(1, len(input_encoded["harmony_ids"]))
    harmony_input = torch.LongTensor(input_encoded["harmony_ids"]).reshape(1, len(input_encoded["harmony_ids"]))

    if intertwine_bar_info and not use_constraints:
        harmony_input[harmony_input != tokenizer.bar_token_id] = tokenizer.mask_token_id

    melody_grid = torch.FloatTensor(input_encoded["pianoroll"]).reshape(
        1,
        input_encoded["pianoroll"].shape[0],
        input_encoded["pianoroll"].shape[1],
    )

    if use_conditions:
        conditioning_vec = torch.FloatTensor(input_encoded["time_signature"]).reshape(
            1,
            len(input_encoded["time_signature"]),
        )
    else:
        conditioning_vec = None

    if create_gen:
        generated_harmony = structured_progressive_generate(
            model=model,
            melody_grid=melody_grid.to(model.device),
            conditioning_vec=None if conditioning_vec is None else conditioning_vec.to(model.device),
            num_stages=num_stages,
            mask_token_id=tokenizer.mask_token_id,
            temperature=temperature,
            strategy="nucleus",
            nucleus_p=p,
            pad_token_id=pad_token_id,
            nc_token_id=nc_token_id,
            force_fill=True,
            chord_constraints=harmony_input.to(model.device) if use_constraints or intertwine_bar_info else None,
        )
        gen_output_tokens = [tokenizer.ids_to_tokens[t] for t in generated_harmony[0].tolist()]
    else:
        gen_output_tokens = None

    harmony_real_tokens = [tokenizer.ids_to_tokens[t] for t in harmony_real[0].tolist()]

    gen_score = None
    real_score = None

    if create_gen:
        gen_score = overlay_generated_harmony(
            input_encoded["melody_part"],
            gen_output_tokens,
            input_encoded["ql_per_quantum"],
            input_encoded["skip_steps"],
        )
        if normalize_tonality:
            gen_score = transpose_score(gen_score, input_encoded["back_interval"])
        if mxl_folder is not None:
            save_harmonized_score(gen_score, out_path=os.path.join(mxl_folder, f"gen_{name_suffix}.mxl"))
        if midi_folder is not None:
            save_harmonized_score(gen_score, out_path=os.path.join(midi_folder, f"gen_{name_suffix}.mid"))

    if create_real:
        real_score = overlay_generated_harmony(
            input_encoded["melody_part"],
            harmony_real_tokens,
            input_encoded["ql_per_quantum"],
            input_encoded["skip_steps"],
        )
        if normalize_tonality:
            real_score = transpose_score(real_score, input_encoded["back_interval"])
        if mxl_folder is not None:
            save_harmonized_score(real_score, out_path=os.path.join(mxl_folder, f"real_{name_suffix}.mxl"))
        if midi_folder is not None:
            save_harmonized_score(real_score, out_path=os.path.join(midi_folder, f"real_{name_suffix}.mid"))

    return gen_output_tokens, harmony_real_tokens, gen_score, real_score


def generate_files_with_random(
    model,
    tokenizer,
    input_f,
    mxl_folder,
    midi_folder,
    name_suffix,
    use_constraints=False,
    intertwine_bar_info=False,
    normalize_tonality=False,
    temperature=1.0,
    p=0.9,
    unmasking_order="None",
    num_stages=10,
    use_conditions=False,
    create_gen=True,
    create_real=False,
):
    pad_token_id = tokenizer.pad_token_id
    nc_token_id = tokenizer.nc_token_id

    input_encoded = tokenizer.encode(
        input_f,
        keep_durations=True,
        normalize_tonality=normalize_tonality,
    )

    harmony_real = torch.LongTensor(input_encoded["harmony_ids"]).reshape(1, len(input_encoded["harmony_ids"]))
    harmony_input = torch.LongTensor(input_encoded["harmony_ids"]).reshape(1, len(input_encoded["harmony_ids"]))

    if intertwine_bar_info and not use_constraints:
        harmony_input[harmony_input != tokenizer.bar_token_id] = tokenizer.mask_token_id

    melody_grid = torch.FloatTensor(input_encoded["pianoroll"]).reshape(
        1,
        input_encoded["pianoroll"].shape[0],
        input_encoded["pianoroll"].shape[1],
    )

    if use_conditions:
        conditioning_vec = torch.FloatTensor(input_encoded["time_signature"]).reshape(
            1,
            len(input_encoded["time_signature"]),
        )
    else:
        conditioning_vec = None

    if create_gen:
        generated_harmony = random_progressive_generate(
            model=model,
            melody_grid=melody_grid.to(model.device),
            conditioning_vec=None if conditioning_vec is None else conditioning_vec.to(model.device),
            num_stages=num_stages,
            mask_token_id=tokenizer.mask_token_id,
            temperature=temperature,
            strategy="topk",
            token_strategy="nucleus",
            nucleus_p=p,
            pad_token_id=pad_token_id,
            nc_token_id=nc_token_id,
            force_fill=True,
            chord_constraints=harmony_input.to(model.device) if use_constraints or intertwine_bar_info else None,
        )
        gen_output_tokens = [tokenizer.ids_to_tokens[t] for t in generated_harmony[0].tolist()]
    else:
        gen_output_tokens = None

    harmony_real_tokens = [tokenizer.ids_to_tokens[t] for t in harmony_real[0].tolist()]

    gen_score = None
    real_score = None

    if create_gen:
        gen_score = overlay_generated_harmony(
            input_encoded["melody_part"],
            gen_output_tokens,
            input_encoded["ql_per_quantum"],
            input_encoded["skip_steps"],
        )
        if normalize_tonality:
            gen_score = transpose_score(gen_score, input_encoded["back_interval"])
        if mxl_folder is not None:
            save_harmonized_score(gen_score, out_path=os.path.join(mxl_folder, f"gen_{name_suffix}.mxl"))
        if midi_folder is not None:
            save_harmonized_score(gen_score, out_path=os.path.join(midi_folder, f"gen_{name_suffix}.mid"))

    if create_real:
        real_score = overlay_generated_harmony(
            input_encoded["melody_part"],
            harmony_real_tokens,
            input_encoded["ql_per_quantum"],
            input_encoded["skip_steps"],
        )
        if normalize_tonality:
            real_score = transpose_score(real_score, input_encoded["back_interval"])
        if mxl_folder is not None:
            save_harmonized_score(real_score, out_path=os.path.join(mxl_folder, f"real_{name_suffix}.mxl"))
        if midi_folder is not None:
            save_harmonized_score(real_score, out_path=os.path.join(midi_folder, f"real_{name_suffix}.mid"))

    return gen_output_tokens, harmony_real_tokens, gen_score, real_score


def generate_files_with_greedy(
    model,
    tokenizer,
    input_f,
    mxl_folder,
    midi_folder,
    name_suffix,
    use_constraints=False,
    condition="time_signature",
    force_condition=None,
    intertwine_bar_info=False,
    trim_start=True,
    normalize_tonality=False,
    num_stages=10,
    temperature=1.0,
):
    pad_token_id = tokenizer.pad_token_id
    nc_token_id = tokenizer.nc_token_id

    input_encoded = tokenizer.encode(
        input_f,
        keep_durations=True,
        normalize_tonality=normalize_tonality,
    )

    harmony_real = torch.LongTensor(input_encoded["harmony_ids"]).reshape(1, len(input_encoded["harmony_ids"]))
    harmony_input = torch.LongTensor(input_encoded["harmony_ids"]).reshape(1, len(input_encoded["harmony_ids"]))

    if intertwine_bar_info and not use_constraints:
        harmony_input[harmony_input != tokenizer.bar_token_id] = tokenizer.mask_token_id

    melody_grid = torch.FloatTensor(input_encoded["pianoroll"]).reshape(
        1,
        input_encoded["pianoroll"].shape[0],
        input_encoded["pianoroll"].shape[1],
    )

    conditioning_vec = torch.FloatTensor(input_encoded[condition]).reshape(1, len(input_encoded[condition]))
    if force_condition is not None:
        conditioning_vec = torch.FloatTensor(force_condition).reshape(1, len(force_condition))

    generated_harmony, avg_diffs = greedy_token_by_token_generate(
        model=model,
        melody_grid=melody_grid.to(model.device),
        conditioning_vec=conditioning_vec.to(model.device),
        num_stages=num_stages,
        mask_token_id=tokenizer.mask_token_id,
        bar_token_id=tokenizer.bar_token_id,
        temperature=temperature,
        pad_token_id=pad_token_id,
        nc_token_id=nc_token_id,
        force_fill=True,
        chord_constraints=harmony_input.to(model.device) if use_constraints or intertwine_bar_info else None,
    )

    gen_output_tokens = [tokenizer.ids_to_tokens[t] for t in generated_harmony[0].tolist()]
    harmony_real_tokens = [tokenizer.ids_to_tokens[t] for t in harmony_real[0].tolist()]

    gen_score = overlay_generated_harmony(
        input_encoded["melody_part"],
        gen_output_tokens,
        input_encoded["ql_per_quantum"],
        input_encoded["skip_steps"],
    )
    if normalize_tonality:
        gen_score = transpose_score(gen_score, input_encoded["back_interval"])
    save_harmonized_score(gen_score, out_path=os.path.join(mxl_folder, f"gen_{name_suffix}.mxl"))
    save_harmonized_score(gen_score, out_path=os.path.join(midi_folder, f"gen_{name_suffix}.mid"))

    real_score = overlay_generated_harmony(
        input_encoded["melody_part"],
        harmony_real_tokens,
        input_encoded["ql_per_quantum"],
        input_encoded["skip_steps"],
    )
    if normalize_tonality:
        real_score = transpose_score(real_score, input_encoded["back_interval"])
    save_harmonized_score(real_score, out_path=os.path.join(mxl_folder, f"real_{name_suffix}.mxl"))
    save_harmonized_score(real_score, out_path=os.path.join(midi_folder, f"real_{name_suffix}.mid"))

    return gen_output_tokens, harmony_real_tokens, gen_score, real_score, avg_diffs


def generate_files_with_beam(
    model,
    tokenizer,
    input_f,
    mxl_folder,
    midi_folder,
    name_suffix,
    use_constraints=False,
    intertwine_bar_info=False,
    normalize_tonality=False,
    temperature=1.0,
    beam_size=5,
    top_k=5,
    unmasking_order="random",
    create_gen=True,
    create_real=False,
):
    pad_token_id = tokenizer.pad_token_id
    nc_token_id = tokenizer.nc_token_id

    input_encoded = tokenizer.encode(
        input_f,
        keep_durations=True,
        normalize_tonality=normalize_tonality,
    )

    harmony_real = torch.LongTensor(input_encoded["harmony_ids"]).reshape(1, len(input_encoded["harmony_ids"]))
    harmony_input = torch.LongTensor(input_encoded["harmony_ids"]).reshape(1, len(input_encoded["harmony_ids"]))

    if intertwine_bar_info and not use_constraints:
        harmony_input[harmony_input != tokenizer.bar_token_id] = tokenizer.mask_token_id

    melody_grid = torch.FloatTensor(input_encoded["pianoroll"]).reshape(
        1,
        input_encoded["pianoroll"].shape[0],
        input_encoded["pianoroll"].shape[1],
    )

    if create_gen:
        generated_harmony, avg_diffs = beam_token_by_token_generate(
            model=model,
            melody_grid=melody_grid.to(model.device),
            mask_token_id=tokenizer.mask_token_id,
            bar_token_id=tokenizer.bar_token_id,
            temperature=temperature,
            pad_token_id=pad_token_id,
            nc_token_id=nc_token_id,
            force_fill=True,
            chord_constraints=harmony_input.to(model.device) if use_constraints or intertwine_bar_info else None,
            beam_size=beam_size,
            top_k=top_k,
            unmasking_order=unmasking_order,
        )
        gen_output_tokens = [tokenizer.ids_to_tokens[t] for t in generated_harmony[0].tolist()]
    else:
        avg_diffs = None
        gen_output_tokens = None

    harmony_real_tokens = [tokenizer.ids_to_tokens[t] for t in harmony_real[0].tolist()]

    gen_score = None
    real_score = None

    if create_gen:
        gen_score = overlay_generated_harmony(
            input_encoded["melody_part"],
            gen_output_tokens,
            input_encoded["ql_per_quantum"],
            input_encoded["skip_steps"],
        )
        if normalize_tonality:
            gen_score = transpose_score(gen_score, input_encoded["back_interval"])
        if mxl_folder is not None:
            save_harmonized_score(gen_score, out_path=os.path.join(mxl_folder, f"gen_{name_suffix}.mxl"))
        if midi_folder is not None:
            save_harmonized_score(gen_score, out_path=os.path.join(midi_folder, f"gen_{name_suffix}.mid"))

    if create_real:
        real_score = overlay_generated_harmony(
            input_encoded["melody_part"],
            harmony_real_tokens,
            input_encoded["ql_per_quantum"],
            input_encoded["skip_steps"],
        )
        if normalize_tonality:
            real_score = transpose_score(real_score, input_encoded["back_interval"])
        if mxl_folder is not None:
            save_harmonized_score(real_score, out_path=os.path.join(mxl_folder, f"real_{name_suffix}.mxl"))
        if midi_folder is not None:
            save_harmonized_score(real_score, out_path=os.path.join(midi_folder, f"real_{name_suffix}.mid"))

    return gen_output_tokens, harmony_real_tokens, gen_score, real_score, avg_diffs


def generate_files_with_nucleus(
    model,
    tokenizer,
    input_f,
    mxl_folder,
    midi_folder,
    name_suffix,
    use_constraints=False,
    intertwine_bar_info=False,
    normalize_tonality=False,
    temperature=1.0,
    p=0.9,
    unmasking_order="random",
    num_stages=None,
    use_conditions=False,
    create_gen=True,
    create_real=False,
):
    base_name = Path(input_f).stem
    pad_token_id = tokenizer.pad_token_id
    nc_token_id = tokenizer.nc_token_id

    input_encoded = tokenizer.encode(
        input_f,
        keep_durations=True,
        normalize_tonality=normalize_tonality,
    )

    harmony_real = torch.LongTensor(input_encoded["harmony_ids"]).reshape(1, len(input_encoded["harmony_ids"]))
    harmony_input = torch.LongTensor(input_encoded["harmony_ids"]).reshape(1, len(input_encoded["harmony_ids"]))

    if intertwine_bar_info and not use_constraints:
        harmony_input[harmony_input != tokenizer.bar_token_id] = tokenizer.mask_token_id

    melody_grid = torch.FloatTensor(input_encoded["pianoroll"]).reshape(
        1,
        input_encoded["pianoroll"].shape[0],
        input_encoded["pianoroll"].shape[1],
    )

    if use_conditions:
        conditioning_vec = torch.FloatTensor(input_encoded["time_signature"]).reshape(
            1,
            len(input_encoded["time_signature"]),
        )
    else:
        conditioning_vec = None

    if create_gen:
        generated_harmony = nucleus_token_by_token_generate(
            model=model,
            melody_grid=melody_grid.to(model.device),
            mask_token_id=tokenizer.mask_token_id,
            temperature=temperature,
            pad_token_id=pad_token_id,
            nc_token_id=nc_token_id,
            force_fill=True,
            chord_constraints=harmony_input.to(model.device) if use_constraints or intertwine_bar_info else None,
            p=p,
            unmasking_order=unmasking_order,
            num_stages=num_stages,
            conditioning_vec=None if conditioning_vec is None else conditioning_vec.to(model.device),
        )
        gen_output_tokens = [tokenizer.ids_to_tokens[t] for t in generated_harmony[0].tolist()]
    else:
        gen_output_tokens = None

    harmony_real_tokens = [tokenizer.ids_to_tokens[t] for t in harmony_real[0].tolist()]

    gen_score = None
    real_score = None

    if create_gen:
        gen_score = overlay_generated_harmony(
            input_encoded["melody_part"],
            gen_output_tokens,
            input_encoded["ql_per_quantum"],
            input_encoded["skip_steps"],
        )
        if normalize_tonality:
            gen_score = transpose_score(gen_score, input_encoded["back_interval"])
        if mxl_folder is not None:
            save_harmonized_score(gen_score, out_path=os.path.join(mxl_folder, base_name + ".mxl"))
        if midi_folder is not None:
            save_harmonized_score(gen_score, out_path=os.path.join(midi_folder, base_name + ".mid"))

    if create_real:
        real_score = overlay_generated_harmony(
            input_encoded["melody_part"],
            harmony_real_tokens,
            input_encoded["ql_per_quantum"],
            input_encoded["skip_steps"],
        )
        if normalize_tonality:
            real_score = transpose_score(real_score, input_encoded["back_interval"])
        if mxl_folder is not None:
            save_harmonized_score(real_score, out_path=os.path.join(mxl_folder, f"real_{name_suffix}.mxl"))
        if midi_folder is not None:
            save_harmonized_score(real_score, out_path=os.path.join(midi_folder, f"real_{name_suffix}.mid"))

    return gen_output_tokens, harmony_real_tokens, gen_score, real_score