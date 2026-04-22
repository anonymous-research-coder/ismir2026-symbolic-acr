from music21 import converter, key, pitch, interval, midi

def detect_key(score):
    return score.analyze('key')

def get_transposition_interval(k):
    if k.mode == 'major':
        return interval.Interval(k.tonic, pitch.Pitch('C'))
    elif k.mode == 'minor':
        return interval.Interval(k.tonic, pitch.Pitch('A'))
    else:
        return interval.Interval(0)

def transpose_score(score, transposition_interval):
    return score.transpose(transposition_interval)
