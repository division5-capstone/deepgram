import json
import os

from typing import List, Dict, Optional


def _require_deps():
    try:
        import numpy  # noqa: F401
        import librosa  # noqa: F401
        from sklearn.cluster import AgglomerativeClustering  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Missing dependencies for diarization. Install: librosa numpy scikit-learn"
        ) from exc


def diarize_segments(
    file_path: str,
    segments: List[Dict],
    n_speakers: Optional[int] = None,
    sr: int = 16000,
    n_mfcc: int = 13,
) -> List[Dict]:
    """
    Simple diarization by extracting MFCC features per segment and clustering.
    This avoids heavy native dependencies (no resemblyzer/webrtcvad).

    Args:
      file_path: path to audio file (readable by librosa).
      segments: list of dicts with numeric 'start' and 'end' (seconds).
      n_speakers: fixed number of speakers; if None defaults to 2.
      sr: sample rate to load audio.
      n_mfcc: number of MFCC coefficients to compute.

    Returns:
      segments with added 'speaker' key.
    """
    _require_deps()
    import numpy as np
    import librosa
    from sklearn.cluster import AgglomerativeClustering

    if not segments:
        return []

    # validate/normalize
    sanitized = []
    for s in segments:
        try:
            start = float(s["start"])
            end = float(s["end"])
        except Exception:
            raise ValueError("Each segment must include numeric 'start' and 'end' fields.")
        if end <= start:
            raise ValueError(f"Segment end must be > start (got start={start}, end={end}).")
        sanitized.append({"start": start, "end": end, **{k: v for k, v in s.items() if k not in ("start", "end")}})

    # load entire audio once
    audio, file_sr = librosa.load(file_path, sr=sr, mono=True)

    feats = []
    for seg in sanitized:
        start = max(0.0, seg["start"])
        end = seg["end"]
        s_idx = int(round(start * sr))
        e_idx = int(round(end * sr))
        seg_audio = audio[s_idx:e_idx]
        if seg_audio.size == 0:
            # pad very short/empty segments
            seg_audio = np.zeros(int(0.1 * sr), dtype=audio.dtype)
        # compute MFCC and take mean over time frames -> fixed-size vector
        mfcc = librosa.feature.mfcc(y=seg_audio, sr=sr, n_mfcc=n_mfcc)
        vec = np.mean(mfcc, axis=1)
        feats.append(vec)

    if len(feats) == 0:
        return [{"start": seg["start"], "end": seg["end"], "text": seg.get("text", ""), "speaker": "Speaker 1"} for seg in sanitized]

    X = np.vstack(feats)

    if n_speakers is None:
        n_speakers = 2
    n_speakers = max(1, min(n_speakers, X.shape[0]))

    if n_speakers == 1 or X.shape[0] == 1:
        labels = [0] * X.shape[0]
    else:
        clustering = AgglomerativeClustering(n_clusters=n_speakers)
        labels = clustering.fit_predict(X)

    cluster_to_speaker = {}
    speaker_counter = 1
    results = []
    for seg, label in zip(sanitized, labels):
        if label not in cluster_to_speaker:
            cluster_to_speaker[label] = f"Speaker {speaker_counter}"
            speaker_counter += 1
        out = {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg.get("text", ""),
            "speaker": cluster_to_speaker[label],
        }
        results.append(out)

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python diarization.py <audio_path> <segments.json> [n_speakers]")
        sys.exit(1)
    audio_path = sys.argv[1]
    segments_path = sys.argv[2]
    n_speakers = int(sys.argv[3]) if len(sys.argv) > 3 else None

    if not os.path.exists(segments_path):
        print(f"Segments file not found: {segments_path}\nPlease generate a segments.json (e.g. via Whisper) before running diarization.")
        sys.exit(1)

    with open(segments_path, "r", encoding="utf-8") as fh:
        segs = json.load(fh)
    out = diarize_segments(audio_path, segs, n_speakers=n_speakers)
    print(json.dumps(out, indent=2))