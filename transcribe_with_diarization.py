# python diarization.py data/bwc_4.mp4 data/whisper_segments.json 2

import os
import sys
import time
import json

# add local ffmpeg folder to PATH if present
ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg")
if os.path.isdir(ffmpeg_path):
    os.environ["PATH"] += os.pathsep + ffmpeg_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_with_diarization.py <audio_path> [segments.json] [n_speakers]")
        sys.exit(1)

    start_total = time.time()

    audio_path = sys.argv[1]
    segments_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].isdigit() else "data/whisper_segments.json"
    # allow calling: <audio> <n_speakers>
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        n_speakers = int(sys.argv[2])
    elif len(sys.argv) > 3 and sys.argv[3].isdigit():
        n_speakers = int(sys.argv[3])
    else:
        n_speakers = None

    # Transcribe with Whisper if segments file missing
    if not os.path.exists(segments_path):
        try:
            import whisper
        except Exception as exc:
            print("Whisper is not installed. Install with: python -m pip install openai-whisper")
            raise

        print(f"Transcribing {audio_path} with Whisper (this may take a while)...")
        start_time = time.time()
        model = whisper.load_model("base")
        res = model.transcribe(audio_path, task="transcribe", verbose=False)
        duration = time.time() - start_time
        segments = res.get("segments", [])
        os.makedirs(os.path.dirname(segments_path) or ".", exist_ok=True)
        with open(segments_path, "w", encoding="utf-8") as fh:
            json.dump(segments, fh, indent=2)
        print(f"Transcription saved to {segments_path} ({len(segments)} segments, took {duration:.1f}s)")
    else:
        with open(segments_path, "r", encoding="utf-8") as fh:
            segments = json.load(fh)
        print(f"Loaded {len(segments)} segments from {segments_path}")

    # Run diarization using the project's diarization helper
    try:
        from diarization import diarize_segments
    except Exception:
        print("Could not import diarization.diarize_segments. Ensure diarization.py exists and imports cleanly.")
        raise
    print("Running diarization...")
    diar_start = time.time()
    out = diarize_segments(audio_path, segments, n_speakers=n_speakers)
    diar_end = time.time()
    diar_dur = diar_end - diar_start

    # print speaker-labeled transcript
    print("\nSpeaker-labeled transcript:")
    for seg in out:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        speaker = seg.get("speaker", "Speaker 1")
        text = seg.get("text", "").strip()
        print(f"[{start:.2f}-{end:.2f}] {speaker}: {text}")

    total_end = time.time()
    total_dur = total_end - start_total

    # also write a combined output file with timing metadata
    out_path = "data/whisper_segments_speakers.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_blob = {
        "segments": out,
        "meta": {
            "total_took_seconds": round(total_dur, 3),
        },
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out_blob, fh, indent=2)
    print(f"\nSaved speaker-labeled segments + metadata to {out_path}")
    print(f"Total pipeline took {total_dur:.2f}s")


if __name__ == "__main__":
    main()
