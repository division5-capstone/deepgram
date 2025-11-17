#!/usr/bin/env python3
"""
Deepgram Master Testing Script

Runs multiple Deepgram models with multiple confidence thresholds,
collects transcripts, word-level CSVs, summaries, and comparison plots.

Tested models now include:
- nova-2
- nova-2-meeting
- whisper-large
- nova-3
- nova-2-video
"""

import os
import sys
import json
import argparse
from datetime import datetime
from deepgram import DeepgramClient
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_transcription(client, audio_path, model, diarize, utterances):
    with open(audio_path, "rb") as f:
        resp = client.listen.v1.media.transcribe_file(
            request=f.read(),
            model=model,
            diarize=diarize,
            utterances=utterances,
            smart_format=True
        )
    return resp


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_full_transcript(utterances, path):
    with open(path, "w", encoding="utf-8") as f:
        for utt in utterances:
            start = utt.get("start", 0)
            end = utt.get("end", 0)
            speaker = utt.get("speaker", "Unknown")
            text = utt.get("transcript", "").strip()
            f.write(f"[{start:.2f} - {end:.2f}] Speaker {speaker}: {text}\n")


def save_confidence_filtered(utterances, path, threshold):
    with open(path, "w", encoding="utf-8") as f:
        for utt in utterances:
            start = utt.get("start", 0)
            end = utt.get("end", 0)
            speaker = utt.get("speaker", "Unknown")
            words = utt.get("words", [])

            filtered = []
            for w in words:
                txt = w.get("word")
                conf = w.get("confidence")
                if conf is None:
                    filtered.append("-")
                elif conf >= threshold:
                    filtered.append(txt)
                else:
                    filtered.append("-")

            f.write(f"[{start:.2f} - {end:.2f}] Speaker {speaker}: {' '.join(filtered)}\n")


def explode_words_to_df(utterances, model, run_id):
    rows = []
    for utt in utterances:
        start = utt.get("start", 0)
        end = utt.get("end", 0)
        speaker = utt.get("speaker", "Unknown")

        words = utt.get("words", [])
        if words:
            for w in words:
                rows.append({
                    "model": model,
                    "run_id": run_id,
                    "speaker": speaker,
                    "utt_start": start,
                    "utt_end": end,
                    "word": w.get("word"),
                    "word_start": w.get("start"),
                    "word_end": w.get("end"),
                    "confidence": w.get("confidence")
                })
        else:
            rows.append({
                "model": model,
                "run_id": run_id,
                "speaker": speaker,
                "utt_start": start,
                "utt_end": end,
                "word": utt.get("transcript", ""),
                "word_start": None,
                "word_end": None,
                "confidence": None
            })

    return pd.DataFrame(rows)


def summarize_df(df, threshold):
    conf_vals = df["confidence"].dropna()

    avg_conf = conf_vals.mean() if not conf_vals.empty else None
    numeric_count = conf_vals.count()
    low_count = df[df["confidence"].notna() & (df["confidence"] < threshold)].shape[0]

    low_pct = (low_count / numeric_count * 100) if numeric_count > 0 else None

    return {
        "avg_confidence": float(avg_conf) if avg_conf is not None else None,
        "total_words": int(len(df)),
        "numeric_conf_words": int(numeric_count),
        "low_conf_count": int(low_count),
        "low_conf_percent": float(low_pct) if low_pct is not None else None
    }


def plot_metrics(summary_rows, out_folder):
    df = pd.DataFrame(summary_rows)

    if "avg_confidence" not in df.columns:
        print("No avg_confidence column — skipping plots.")
        return

    df = df.dropna(subset=["avg_confidence"])
    if df.empty:
        print("No valid rows for plotting.")
        return

    # Avg confidence
    plt.figure(figsize=(8, 4))
    plt.bar(df["label"], df["avg_confidence"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Confidence")
    plt.title("Average Word Confidence by Run")
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "avg_confidence.png"))
    plt.close()

    # Low confidence %
    plt.figure(figsize=(8, 4))
    plt.bar(df["label"], df["low_conf_percent"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Low Confidence %")
    plt.title("Percent Low-Confidence Words")
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "low_conf_pct.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Deepgram Master Test Script")
    parser.add_argument("--audio", required=True, help="Audio file path")
    parser.add_argument("--models", nargs="+", default=[
        "nova-2", "nova-2-meeting", "whisper-large", "nova-3", "nova-2-video"
    ])
    parser.add_argument("--conf-thresholds", nargs="+", type=float, default=[0.8, 0.9])
    parser.add_argument("--diarize", action="store_true")
    parser.add_argument("--utterances", action="store_true")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    dg = DeepgramClient()

    run_base = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    comparison_rows = []

    for model in tqdm(args.models, desc="Models"):
        for thr in args.conf_thresholds:

            run_id = f"{run_base}_{model}_conf{int(thr*100)}"
            out_folder = os.path.join(args.outdir, run_id)
            ensure_dir(out_folder)

            print(f"\n--- Running {model} @ threshold {thr} ---")

            try:
                resp = run_transcription(dg, args.audio, model, args.diarize, args.utterances)
                resp_dict = json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", None)))

                save_json(resp_dict, os.path.join(out_folder, "response.json"))

                utterances = resp_dict.get("results", {}).get("utterances", [])

                save_full_transcript(utterances, os.path.join(out_folder, "transcript_full.txt"))
                save_confidence_filtered(utterances, os.path.join(out_folder, f"transcript_conf{int(thr*100)}.txt"), thr)

                df_words = explode_words_to_df(utterances, model, run_id)
                df_words.to_csv(os.path.join(out_folder, "words.csv"), index=False)

                summary = summarize_df(df_words, thr)
                summary.update({
                    "model": model,
                    "threshold": thr,
                    "run_id": run_id,
                    "label": run_id
                })
                comparison_rows.append(summary)
                save_json(summary, os.path.join(out_folder, "summary.json"))

            except Exception as e:
                print(f"Error running model {model}: {e}")
                comparison_rows.append({
                    "model": model,
                    "threshold": thr,
                    "run_id": run_id,
                    "label": run_id,
                    "error": str(e)
                })

    comp_df = pd.DataFrame(comparison_rows)
    csv_path = os.path.join(args.outdir, f"comparison_{run_base}.csv")
    comp_df.to_csv(csv_path, index=False)

    plot_metrics(comparison_rows, args.outdir)

    print("\n=== DONE ===")
    print(f"All outputs saved to: {os.path.abspath(args.outdir)}")
    print(f"Comparison CSV: {os.path.abspath(csv_path)}")


if __name__ == "__main__":
    main()