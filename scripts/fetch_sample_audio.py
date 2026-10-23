"""Fetch small open-licensed speech samples for pipeline testing.

Sources (verified 2026-10-15; licenses recorded automatically from the Hub):
- EN: hf-internal-testing/librispeech_asr_dummy (LibriVox-derived, public domain)
- HI: Paytmlabs/S2R_Shrutilipi_hindi validation shard (license from dataset card)
- HI words: Lingua Libre via Wikimedia Commons (CC BY-SA 4.0, native speakers)

Output: sample_audio/<lang>/clip-<n>.wav + transcript.txt + manifest.json.
Everything is git-ignored; re-run this script to reproduce. No torch needed
(pyarrow + soundfile only). Real human speech — never synthetic.
"""

from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import dataset_info, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sample_audio"
UA = {"User-Agent": "EchoTwin/0.1.0 (research evaluation; local use)"}


def _safe(msg: str) -> str:
    return msg.encode("ascii", errors="replace").decode("ascii")


def save_clip(wav_dir: Path, stem: str, audio_bytes: bytes, transcript: str) -> dict:
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    out = wav_dir / f"{stem}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), data, sr, subtype="PCM_16")
    (wav_dir / f"{stem}.transcript.txt").write_text(transcript or "", encoding="utf-8")
    seconds = len(data) / sr if sr else 0.0
    return {"file": out.name, "sample_rate": sr, "seconds": round(seconds, 2),
            "transcript_chars": len(transcript or "")}


def parquet_clips(repo: str, parquet_path: str, lang: str, n: int, prefix: str) -> list[dict]:
    local = hf_hub_download(repo, parquet_path, repo_type="dataset")
    table = pq.read_table(local)
    try:
        info = dataset_info(repo)
        license_name = info.cardData.get("license", "see dataset page") if info.cardData else "see dataset page"
    except Exception:
        license_name = "see dataset page"
    clips = []
    taken = 0
    for row in table.to_pylist():
        if taken >= n:
            break
        blob = (row.get("audio") or {}).get("bytes")
        text = row.get("text") or ""
        if not blob or not str(text).strip():
            continue
        try:
            meta = save_clip(OUT / lang, f"{prefix}-{taken:02d}", blob, str(text).strip())
        except Exception as e:
            print(_safe(f"  skip unreadable row: {e}"))
            continue
        meta.update({"source_repo": repo, "license": license_name, "language": lang})
        clips.append(meta)
        taken += 1
    return clips


def commons_words(n: int) -> list[dict]:
    search_api = (
        "https://commons.wikimedia.org/w/api.php?action=query&format=json"
        "&list=search&srsearch=LL-Q1568%20hin&srnamespace=6&srlimit=30"
    )
    req = urllib.request.Request(search_api, headers=UA)
    data = json.load(urllib.request.urlopen(req, timeout=30))
    titles = [h["title"] for h in data.get("query", {}).get("search", [])]
    info_api = (
        "https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo"
        "&iiprop=url%7Csize%7Cmime&titles=" + urllib.parse.quote("|".join(titles[:12]))
    )
    req = urllib.request.Request(info_api, headers=UA)
    info = json.load(urllib.request.urlopen(req, timeout=30))
    clips = []
    for page in info.get("query", {}).get("pages", {}).values():
        if len(clips) >= n:
            break
        for ii in page.get("imageinfo", []):
            url = ii.get("url", "").split("?")[0]
            title = page.get("title", "")
            # spoken word is the last segment of the LL filename
            word = title.rsplit("-", 1)[-1].replace(".wav", "").strip() if "-" in title else ""
            try:
                req = urllib.request.Request(url, headers=UA)
                blob = urllib.request.urlopen(req, timeout=60).read()
                meta = save_clip(OUT / "hi-words", f"ll-{len(clips):02d}", blob, word)
            except Exception as e:
                print(_safe(f"  skip {title}: {e}"))
                continue
            meta.update({"source_url": url, "license": "CC BY-SA 4.0 (Lingua Libre via Wikimedia Commons)",
                         "language": "hi"})
            clips.append(meta)
    return clips


def main() -> None:
    manifest = {"clips": []}
    print("EN librispeech dummy ...")
    manifest["clips"] += parquet_clips(
        "hf-internal-testing/librispeech_asr_dummy", "clean/validation-00000-of-00001.parquet",
        "en", 3, "en-libri",
    )
    print("HI shrutilipi ...")
    manifest["clips"] += parquet_clips(
        "Paytmlabs/S2R_Shrutilipi_hindi", "data/validation/validation-00000-of-00001.parquet",
        "hi", 4, "hi-shruti",
    )
    print("HI lingua-libre words ...")
    manifest["clips"] += commons_words(8)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(manifest['clips'])} clips + manifest to {OUT}")


if __name__ == "__main__":
    main()
