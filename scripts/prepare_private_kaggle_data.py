"""Build the PRIVATE Kaggle dataset used for the personal training + voice run.

What leaves this machine (only after you run this and upload it yourself):
  chats/        text-only chats, other people's names replaced by "Person A/B/...",
                phone numbers / emails / URLs redacted, file names anonymised
  voice/        your enrolment clips (WAV) + manifest.json with script-aligned text
  dataset-metadata.json   private Kaggle dataset descriptor
  voice_consent.json      written ONLY if voice consent is already granted in the studio

Usage:
  python scripts/prepare_private_kaggle_data.py [--out data/personal/kaggle]
  kaggle datasets create -p data/personal/kaggle --dir-mode zip
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from echotwin.privacy.redact import RedactionConfig, redact_text

LINE = re.compile(r"^(\[\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2}\]) ([^:]+): (.*)$")
DATASET_ID = "shashwat1729/echotwin-personal"


def anonymise_chat(path: Path, me: str, cfg: RedactionConfig) -> str:
    aliases: dict[str, str] = {}
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = LINE.match(raw)
        if not m:
            text, _ = redact_text(raw, cfg)
            out.append(text)
            continue
        stamp, sender, body = m.groups()
        if sender != me:
            aliases.setdefault(sender, f"Person {chr(65 + len(aliases) % 26)}{len(aliases) // 26 or ''}")
            sender = aliases[sender]
        body, _ = redact_text(body, cfg)
        out.append(f"{stamp} {sender}: {body}")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chats", type=Path, default=Path("data/whatsapp_export/chats"))
    ap.add_argument("--out", type=Path, default=Path("data/personal/kaggle"))
    ap.add_argument("--me", default="Me")
    args = ap.parse_args()

    cfg = RedactionConfig(enabled=True)
    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "chats").mkdir(parents=True)
    n = 0
    for n, src in enumerate(sorted(args.chats.glob("*_chat.txt")), 1):
        (args.out / "chats" / f"chat-{n:04d}_chat.txt").write_text(
            anonymise_chat(src, args.me, cfg), encoding="utf-8"
        )

    from echotwin.services import voice as voice_service

    profile = voice_service.current_profile()
    clips = []
    if profile:
        (args.out / "voice").mkdir()
        for r in profile.recordings:
            if r.quality_verdict == "fail":
                continue
            name = Path(r.path).name
            shutil.copy2(r.path, args.out / "voice" / name)
            clips.append(
                {"file": name, "seconds": r.duration_seconds, "language": r.language,
                 "transcript": r.transcript or "", "transcript_source": "script-aligned (unverified)"}
            )
        (args.out / "voice" / "manifest.json").write_text(
            json.dumps(clips, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    consent = voice_service.consent_status()
    if consent.granted:
        (args.out / "voice_consent.json").write_text(
            json.dumps({"granted": True, "speaker": args.me}), encoding="utf-8"
        )
    (args.out / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": "echotwin-personal", "id": DATASET_ID, "licenses": [{"name": "CC0-1.0"}],
             "isPrivate": True},
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"chats={n} voice_clips={len(clips)} consent_granted={consent.granted} -> {args.out}")


if __name__ == "__main__":
    main()
