"""Fetch a real, public, multi-conversation chat corpus and convert it to
WhatsApp-style export files that EchoTwin's own importer can read.

Why this exists
---------------
EchoTwin needs a *real* corpus to be measurable on: many conversations, and one
consistent author whose messages span all of them. That is exactly the premise
of the product (a person who chats with you across many conversations).

Source: DailyDialog (Liu et al., ACL 2017), mirrored as a ConvLab dataset
``ConvLab/dailydialog`` (13,118 dialogues, two speakers per dialogue, unique
``dialogue_id``). License: Creative Commons Attribution 4.0 (see the dataset
card). We use it as *unlabeled style data for evaluation only* - no private
data is ever involved.

Experimental design (stated honestly)
--------------------------------------
The second participant of each dialogue plays the role of "the author" and the
other plays "the partner". The author is a *role*, not a single human being:
the experiment therefore measures "can LoRA on a consistent speaker's messages,
across held-out conversations, improve style match and prompt responsiveness?".
It is a controlled proxy for the real single-person setting, not a claim about
any individual speaker.

Output
------
``<out_dir>/<conversation_id>/_chat.txt``   WhatsApp-style exports, one per
conversation, so the standard importer path is exercised end to end.
``<out_dir>/manifest.json``                  provenance, licenses, counts.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

LICENSES = {
    "dailydialog": {
        "source": "ConvLab/dailydialog (mirror of DailyDialog, ACL 2017)",
        "url": "https://huggingface.co/datasets/ConvLab/dailydialog",
        "license": "CC BY 4.0 (DailyDialog); verify before redistribution",
        "citation": (
            "Liu, P. et al. (2017). A Causal Model for Linguistic Acceptability "
            "Judgement. ICML. (source corpus of DailyDialog)"
        ),
    }
}

# A deterministic base timestamp so the converted corpus is reproducible.
BASE = datetime(2024, 1, 8, 9, 0, 0)


def _clean(text: str) -> str:
    """Normalize the corpus' whitespace artifacts without changing style."""
    text = text.replace("__eou__", " ").replace("__eot__", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def download_dialogs(cache_dir: Path) -> list[dict]:
    """Fetch the DailyDialog mirror and return the raw dialogue records."""
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        "ConvLab/dailydialog", "data.zip", repo_type="dataset", cache_dir=str(cache_dir)
    )
    with zipfile.ZipFile(path) as zf:
        raw = zf.read("data/dialogues.json").decode("utf-8")
    data = json.loads(raw)
    if isinstance(data, dict):
        data = list(data.values())
    return data


def convert(
    dialogues: list[dict],
    out_dir: Path,
    *,
    author_speaker: str = "user",
    partner_speaker: str = "system",
    max_conversations: int | None = None,
    min_turns: int = 6,
    seed: int = 20250917,
) -> dict:
    """Write one WhatsApp-style export per dialogue.

    The author is the ``author_speaker``; the partner is the other side. Turns
    are stamped with a plausible cadence (minutes apart) so the importer's
    time-gap segmentation behaves like it would on a real export.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written: list[dict] = []
    kept = 0
    for record in dialogues:
        if max_conversations is not None and kept >= max_conversations:
            break
        turns = record.get("turns") or []
        if len(turns) < min_turns:
            continue
        conv_id = str(record.get("dialogue_id") or f"dlg-{kept}")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", conv_id)[:60] or f"dlg-{kept}"
        conv_dir = out_dir / safe
        conv_dir.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        stamp = BASE
        n_author = 0
        for turn in turns:
            speaker = turn.get("speaker")
            text = _clean(str(turn.get("utterance", "")))
            if not text:
                continue
            if speaker == author_speaker:
                name = "Author"
                n_author += 1
            elif speaker == partner_speaker:
                name = "Partner"
            else:
                name = "Partner"
            stamp += timedelta(minutes=rng.randint(1, 40), seconds=rng.randint(0, 59))
            lines.append(f"[{stamp:%d/%m/%Y, %H:%M:%S}] {name}: {text}")
        if n_author < 3:
            continue
        (conv_dir / "_chat.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(
            {
                "conversation_id": safe,
                "turns": len(lines),
                "author_messages": n_author,
                "partner_messages": len(lines) - n_author,
            }
        )
        kept += 1
    return {
        "conversations": written,
        "n_conversations": len(written),
        "n_author_messages": sum(w["author_messages"] for w in written),
        "n_partner_messages": sum(w["partner_messages"] for w in written),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("sample_corpus/daily_dialog"))
    ap.add_argument("--max-conversations", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20250917)
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    print("[1/3] downloading DailyDialog mirror ...", flush=True)
    dialogues = download_dialogs(out / "_cache")
    print(f"      {len(dialogues)} raw dialogues", flush=True)

    print(f"[2/3] converting <= {args.max_conversations} conversations ...", flush=True)
    stats = convert(
        dialogues,
        out / "chats",
        max_conversations=args.max_conversations,
        seed=args.seed,
    )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus": "dailydialog",
        "design": (
            "author = second participant ('user' role) across all conversations; "
            "each dialogue is one conversation for leakage-resistant splitting"
        ),
        "licenses": LICENSES,
        "stats": {k: v for k, v in stats.items() if k != "conversations"},
        "conversations": stats["conversations"][:50],
        "note": (
            "Corpus text is third-party and licensed separately from EchoTwin's code. "
            "Do not redistribute the raw text without checking the license."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[3/3] wrote {stats['n_conversations']} conversations, "
        f"{stats['n_author_messages']} author messages -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
