"""Build a personalised single-take recording script from YOUR OWN sent messages.

Reading your own real sentences aloud captures the vocabulary, slang, Hinglish
mix, questions and exclamations you actually use, instead of generic text.

Selection is greedy word-coverage over your most frequent words, with:
  * PII filter  - no digits runs, emails, URLs, or any contact's first name
  * speakable   - emoji/symbol-heavy and keyboard-mash lines are dropped
  * variety     - a minimum number of questions, exclamations and short/long lines

Output (private, git-ignored): data/personal/master_script.txt
The voice studio picks it up automatically as the "master take" script.

Usage:
  python scripts/build_personal_script.py [--messages data/whatsapp_export/messages.jsonl]
        [--target-words 420] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

WORD = re.compile(r"[a-zऀ-ॿ']+")
EMOJI_OR_SYMBOL = re.compile(r"[^\w\s.,!?'\"\-:;()ऀ-ॿ]", re.UNICODE)
PII = re.compile(r"(\d{4,}|@|https?:|www\.|\.com\b|\.in\b)", re.I)
MASH = re.compile(r"(.)\1{3,}")
# Pasted prompts / formal text / structured notes are not how you speak.
FORMAL = re.compile(
    r"[:()\[\]\-/]|\b(please|kindly|regards|assignment|submit|paper|prompts?|gpu|model|api|dataset|"
    r"production|application|linkedin|github|google|drive|econometrics)\b",
    re.I,
)
# Family references stay out of a script that may be shared with a voice backend.
FAMILY = {"mumma", "mummy", "mom", "mama", "maa", "papa", "pappa", "dad", "dada", "dadi",
          "nana", "nani", "didi", "bhabhi", "chacha", "chachi", "mausi", "bua", "fufa", "mamu"}

# Light markers to tell Hinglish (Hindi in Latin letters) from plain English.
HI_MARK = {"hai", "hain", "kya", "nahi", "nhi", "ko", "ke", "ki", "ka", "se", "mein", "bhi", "toh", "to", "tha",
           "thi", "the", "hum", "tum", "aap", "main", "mai", "ye", "vo", "wo", "kar", "karo", "kara", "hamne",
           "tumhe", "tumhare", "mera", "meri", "mere", "abhi", "kal", "aaj", "ho", "hoga", "raha", "rahe",
           "rahi", "vaala", "vaali", "na", "kuch", "bahut", "achha", "theek", "phir", "par", "aur", "lo", "do"}
EN_STOP = {"the", "is", "you", "to", "and", "of", "in", "it", "that", "this", "for", "with", "are", "we", "can",
           "will", "have", "not", "if", "be", "on", "at", "your", "my", "so", "but", "was", "what", "how",
           "when", "do", "does", "did", "just", "all", "me", "there", "any", "i", "a", "an", "as", "or"}


def sentences(text: str) -> list[str]:
    out = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        part = part.strip()
        if part:
            out.append(part)
    return out


def clean(s: str) -> str:
    s = EMOJI_OR_SYMBOL.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def contact_first_names(tsv: Path) -> set[str]:
    names: set[str] = set()
    if tsv.exists():
        for line in tsv.read_text(encoding="utf-8").splitlines():
            name = line.split("\t", 1)[0]
            for tok in WORD.findall(name.lower()):
                if len(tok) >= 3:
                    names.add(tok)
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--messages", type=Path, default=Path("data/whatsapp_export/messages.jsonl"))
    ap.add_argument("--contacts", type=Path, default=Path("data/whatsapp_export/raw/contacts.tsv"))
    ap.add_argument("--out", type=Path, default=Path("data/personal/master_script.txt"))
    ap.add_argument("--target-words", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exclude", default="", help="comma-separated words; any sentence containing one is dropped")
    args = ap.parse_args()

    blocked = contact_first_names(args.contacts) | FAMILY | {w.strip().lower() for w in args.exclude.split(",") if w.strip()}
    # Pre-pass: how often each word is written in lowercase. A capitalised word in
    # mid-sentence that is rarely written lowercase is almost surely a proper name.
    lower_seen: Counter[str] = Counter()
    with args.messages.open(encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            if m["from_me"]:
                lower_seen.update(re.findall(r"(?<![A-Za-z])[a-z']+", m["text"]))

    def has_name(s: str) -> bool:
        return any(
            w[:1].isupper() and w not in {"I", "I'm", "I'll"} and lower_seen[w.lower().strip(".,!?")] < 15
            for w in s.split()[1:]
        )

    freq: Counter[str] = Counter()
    cands: dict[str, int] = {}
    with args.messages.open(encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            if not m["from_me"]:
                continue
            for s in sentences(m["text"]):
                if PII.search(s) or MASH.search(s) or FORMAL.search(s):
                    continue
                s = clean(s)
                toks = WORD.findall(s.lower())
                if not 5 <= len(toks) <= 15 or any(t in blocked for t in toks) or has_name(s):
                    continue
                if sum(c.isalpha() for c in s) / max(len(s), 1) < 0.8:
                    continue
                freq.update(set(toks))
                cands[s] = cands.get(s, 0) + 1

    vocab_floor = 3
    pool = [s for s in cands if all(freq[t] >= vocab_floor for t in WORD.findall(s.lower()))]
    rng = random.Random(args.seed)
    rng.shuffle(pool)

    def kind(s: str) -> str | None:
        toks = WORD.findall(s.lower())
        hi = sum(t in HI_MARK for t in toks)
        en = sum(t in EN_STOP for t in toks)
        if hi == 0 and en / len(toks) >= 0.3:
            return "en"
        if hi >= 2:
            return "hinglish"
        return None

    pools: dict[str, list[str]] = {"en": [], "hinglish": []}
    for s in pool:
        k = kind(s)
        if k:
            pools[k].append(s)

    covered: set[str] = set()
    top = {w for w, _ in freq.most_common(400)}

    def gain(s: str) -> float:
        toks = set(WORD.findall(s.lower()))
        return sum(1.0 + (2.0 if t in top else 0.0) for t in toks - covered) / (len(toks) ** 0.5)

    def pick(name: str, target_words: int, want_q: int, want_x: int) -> list[str]:
        items, out, words = pools[name], [], 0
        while words < target_words and items:
            subset = items
            if sum(c.endswith("?") for c in out) < want_q and len(out) >= 2:
                subset = [s for s in items if s.endswith("?")] or items
            elif sum(c.endswith("!") for c in out) < want_x and len(out) >= 4:
                subset = [s for s in items if s.endswith("!")] or items
            best = max(subset, key=gain)
            items.remove(best)
            out.append(best if best[-1] in ".!?" else best + ".")
            covered.update(WORD.findall(best.lower()))
            words += len(best.split())
        return out

    calm_cycle = ["[casual, relaxed]", "[calm, warm]", "[thoughtful, natural pace]"]

    def paragraphs(sents: list[str]) -> list[str]:
        out = []
        for i in range(0, len(sents), 3):
            grp = sents[i : i + 3]
            q = sum(g.endswith("?") for g in grp)
            x = sum(g.endswith("!") for g in grp)
            cue = "[curious]" if q >= 2 else "[excited]" if x >= 1 else calm_cycle[(i // 3) % 3]
            out.append(cue + NL + " ".join(grp))
        return out

    NL = chr(10)
    blocks = [
        "[warm, friendly]" + NL + "Hey, main apna naam bol raha hoon, aur ye meri awaaz ka ek lamba sample hai. "
        "Bas normal tarike se baat karunga, jaise dost se baat kar raha hoon.",
        *paragraphs(pick("hinglish", args.target_words // 2, 3, 2)),
        *paragraphs(pick("en", args.target_words // 3, 2, 1)),
        "[calm, warm]" + NL + "नमस्ते, मैं शाश्वत बोल रहा हूँ। आज मौसम बहुत अच्छा है, "
        "और मैं थोड़ी देर बैठकर चाय पीना चाहता हूँ।",
        "[excited]" + NL + "अरे, तुम सच में आ रहे हो? यह तो कमाल हो गया! मुझे बहुत खुशी हो रही है!",
        "[reassuring, soft]" + NL + "चिंता मत करो, सब ठीक हो जाएगा। मैं तुम्हारे साथ हूँ। "
        "Tension mat lo, hum mil ke sambhal lenge.",
        "[annoyed, a little frustrated]" + NL + "Yaar, maine kitni baar bola tha, phir bhi tumne bataya nahi. "
        "This is not okay. मुझे सच में गुस्सा आ रहा है।",
        "[sad, low voice]" + NL + "आज मन थोड़ा उदास है, पर कोई बात नहीं। Kal ek naya din hoga, "
        "and things will get better.",
        "[whispering]" + NL + "Dheere bolo, sab so rahe hain. Kal subah baat karte hain, okay?",
        "[laughing, joyful]" + NL + "Haha, ye toh kamaal ho gaya! Maza aa gaya yaar, "
        "that was honestly the funniest thing today!",
        "[clear, steady]" + NL + "Ab thoda numbers bolta hoon: ek, do, teen, chaar, paanch, chhe, saat, aath, nau, das. "
        "Phir ginti ulti: das, nau, aath, saat, chhe, paanch, chaar, teen, do, ek. "
        "एक, दो, तीन, चार, पाँच, छह, सात, आठ, नौ, दस।",
        "[warm, relieved]" + NL + "Okay, that is about it for now. Thank you for listening, aur milte hain jaldi. Bye!",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    full = (NL + NL).join(blocks) + NL
    args.out.write_text(full, encoding="utf-8")
    spoken = [ln for ln in full.splitlines() if ln.strip() and not (ln.startswith("[") and ln.endswith("]"))]
    args.out.with_name("master_script.transcript.txt").write_text(NL.join(spoken) + NL, encoding="utf-8")

    words = sum(len(ln.split()) for ln in spoken)
    print(
        f"wrote {args.out}: ~{words} spoken words (~{words / 2.6:.0f}s), "
        f"{len(blocks)} paragraphs, {len(covered)} distinct words from your messages"
    )


if __name__ == "__main__":
    main()
