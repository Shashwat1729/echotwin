"""Split one long, continuous recording into reference-sized clips.

A single "master take" is far easier to record than dozens of short clips, but
voice backends want short references (a few seconds to about a minute). This
module cuts a long take at natural pauses, so no word is chopped, and — when
the script that was read is known — assigns each clip its matching text.

Alignment is by *speech-time vs word-count proportion*, snapped to sentence
boundaries. It is an approximation of what an ASR forced-alignment would give;
callers should label such transcripts as script-aligned, not verified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

CUE_LINE = re.compile(r"^\s*\[[^\]]+\]\s*$")
SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


@dataclass(frozen=True)
class Segment:
    start: int  # sample index, inclusive
    end: int  # sample index, exclusive
    speech_seconds: float

    def seconds(self, sr: int) -> float:
        return (self.end - self.start) / sr


def _frame_db(x: np.ndarray, sr: int, frame_ms: int = 25, hop_ms: int = 10) -> tuple[np.ndarray, int]:
    frame = max(int(sr * frame_ms / 1000), 1)
    hop = max(int(sr * hop_ms / 1000), 1)
    n = max(1 + (len(x) - frame) // hop, 1)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    idx = np.clip(idx, 0, len(x) - 1)
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-9), hop


def split_on_pauses(
    samples: np.ndarray,
    sr: int,
    *,
    min_seconds: float = 6.0,
    target_seconds: float = 14.0,
    max_seconds: float = 25.0,
    min_pause: float = 0.25,
    silence_margin_db: float = 35.0,
    edge_pad: float = 0.12,
) -> list[Segment]:
    """Cut at the best pause near ``target_seconds``; never inside speech."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    db, hop = _frame_db(x, sr)
    if len(db) == 0:
        return []
    thresh = float(np.percentile(db, 95)) - silence_margin_db
    silent = db < thresh
    fps = sr / hop

    # pauses = runs of silent frames at least ``min_pause`` long
    pauses: list[tuple[int, int]] = []  # (start_frame, end_frame)
    i = 0
    while i < len(silent):
        if silent[i]:
            j = i
            while j < len(silent) and silent[j]:
                j += 1
            if (j - i) / fps >= min_pause:
                pauses.append((i, j))
            i = j
        else:
            i += 1

    total_s = len(x) / sr
    cuts: list[int] = []  # sample indices
    start_s = 0.0
    while total_s - start_s > max_seconds:
        lo, hi = start_s + min_seconds, start_s + max_seconds
        best, best_score = None, -1e9
        for a, b in pauses:
            mid = ((a + b) / 2) / fps
            if lo <= mid <= hi:
                score = (b - a) / fps - 0.05 * abs(mid - (start_s + target_seconds))
                if score > best_score:
                    best, best_score = mid, score
        if best is None:  # no pause in range: cut at the quietest frame instead
            a, b = int((start_s + min_seconds) * fps), int((start_s + max_seconds) * fps)
            best = (a + int(np.argmin(db[a:b]))) / fps
        cuts.append(int(best * sr))
        start_s = best

    bounds = [0, *cuts, len(x)]
    # merge a too-short tail into the previous clip
    if len(bounds) > 2 and (bounds[-1] - bounds[-2]) / sr < min_seconds:
        del bounds[-2]

    pad = int(edge_pad * sr)
    out: list[Segment] = []
    for s, e in pairwise(bounds):
        s2, e2 = max(s - pad, 0) if s else 0, min(e + pad, len(x))
        # trim leading/trailing silence but keep a small pad
        fa, fb = int(s2 / hop), min(int(e2 / hop), len(silent))
        voiced = np.where(~silent[fa:fb])[0]
        if len(voiced) == 0:
            continue
        s3 = max(s2, (fa + int(voiced[0])) * hop - pad)
        e3 = min(e2, (fa + int(voiced[-1]) + 1) * hop + pad)
        speech = float((~silent[fa:fb]).sum() / fps)
        out.append(Segment(int(s3), int(e3), speech))
    return out


def spoken_sentences(script_text: str) -> list[str]:
    """Script text without ``[mood cue]`` lines, split into sentences."""
    lines = [ln.strip() for ln in script_text.splitlines() if ln.strip() and not CUE_LINE.match(ln)]
    sentences: list[str] = []
    for ln in lines:
        sentences.extend(p.strip() for p in SENTENCE_END.split(ln) if p.strip())
    return sentences


def assign_transcripts(segments: list[Segment], script_text: str) -> list[str]:
    """Give each segment the sentences whose word-midpoint falls inside its speech share."""
    sents = spoken_sentences(script_text)
    if not segments or not sents:
        return [""] * len(segments)
    total_speech = sum(s.speech_seconds for s in segments) or 1.0
    edges = np.cumsum([s.speech_seconds for s in segments]) / total_speech
    words = np.array([len(s.split()) for s in sents], dtype=float)
    mids = (np.cumsum(words) - words / 2) / words.sum()
    out: list[list[str]] = [[] for _ in segments]
    for sent, mid in zip(sents, mids, strict=True):
        k = int(np.searchsorted(edges, mid, side="left"))
        out[min(k, len(segments) - 1)].append(sent)
    return [" ".join(parts) for parts in out]


def normalize_peak(samples: np.ndarray, ceiling: float = 0.9) -> np.ndarray:
    """Scale down only if the signal is hotter than ``ceiling`` (never boosts)."""
    x = np.asarray(samples, dtype=np.float32)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    return x * (ceiling / peak) if peak > ceiling else x
