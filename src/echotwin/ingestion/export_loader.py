"""Loading WhatsApp exports from disk: plain TXT, ZIP (multiple chats), or directory.

WhatsApp exports come in two shapes:

* **TXT** — one chat per file.
* **ZIP / directory** — one ``_chat.txt`` per conversation (Android exports bundle
  every chat). Each becomes its own conversation_id, derived from the file name
  so datasets get genuine cross-conversation splits.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from ..exceptions import ExportValidationError, UploadValidationError

_MAX_CHAT_FILE_BYTES = 200 * 1024 * 1024  # 200 MB of uncompressed chat text


@dataclass
class LoadedExport:
    source_path: Path
    chat_text: str
    inner_name: str  # name/path of the .txt actually parsed
    encoding: str
    additional_files: list[str] = field(default_factory=list)


@dataclass
class LoadedExportBundle:
    source_path: Path
    chats: list[LoadedExport]

    @property
    def chat_count(self) -> int:
        return len(self.chats)


def conversation_id_from_name(name: str, fallback: str) -> str:
    """Derive a stable conversation id from a chat file name.

    'WhatsApp Chat with Priya/_chat.txt' -> 'priya'
    'Group 'Hostel Crew''_chat.txt        -> 'hostel-crew'
    """
    stem = Path(name).name
    if stem.lower().endswith("_chat.txt"):
        stem = stem[: -len("_chat.txt")]
    stem = stem.strip().strip("'\"")
    stem = re.sub(r"^(whatsapp\s+chat\s+with|whatsapp\s+chat\s+from)\s+", "", stem, flags=re.I)
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return slug[:40] or fallback


def _decode(raw: bytes) -> tuple[str, str]:
    for bom, enc in [
        (b"\xef\xbb\xbf", "utf-8-sig"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
    ]:
        if raw.startswith(bom):
            return raw.decode(enc), enc
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1"


def _pick_chat_names(names: list[str]) -> list[str]:
    candidates = [n for n in names if n.lower().endswith(".txt")]
    if not candidates:
        return []
    scored = sorted(
        candidates,
        key=lambda n: (
            0 if n.lower().endswith("_chat.txt") else 1,
            0 if "chat" in n.lower() else 1,
            n,
        ),
    )
    return scored


def load_export(path: str | Path) -> LoadedExportBundle:
    """Load one TXT, or every chat in a ZIP / directory."""
    p = Path(path)
    if not p.exists():
        raise ExportValidationError(f"Path does not exist: {p}")

    if p.is_dir():
        txts = sorted(p.rglob("*.txt"))
        if not txts:
            raise ExportValidationError(f"No .txt chat file found under directory: {p}")
        chats = [_load_txt(t, stem_i=i) for i, t in enumerate(txts)]
        return LoadedExportBundle(source_path=p, chats=chats)

    if p.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(p) as zf:
                names = zf.namelist()
                bad = [
                    n
                    for n in names
                    if n.startswith("/") or ".." in n.split("/") or ":" in n.split("/")[0]
                ]
                if bad:
                    raise UploadValidationError(f"Unsafe entry names in zip: {bad[:3]}")
                chat_names = _pick_chat_names(names)
                if not chat_names:
                    raise ExportValidationError(
                        "ZIP contains no .txt chat export. WhatsApp exports are named "
                        "like 'WhatsApp Chat with Someone/_chat.txt'."
                    )
                zip_chats: list[LoadedExport] = []
                for chat_name in chat_names:
                    info = zf.getinfo(chat_name)
                    if info.file_size > _MAX_CHAT_FILE_BYTES:
                        raise UploadValidationError("Chat file inside ZIP is too large.")
                    text, enc = _decode(zf.read(chat_name))
                    zip_chats.append(
                        LoadedExport(
                            source_path=p,
                            chat_text=text,
                            inner_name=chat_name,
                            encoding=enc,
                        )
                    )
                others = [n for n in names if not n.endswith("/") and n not in chat_names]
                for chat in zip_chats:
                    chat.additional_files = others[:50]
                return LoadedExportBundle(source_path=p, chats=zip_chats)
        except zipfile.BadZipFile as e:
            raise ExportValidationError(f"Not a valid ZIP file: {p} ({e})") from e

    if p.suffix.lower() == ".txt":
        return LoadedExportBundle(source_path=p, chats=[_load_txt(p, stem_i=0)])

    raise UploadValidationError(
        f"Unsupported file type '{p.suffix}'. Provide a WhatsApp .txt or .zip export."
    )


def _load_txt(target: Path, *, stem_i: int = 0) -> LoadedExport:
    if target.stat().st_size > _MAX_CHAT_FILE_BYTES:
        raise UploadValidationError("Chat file is too large.")
    raw = target.read_bytes()
    text, enc = _decode(raw)
    return LoadedExport(
        source_path=target,
        chat_text=text,
        inner_name=target.name,
        encoding=enc,
    )
