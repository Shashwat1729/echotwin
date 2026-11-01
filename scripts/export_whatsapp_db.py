"""Convert a decrypted WhatsApp ``msgstore.db`` into EchoTwin-ready text exports.

Text only: media, stickers, calls and system events are dropped. Every chat in
which *you* sent at least one text message is exported with both sides of the
conversation (needed for context windows).

Outputs (all under ``--out``, which is git-ignored):
  chats/<n>_<name>_chat.txt   one Android-style export per chat
  whatsapp_text_chats.zip     all of the above, ready for ``echotwin import whatsapp``
  messages.jsonl              one JSON object per text message
  summary.json                counts only, no message content

Usage:
  python scripts/export_whatsapp_db.py --db data/whatsapp_export/raw/msgstore.db \
      --me "Your Name" [--contacts contacts.tsv | --adb-contacts]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def load_contacts(tsv: Path | None, use_adb: bool) -> dict[str, str]:
    """Map last-10-digits of a phone number to a display name."""
    rows: list[tuple[str, str]] = []
    if tsv and tsv.exists():
        for line in tsv.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                n, p = line.split("\t", 1)
                rows.append((n, p))
    elif use_adb and (adb := shutil.which("adb")):
        out = subprocess.run(
            [adb, "shell", "content query --uri content://com.android.contacts/data/phones "
             "--projection display_name:data1"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        ).stdout
        for m in re.finditer(r"display_name=(.*?), data1=(.*)$", out, re.M):
            rows.append((m.group(1).strip(), m.group(2).strip()))
        if tsv:
            tsv.parent.mkdir(parents=True, exist_ok=True)
            tsv.write_text("\n".join(f"{n}\t{p}" for n, p in rows), encoding="utf-8")
    book: dict[str, str] = {}
    for name, phone in rows:
        d = digits(phone)
        if len(d) >= 7:
            book.setdefault(d[-10:], name)
    return book


def safe(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\r\n]+', " ", name).strip(" .")
    return name[:60] or "chat"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/whatsapp_export"))
    ap.add_argument("--me", required=True, help="Your display name (the author EchoTwin learns).")
    ap.add_argument("--contacts", type=Path, default=Path("data/whatsapp_export/raw/contacts.tsv"))
    ap.add_argument("--adb-contacts", action="store_true", help="Read phone contacts over adb for names.")
    args = ap.parse_args()

    book = load_contacts(args.contacts, args.adb_contacts)
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    jids = {r["_id"]: (r["user"], r["server"], r["raw_string"]) for r in con.execute("select * from jid")}
    lid_names = {r["lid_row_id"]: r["display_name"] for r in con.execute("select * from lid_display_name")}
    lid_to_pn: dict[int, int] = {}
    try:
        for r in con.execute("select lid_row_id, jid_row_id from jid_map"):
            lid_to_pn[r["lid_row_id"]] = r["jid_row_id"]
    except sqlite3.Error:
        pass

    def person(jid_row: int | None) -> str:
        if jid_row is None:
            return "Unknown"
        if jid_row in lid_to_pn:
            jid_row = lid_to_pn[jid_row] if lid_to_pn[jid_row] in jids else jid_row
        user, server, _ = jids.get(jid_row, ("", "", ""))
        if server == "s.whatsapp.net" and (nm := book.get(digits(user)[-10:])):
            return nm
        if lid_names.get(jid_row):
            return lid_names[jid_row]
        return f"+{user}" if server == "s.whatsapp.net" and user else (user or "Unknown")

    chats = {}
    for r in con.execute(
        "select c._id, c.subject, c.jid_row_id from chat c where c._id in "
        "(select chat_row_id from message where from_me=1 and message_type=0 and text_data != '')"
    ):
        _, server, _ = jids.get(r["jid_row_id"], ("", "", ""))
        is_group = server == "g.us"
        name = r["subject"] if is_group and r["subject"] else person(r["jid_row_id"])
        chats[r["_id"]] = (name, is_group, server)

    by_chat: dict[int, list[dict]] = defaultdict(list)
    for m in con.execute(
        "select _id, chat_row_id, from_me, sender_jid_row_id, timestamp, text_data from message "
        "where message_type = 0 and text_data is not null and text_data != '' order by chat_row_id, timestamp, _id"
    ):
        cid = m["chat_row_id"]
        if cid not in chats:
            continue
        name, is_group, _ = chats[cid]
        if m["from_me"]:
            sender = args.me
        elif is_group:
            sender = person(m["sender_jid_row_id"])
        else:
            sender = name
        by_chat[cid].append(
            {"ts": m["timestamp"] / 1000, "sender": sender, "from_me": bool(m["from_me"]), "text": m["text_data"]}
        )

    out = args.out
    chat_dir = out / "chats"
    if chat_dir.exists():
        shutil.rmtree(chat_dir)
    chat_dir.mkdir(parents=True)
    header_like = re.compile(r"^\[\d{2}/\d{2}/\d{4}, ")
    total = mine = 0
    summary_chats = []
    with (out / "messages.jsonl").open("w", encoding="utf-8") as jl:
        for n, (cid, msgs) in enumerate(sorted(by_chat.items(), key=lambda kv: -len(kv[1])), 1):
            name, is_group, _ = chats[cid]
            lines = []
            for m in msgs:
                when = datetime.fromtimestamp(m["ts"])
                body = "\n".join(("​" + ln if header_like.match(ln) else ln) for ln in m["text"].splitlines())
                lines.append(f"[{when:%d/%m/%Y, %H:%M:%S}] {m['sender']}: {body}")
                jl.write(json.dumps(
                    {"chat_id": cid, "chat": name, "is_group": is_group, "timestamp": when.isoformat(),
                     "sender": m["sender"], "from_me": m["from_me"], "text": m["text"]},
                    ensure_ascii=False) + "\n")
            (chat_dir / f"{n:04d} WhatsApp Chat with {safe(name)}_chat.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            sent = sum(m["from_me"] for m in msgs)
            total += len(msgs)
            mine += sent
            summary_chats.append({"file_no": n, "is_group": is_group, "messages": len(msgs), "sent_by_me": sent})

    with zipfile.ZipFile(out / "whatsapp_text_chats.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(chat_dir.glob("*_chat.txt")):
            z.write(p, p.name)

    (out / "summary.json").write_text(json.dumps(
        {"chats": len(by_chat), "messages_total": total, "messages_sent_by_me": mine,
         "group_chats": sum(c["is_group"] for c in summary_chats), "per_chat": summary_chats}, indent=1),
        encoding="utf-8")
    print(f"chats={len(by_chat)} messages={total} sent_by_me={mine}")


if __name__ == "__main__":
    main()
