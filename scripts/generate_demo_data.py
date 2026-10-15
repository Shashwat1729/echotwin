"""Generate deterministic synthetic demo data for EchoTwin.

Everything written here is FICTIONAL. No real conversations, names, or voices.
Running this script reproduces the checked-in files under ``examples/`` exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

TARGET_AUTHOR = "Aarav"

CHAT_LINES: list[str] = [
    # --- system / group setup ---
    "[10/01/2025, 20:00:01] Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them. Tap to learn more.",
    "[10/01/2025, 20:00:02] Rohit created group \"Hostel Crew 🏠\"",
    "[10/01/2025, 20:00:05] Rohit added Priya and Sana",
    # --- day 1 ---
    "[10/01/2025, 20:14:33] Priya: Good morning everyone ☀️",
    "[10/01/2025, 20:15:10] Rohit: subah subah good morning 😄",
    "[10/01/2025, 20:16:02] Aarav: morning! ummeed hai aaj ka din achha jayega",
    "[10/01/2025, 20:17:44] Sana: Good Morning!! ✨☕",
    "[10/01/2025, 20:20:19] Priya: guys canteen ka menu badal gaya hai",
    "[10/01/2025, 20:21:03] Priya: paneer tikkaavailable from monday apparently",
    "[10/01/2025, 20:22:41] Aarav: serious? tab toh mess ka khana chhod dena chahiye",
    "[10/01/2025, 20:23:10] Rohit: <Media omitted>",
    "[10/01/2025, 20:23:12] Rohit: dekh lo 🤤",
    "[10/01/2025, 20:31:57] Sana: ooh batao kitne ka hai",
    "[10/01/2025, 20:33:26] Aarav: abhi pata nahi",
    "[10/01/2025, 21:02:00] Priya: chalo kal try karte h main jeet jaungi 😎",
    # --- day 2, 12h timestamp format line ---
    "[11/01/2025, 09:05:12 am] Aarav: good morning!! kal ka plan kya hai?",
    "[11/01/2025, 09:06:45 am] Priya: 10 bje library milte h, project due hai friday ko",
    "[11/01/2025, 09:07:20 am] Rohit: ruk 10 bje? class 9:30 hai bhai 😭",
    "[11/01/2025, 09:08:03 am] Priya: theek hai phir 11 bje",
    "[11/01/2025, 09:08:44 am] Aarav: 11 bje theek hai, main thoda late ho sakta hoon. traffic hai kal subah usually",
    "[11/01/2025, 09:09:02 am] Sana: ok 👍",
    "[11/01/2025, 09:15:31 am] Aarav: kal subah late ho jayega toh call kar lio mujhe",
    # --- day 2 evening, hindi devanagari, multiline, edited, deleted ---
    "[11/01/2025, 19:40:11] Sana: आज की कक्षा बहुत अच्छी थी 😊",
    "[11/01/2025, 19:41:00] Aarav: वाकई? मैं क्लास मिस कर गया था 😅",
    "[11/01/2025, 19:41:35] Priya: मैंने नोट्स बनाए हैं, शाम को भेजती हूँ <This message was edited>",
    "[11/01/2025, 19:42:10] Aarav: thank you!!",
    "[11/01/2025, 19:44:02] Rohit: yeh kya notes culture chal rahi hai hostel me 😂",
    "[11/01/2025, 20:10:55] Rohit: This message was deleted",
    "[11/01/2025, 20:12:00] Aarav: kya hua?",
    # --- day 3 ---
    "[12/01/2025, 08:30:00] Priya: reminders: project demo 4pm today",
    "[12/01/2025, 08:31:20] Aarav: arey haan. kal raat tak coding pending hai, aaj chalega hi nahi",
    "[12/01/2025, 08:32:10] Sana: main UI dekh lungi tu tension mat lo",
    "[12/01/2025, 08:33:00] Aarav: thank you yaar 🙏",
    "[12/01/2025, 14:02:11] Priya: demo kaisa raha?",
    "[12/01/2025, 14:05:47] Aarav: expected se better gaya actually!! sab kuch ek hi baar me chal gaya, aur sir ne flow appreciate kiya",
    "[12/01/2025, 14:06:30] Rohit: badhiya!! party kaun dega 😌",
    "[12/01/2025, 14:10:02] Aarav: canteen ka paneer tika uspe treat, sunday ko",
    # --- malformed trailing lines (truncated export) ---
    "[12/01/2025, 14:1",
    "random text without any timestamp",
]


def build_chat_text() -> str:
    return "\n".join(CHAT_LINES) + "\n"


def build_training_examples() -> list[dict]:
    """Illustrative training examples derived from the synthetic conversation.

    example_type values: natural_reply (context from another participant) and
    natural_statement (author speaking without an immediately preceding turn).
    """
    return [
        {
            "messages": [
                {"role": "user", "content": "guys canteen ka menu badal gaya hai"},
                {"role": "assistant", "content": "serious? tab toh mess ka khana chhod dena chahiye"},
            ],
            "metadata": {
                "conversation_id": "conversation-1",
                "language": "hi-Latn",
                "example_type": "natural_reply",
                "source_message_line": 12,
            },
        },
        {
            "messages": [
                {"role": "user", "content": "ruko, 10 bje?"},
                {"role": "assistant", "content": "bas thoda late ho sakta hoon, traffic usually hota hai subah"},
            ],
            "metadata": {
                "conversation_id": "conversation-1",
                "language": "hi-Latn",
                "example_type": "natural_reply",
                "source_message_line": 19,
            },
        },
        {
            "messages": [
                {"role": "user", "content": "project ka demo kaisa raha bhai?"},
                {"role": "assistant", "content": "bekaar nahi gaya, expected se better. sir ne tareef ki 😄"},
            ],
            "metadata": {
                "conversation_id": "conversation-1",
                "language": "hi-Latn",
                "example_type": "natural_reply",
                "source_message_line": 38,
            },
        },
    ]


def build_preference_pairs() -> list[dict]:
    """Synthetic preference annotations for demo/testing.

    IMPORTANT: these are hand-authored demo labels, NOT human annotations of real
    model outputs. The pipeline carries this provenance in every record and the UI
    displays it.
    """
    return [
        {
            "context": [
                {"role": "user", "content": "guys canteen ka menu badal gaya hai"},
            ],
            "chosen": "serious? tab toh mess ka khana chhod dena chahiye",
            "rejected": "That is indeed noteworthy and rather good news for the hostel residents.",
            "provenance": "synthetic-demo",
            "confidence": 1.0,
            "notes": "Rejected response is formal English, unlike the author's Hinglish style.",
        },
        {
            "context": [
                {"role": "user", "content": "demo kaisa raha?"},
            ],
            "chosen": "bekaar nahi gaya, expected se better. sir ne tareef ki 😄",
            "rejected": "The demonstration proceeded adequately.",
            "provenance": "synthetic-demo",
            "confidence": 1.0,
            "notes": "Chosen preserves length, emoji and code-mixing patterns.",
        },
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    examples = repo_root / "examples"
    examples.mkdir(exist_ok=True)

    (examples / "synthetic_whatsapp.txt").write_text(build_chat_text(), encoding="utf-8", newline="\n")
    write_jsonl(examples / "tiny_training_dataset.jsonl", build_training_examples())
    write_jsonl(examples / "synthetic_preference_pairs.jsonl", build_preference_pairs())

    print("Wrote examples/synthetic_whatsapp.txt")
    print("Wrote examples/tiny_training_dataset.jsonl")
    print("Wrote examples/synthetic_preference_pairs.jsonl")


if __name__ == "__main__":
    main()
