"""Built-in bilingual recording-script library.

All scripts are real sentences the user reads aloud — no placeholders. The
starter set covers the Hindi (Devanagari), English and Hinglish scripts plus
additional categories (numbers, questions, phonetic coverage, emotions, short
voice-message style, long paragraphs).

Editable: every entry can be customized or skipped. ``[NAME]``, ``[DATE]`` and
``[TIME]`` placeholders are rendered from user-supplied values; when unset the
placeholder is shown as an editable blank instead of fabricating a date/time.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

SCRIPT_LIBRARY_VERSION = "1.0.0"


class ScriptLanguage(StrEnum):
    HI = "hi"
    EN = "en"
    HINGLISH = "hinglish"


class ScriptCategory(StrEnum):
    CLEAN_REFERENCE = "clean_reference"
    STATEMENT = "statement"
    QUESTION = "question"
    NUMBERS = "numbers"
    DATES_TIMES = "dates_times"
    ACKNOWLEDGEMENT = "acknowledgement"
    EXPLANATION = "explanation"
    CONVERSATIONAL = "conversational"
    EMOTION_CALM = "emotion_calm"
    EMOTION_REASSURANCE = "emotion_reassurance"
    EMPHASIS_CONTRAST = "emphasis_contrast"
    CODE_SWITCH = "code_switch"
    SHORT_VOICE_NOTE = "short_voice_note"
    LONG_PARAGRAPH = "long_paragraph"
    PHONETIC_HI_DENTAL_RETROFLEX = "phonetic_hi_dental_retroflex"
    PHONETIC_HI_ASPIRATION = "phonetic_hi_aspiration"
    PHONETIC_HI_NASALS_VOWELS = "phonetic_hi_nasals_vowels"
    PHONETIC_HI_CLUSTERS = "phonetic_hi_clusters"
    PHONETIC_EN_VOWELS = "phonetic_en_vowels"
    PHONETIC_EN_CLUSTERS = "phonetic_en_clusters"
    PHONETIC_EN_FRICATIVES = "phonetic_en_fricatives"


class ScriptEntry(BaseModel):
    id: str
    language: ScriptLanguage
    script: str = Field(description="devanagari | latin")
    category: ScriptCategory
    text: str
    translation: str | None = None
    pronunciation: str | None = Field(
        default=None, description="Approximate ASCII hint (NOT IPA); Devanagari entries only"
    )
    est_seconds: float = 6.0
    tags: list[str] = Field(default_factory=list)
    contains_placeholders: bool = False


def _hi(
    entry_id: str,
    text: str,
    category: ScriptCategory,
    translation: str,
    pronunciation: str | None = None,
    est_seconds: float = 6.0,
    **kwargs,
) -> ScriptEntry:
    return ScriptEntry(
        id=entry_id,
        language=ScriptLanguage.HI,
        script="devanagari",
        category=category,
        text=text,
        translation=translation,
        pronunciation=pronunciation,
        est_seconds=est_seconds,
        contains_placeholders=any(p in text for p in ("[NAME]", "[DATE]", "[TIME]")),
        **kwargs,
    )


def _en(
    entry_id: str,
    text: str,
    category: ScriptCategory,
    est_seconds: float = 6.0,
    **kwargs,
) -> ScriptEntry:
    return ScriptEntry(
        id=entry_id,
        language=ScriptLanguage.EN,
        script="latin",
        category=category,
        text=text,
        translation=None,
        pronunciation=None,
        est_seconds=est_seconds,
        contains_placeholders=any(p in text for p in ("[NAME]", "[DATE]", "[TIME]")),
        **kwargs,
    )


def _he(
    entry_id: str,
    text: str,
    translation: str,
    category: ScriptCategory = ScriptCategory.CODE_SWITCH,
    est_seconds: float = 6.0,
    **kwargs,
) -> ScriptEntry:
    return ScriptEntry(
        id=entry_id,
        language=ScriptLanguage.HINGLISH,
        script="latin",
        category=category,
        text=text,
        translation=translation,
        pronunciation=None,
        est_seconds=est_seconds,
        contains_placeholders=any(p in text for p in ("[NAME]", "[DATE]", "[TIME]")),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# A. Hindi — Devanagari (starter set from the product spec + additions)
# ---------------------------------------------------------------------------
_HINDI_STARTER: list[ScriptEntry] = [
    _hi(
        "hi-clean-01",
        "\u0928\u092e\u0938\u094d\u0924\u0947, \u092e\u0947\u0930\u093e \u0928\u093e\u092e [NAME] \u0939\u0948 \u0914\u0930 \u092e\u0948\u0902 \u0905\u092a\u0928\u0940 \u0906\u0935\u093e\u095c \u0915\u093e \u090f\u0915 \u0928\u092e\u0942\u0928\u093e \u0930\u093f\u0915\u0949\u0930\u094d\u0921 \u0915\u0930 \u0930\u0939\u093e \u0939\u0942\u0901\u0964",
        ScriptCategory.CLEAN_REFERENCE,
        "Hello, my name is [NAME] and I am recording a sample of my voice.",
        "namaste, mera naam [NAME] hai aur main apni aawaz ka ek namoona record kar raha hoon.",
        8.0,
    ),
    _hi(
        "hi-weather",
        "\u0906\u091c \u092e\u094c\u0938\u092e \u0905\u091a\u094d\u091b\u093e \u0939\u0948 \u0914\u0930 \u092e\u0948\u0902 \u0906\u0930\u093e\u092e \u0938\u0947, \u0938\u093e\u095b \u0906\u0935\u093e\u095c \u092e\u0947\u0902 \u092c\u093e\u0924 \u0915\u0930 \u0930\u0939\u093e \u0939\u0942\u0901\u0964",
        ScriptCategory.STATEMENT,
        "Today the weather is nice and I am speaking comfortably, in a clear voice.",
        "aaj mausam achha hai aur main aaram se, saaf aawaz mein baat kar raha hoon.",
    ),
    _hi(
        "hi-question-day",
        "\u0906\u092a \u0915\u0948\u0938\u0947 \u0939\u0948\u0902? \u0909\u092e\u094d\u092e\u0940\u0926 \u0939\u0948 \u0915\u093f \u0906\u092a\u0915\u093e \u0926\u093f\u0928 \u0905\u091a\u094d\u091b\u093e \u091c\u093e \u0930\u0939\u093e \u0939\u094b\u0917\u093e\u0964",
        ScriptCategory.QUESTION,
        "How are you? I hope your day is going well.",
        "aap kaise hain? umeed hai ki aapka din achha ja raha hoga.",
    ),
    _hi(
        "hi-think-more",
        "\u092e\u0941\u091d\u0947 \u092f\u0939 \u092c\u093e\u0924 \u0938\u092e\u091d \u092e\u0947\u0902 \u0906 \u0917\u0908 \u0939\u0948, \u0932\u0947\u0915\u093f\u0928 \u092e\u0948\u0902 \u0907\u0938\u0915\u0947 \u092c\u093e\u0930\u0947 \u092e\u0947\u0902 \u0925\u094b\u095c\u093e \u0914\u0930 \u0938\u094b\u091a\u0928\u093e \u091a\u093e\u0939\u0924\u093e \u0939\u0942\u0901\u0964",
        ScriptCategory.EXPLANATION,
        "I understood this point, but I would like to think about it a little more.",
        "mujhe ye baat samajh mein aa gayi hai, lekin main iske baare mein thoda aur sochna chahata hoon.",
        8.0,
    ),
    _hi(
        "hi-what-next",
        "\u0905\u0917\u0930 \u0906\u092a\u0915\u094b \u0938\u092e\u092f \u092e\u093f\u0932\u0947, \u0924\u094b \u092e\u0941\u091d\u0947 \u092c\u0924\u093e\u0907\u090f \u0915\u093f \u0939\u092e \u0906\u0917\u0947 \u0915\u094d\u092f\u093e \u0915\u0930 \u0938\u0915\u0924\u0947 \u0939\u0948\u0902\u0964",
        ScriptCategory.QUESTION,
        "If you have time, let me know what we can do next.",
        "agar aapko samay mile, toh mujhe batayiye ki hum aage kya kar sakte hain.",
    ),
    _hi(
        "hi-at-home",
        "\u092e\u0948\u0902 \u0905\u092d\u0940 \u0918\u0930 \u092a\u0930 \u0939\u0942\u0901 \u0914\u0930 \u0925\u094b\u095c\u0940 \u0926\u0947\u0930 \u092c\u093e\u0926 \u0906\u092a\u0938\u0947 \u092c\u093e\u0924 \u0915\u0930\u0942\u0901\u0917\u093e\u0964",
        ScriptCategory.STATEMENT,
        "I am at home right now and I will talk to you a little later.",
        "main abhi ghar par hoon aur thodi der baad aapse baat karunga.",
    ),
    _hi(
        "hi-read-message",
        "\u0915\u094d\u092f\u093e \u0906\u092a\u0928\u0947 \u0935\u0939 \u0938\u0902\u0926\u0947\u0936 \u092a\u095d\u093e? \u092e\u0941\u091d\u0947 \u0909\u0938\u0915\u093e \u091c\u0935\u093e\u092c \u0926\u0947\u0928\u093e \u0939\u0948\u0964",
        ScriptCategory.QUESTION,
        "Did you read that message? I need to reply to it.",
        "kya aapne wah sandesh padha? mujhe uska jawab dena hai.",
    ),
    _hi(
        "hi-small-detail",
        "\u0915\u092d\u0940-\u0915\u092d\u0940 \u091b\u094b\u091f\u0940-\u0938\u0940 \u092c\u093e\u0924 \u092d\u0940 \u092c\u0939\u0941\u0924 \u092e\u0939\u0924\u094d\u0935\u092a\u0942\u0930\u094d\u0923 \u0939\u094b \u0938\u0915\u0924\u0940 \u0939\u0948\u0964",
        ScriptCategory.EMPHASIS_CONTRAST,
        "Sometimes even a small thing can be very important.",
        "kabhi-kabhi chhoti-si baat bhi bahut mahatwapurn ho sakti hai.",
    ),
    _hi(
        "hi-first-then",
        "\u092a\u0939\u0932\u0947 \u0939\u092e \u092a\u0942\u0930\u0940 \u092c\u093e\u0924 \u0938\u092e\u091d\u0947\u0902\u0917\u0947, \u092b\u093f\u0930 \u0906\u0930\u093e\u092e \u0938\u0947 \u092b\u0948\u0938\u0932\u093e \u0915\u0930\u0947\u0902\u0917\u0947\u0964",
        ScriptCategory.EXPLANATION,
        "First we will understand the whole matter, then decide at ease.",
        "pehle hum poori baat samjhenge, phir aaram se faisla karenge.",
    ),
    _hi(
        "hi-good-idea",
        "\u092e\u0941\u091d\u0947 \u0932\u0917\u0924\u093e \u0939\u0948 \u0915\u093f \u092f\u0939 \u090f\u0915 \u0905\u091a\u094d\u091b\u093e \u0935\u093f\u091a\u093e\u0930 \u0939\u0948, \u0932\u0947\u0915\u093f\u0928 \u0907\u0938\u0915\u0947 \u0915\u0941\u091b \u0926\u0942\u0938\u0930\u0947 \u092a\u0939\u0932\u0942 \u092d\u0940 \u0939\u0948\u0902\u0964",
        ScriptCategory.EMPHASIS_CONTRAST,
        "I think this is a good idea, but there are some other aspects too.",
        "mujhe lagta hai ki ye ek achha vichaar hai, lekin iske kuch doosre pehlu bhi hain.",
        8.0,
    ),
    _hi(
        "hi-speak-slowly",
        "\u0915\u0943\u092a\u092f\u093e \u0927\u0940\u0930\u0947-\u0927\u0940\u0930\u0947 \u092c\u094b\u0932\u093f\u090f \u0924\u093e\u0915\u093f \u0939\u0930 \u0936\u092c\u094d\u0926 \u0938\u093e\u095b \u0938\u0941\u0928\u093e\u0908 \u0926\u0947\u0964",
        ScriptCategory.EXPLANATION,
        "Please speak slowly enough that every word is heard clearly.",
        "kripya dheere-dheere boliye taaki har shabd saaf sunaai de.",
    ),
    _hi(
        "hi-both-languages",
        "\u092e\u0948\u0902 \u0939\u093f\u0902\u0926\u0940 \u0914\u0930 \u0905\u0902\u0917\u094d\u0930\u0947\u095c\u0940 \u0926\u094b\u0928\u094b\u0902 \u092d\u093e\u0937\u093e\u0913\u0902 \u092e\u0947\u0902 \u0938\u094d\u0935\u093e\u092d\u093e\u0935\u093f\u0915 \u0930\u0942\u092a \u0938\u0947 \u092c\u093e\u0924 \u0915\u0930\u0928\u0947 \u0915\u0940 \u0915\u094b\u0936\u093f\u0936 \u0915\u0930 \u0930\u0939\u093e \u0939\u0942\u0901\u0964",
        ScriptCategory.STATEMENT,
        "I am trying to speak naturally in both Hindi and English.",
        "main hindi aur angrezi dono bhashaon mein swaabhavik roop se baat karne ki koshish kar raha hoon.",
        8.0,
    ),
    _hi(
        "hi-numbers",
        "\u090f\u0915, \u0926\u094b, \u0924\u0940\u0928, \u091a\u093e\u0930, \u092a\u093e\u0901\u091a, \u091b\u0939, \u0938\u093e\u0924, \u0906\u0920, \u0928\u094c, \u0926\u0938\u0964",
        ScriptCategory.NUMBERS,
        "One, two, three, four, five, six, seven, eight, nine, ten.",
        "ek, do, teen, chaar, paanch, chhah, saat, aath, nau, das.",
    ),
    _hi(
        "hi-date-time",
        "\u0906\u091c [DATE] \u0939\u0948 \u0914\u0930 \u0905\u092d\u0940 \u0936\u093e\u092e \u0915\u0947 \u0932\u0917\u092d\u0917 [TIME] \u092c\u091c\u0947 \u0939\u0948\u0902\u0964",
        ScriptCategory.DATES_TIMES,
        "Today is [DATE] and it is now about [TIME] in the evening.",
        "aaj [DATE] hai aur abhi shaam ke lagbhag [TIME] baje hain.",
    ),
    _hi(
        "hi-thanks",
        "\u0927\u0928\u094d\u092f\u0935\u093e\u0926, \u092e\u0948\u0902 \u0906\u092a\u0938\u0947 \u092b\u093f\u0930 \u092c\u093e\u0924 \u0915\u0930\u0942\u0901\u0917\u093e\u0964",
        ScriptCategory.ACKNOWLEDGEMENT,
        "Thank you, I will talk to you again.",
        "dhanyavaad, main aapse phir baat karunga.",
    ),
]

_HINDI_ADDITIONAL: list[ScriptEntry] = [
    _hi(
        "hi-retroflex-01",
        "\u091f\u092e\u093e\u091f\u0930, \u0924\u0940\u0928, \u0924\u092c, \u0921\u0930, \u0921\u093e\u0915, \u0926\u093f\u0928, \u0925\u093e\u0932\u0940, \u0905\u0902\u0921\u093e, \u0922\u0915\u094d\u0915\u0928\u0964",
        ScriptCategory.PHONETIC_HI_DENTAL_RETROFLEX,
        "Tomato, three, then, fear, post, day, plate, egg, lid. (dental vs retroflex contrast)",
        "tamaatar, teen, tab, dar, daak, din, thaali, anda, dhakkan.",
        7.0,
    ),
    _hi(
        "hi-aspirate-01",
        "\u0916\u093e\u0928\u093e, \u0918\u0930, \u091b\u0924, \u091d\u0930\u0928\u093e, \u0927\u094b\u092c\u0940, \u092d\u093e\u0908, \u092b\u0932, \u0925\u094b\u095c\u093e, \u092e\u0947\u0930\u0947, \u0928\u0939\u0940\u0902\u0964",
        ScriptCategory.PHONETIC_HI_ASPIRATION,
        "Food, home, roof, waterfall, washer, brother, fruit, a little, mine, no. (aspirated vs plain)",
        "khaana, ghar, chhat, jharna, dhobi, bhai, phal, thoda, mere, nahin.",
        8.0,
    ),
    _hi(
        "hi-nasal-vowel-01",
        "\u0939\u0901\u0938\u0928\u093e, \u091a\u093e\u0901\u0926\u0940, \u092c\u093e\u0901\u0927, \u0938\u093e\u0925 \u092e\u0947\u0902, \u0915\u093e\u092e, \u092c\u093f\u0928, \u092a\u0942\u0930\u093e, \u0938\u094d\u0935\u093e\u0926\u0964",
        ScriptCategory.PHONETIC_HI_NASALS_VOWELS,
        "Laughter, silver, bind, together, work, without, whole, taste. (nasals and vowel length)",
        "hansna, chaandi, baandh, saath mein, kaam, bin, poora, swaad.",
        7.0,
    ),
    _hi(
        "hi-reassure",
        "\u091a\u093f\u0902\u0924\u093e \u092e\u0924 \u0915\u0930\u094b, \u0938\u092c \u0920\u0940\u0915 \u0939\u094b \u091c\u093e\u090f\u0917\u093e\u0964 \u092e\u0948\u0902 \u092f\u0939\u093e\u0902 \u0939\u0942\u0902 \u0928\u093e\u0964",
        ScriptCategory.EMOTION_REASSURANCE,
        "Don't worry, everything will be fine. I'm right here.",
        "chinta mat karo, sab theek ho jayega. main yahaan hoon na.",
    ),
    _hi(
        "hi-calm-explain",
        "\u092e\u0948\u0902 \u0906\u0930\u093e\u092e \u0938\u0947 \u0938\u092e\u091d\u093e\u0924\u093e \u0939\u0942\u0902: \u092a\u0939\u0932\u0947 \u092f\u0939, \u092b\u093f\u0930 \u0935\u0939, \u0914\u0930 \u0906\u0916\u093f\u0930 \u092e\u0947\u0902 \u092f\u0939 \u091b\u094b\u091f\u093e \u0938\u093e \u0915\u0926\u092e\u0964",
        ScriptCategory.EXPLANATION,
        "Let me explain calmly: first this, then that, and finally this small step.",
        "main aaram se samjhaata hoon: pehle ye, phir wah, aur aakhir mein ye chhota sa kadam.",
        8.0,
    ),
    _hi(
        "hi-short-vm",
        "\u0939\u093e\u0902, \u0920\u0940\u0915 \u0939\u0948\u0964 \u092a\u0939\u0941\u0901\u091a \u0917\u092f\u093e\u0964 \u092c\u093e\u0926 \u092e\u0947\u0902 \u092c\u093e\u0924 \u0915\u0930\u0924\u0947 \u0939\u0948\u0902\u0964",
        ScriptCategory.SHORT_VOICE_NOTE,
        "Yes, okay. I got it. Let's talk later.",
        "haan, theek hai. pahunch gaya. baad mein baat karte hain.",
        5.0,
    ),
    _hi(
        "hi-long-paragraph",
        "\u0915\u0932 \u0938\u0941\u092c\u0939 \u092e\u0948\u0902 \u091c\u0932\u094d\u0926\u0940 \u0909\u0920\u093e, \u091a\u093e\u092f \u092c\u0928\u093e\u0908, \u0914\u0930 \u0916\u093f\u095c\u0915\u0940 \u0915\u0947 \u092a\u093e\u0938 \u092c\u0948\u0920\u0915\u0930 \u0927\u0942\u092a \u092e\u0947\u0902 \u0915\u093f\u0924\u093e\u092c \u092a\u095d\u0940\u0964 "
        "\u0926\u094b\u092a\u0939\u0930 \u092e\u0947\u0902 \u0926\u094b\u0938\u094d\u0924\u094b\u0902 \u0938\u0947 \u092e\u093f\u0932\u093e, \u0936\u093e\u092e \u0915\u094b \u0918\u0941\u092e\u0928\u0947 \u0928\u093f\u0915\u0932\u0947, \u0914\u0930 \u0930\u093e\u0924 \u092e\u0947\u0902 \u0906\u0930\u093e\u092e \u0938\u0947 \u0916\u093e\u0928\u093e \u0916\u093e\u092f\u093e\u0964 "
        "\u0926\u093f\u0928 \u092c\u0939\u0941\u0924 \u0905\u091a\u094d\u091b\u093e \u0925\u093e, \u0925\u094b\u095c\u093e \u0925\u0915\u093e \u0926\u0947\u0928\u093e \u0932\u0947\u0915\u093f\u0928 \u0926\u093f\u0932 \u0916\u0941\u0936 \u0930\u0939\u093e\u0964",
        ScriptCategory.LONG_PARAGRAPH,
        "Next morning I woke early, made tea, and sat by the window reading in the sun. "
        "In the afternoon I met friends, went for an evening walk, and at night ate a "
        "relaxed dinner. The day was very good, a bit tiring but the heart stayed happy.",
        "kal subah main jaldi utha, chai banayi, aur khidki ke paas baithkar dhoop mein "
        "kitaab padhi. ...",
        18.0,
    ),
]

# ---------------------------------------------------------------------------
# B. English (starter set from the product spec + additions)
# ---------------------------------------------------------------------------
_ENGLISH_STARTER: list[ScriptEntry] = [
    _en(
        "en-clean-01",
        "Hello, my name is [NAME], and I am recording a sample of my voice.",
        ScriptCategory.CLEAN_REFERENCE,
        8.0,
    ),
    _en(
        "en-good-day",
        "Today is a good day, and I am speaking naturally and clearly.",
        ScriptCategory.STATEMENT,
    ),
    _en("en-question-day", "How are you? I hope your day is going well.", ScriptCategory.QUESTION),
    _en(
        "en-understand-more",
        "I understand what you mean, but I would like to think about it a little more.",
        ScriptCategory.EXPLANATION,
        7.0,
    ),
    _en(
        "en-what-next",
        "When you have some time, please let me know what we should do next.",
        ScriptCategory.QUESTION,
        7.0,
    ),
    _en(
        "en-at-home",
        "I am at home right now, and I will talk to you a little later.",
        ScriptCategory.STATEMENT,
    ),
    _en(
        "en-read-message",
        "Did you read that message? I need to reply to it.",
        ScriptCategory.QUESTION,
    ),
    _en(
        "en-small-detail",
        "Sometimes, even a small detail can make a big difference.",
        ScriptCategory.EMPHASIS_CONTRAST,
    ),
    _en(
        "en-first-then",
        "Let's understand the situation first and then decide what to do.",
        ScriptCategory.EXPLANATION,
        7.0,
    ),
    _en(
        "en-good-idea",
        "I think this is a good idea, although there are a few other things to consider.",
        ScriptCategory.EMPHASIS_CONTRAST,
        7.0,
    ),
    _en(
        "en-speak-slowly",
        "Please speak slowly enough for every word to be heard clearly.",
        ScriptCategory.EXPLANATION,
    ),
    _en(
        "en-both-languages",
        "I use both Hindi and English in my everyday conversations.",
        ScriptCategory.STATEMENT,
    ),
    _en(
        "en-numbers",
        "One, two, three, four, five, six, seven, eight, nine, ten.",
        ScriptCategory.NUMBERS,
    ),
    _en(
        "en-meeting-times",
        "The meeting starts at ten thirty in the morning and ends at four fifteen in the "
        "afternoon.",
        ScriptCategory.DATES_TIMES,
        8.0,
    ),
    _en(
        "en-honest",
        "I would rather be honest about what I know than guess an answer.",
        ScriptCategory.EMPHASIS_CONTRAST,
        7.0,
    ),
    _en(
        "en-thanks",
        "Thank you for listening. I will talk to you again soon.",
        ScriptCategory.ACKNOWLEDGEMENT,
    ),
]

_ENGLISH_ADDITIONAL: list[ScriptEntry] = [
    _en(
        "en-vowel-contrast",
        "Ship and sheep, sit and seat, full and fool, bed and bad, pull and pool.",
        ScriptCategory.PHONETIC_EN_VOWELS,
        8.0,
    ),
    _en(
        "en-clusters",
        "Street, spring, splash, twist, prompt, school, scream, threshold.",
        ScriptCategory.PHONETIC_EN_CLUSTERS,
        8.0,
    ),
    _en(
        "en-fricatives",
        "This zebra shops for fresh shoes, television, vision, and measure.",
        ScriptCategory.PHONETIC_EN_FRICATIVES,
        7.0,
    ),
    _en(
        "en-reassure",
        "Hey, don't stress about it. We'll figure it out together, one step at a time.",
        ScriptCategory.EMOTION_REASSURANCE,
        7.0,
    ),
    _en(
        "en-surprise",
        "Oh wow, I did not expect that at all! That's actually great news.",
        ScriptCategory.EMOTION_CALM,
        6.0,
    ),
    _en(
        "en-short-vm",
        "Yeah, sounds good. On my way. See you in five.",
        ScriptCategory.SHORT_VOICE_NOTE,
        5.0,
    ),
    _en(
        "en-long-paragraph",
        "On weekends I usually wake up late, make a strong coffee, and spend the first "
        "hour reading. Later I go out for groceries, sometimes meet a friend for lunch, "
        "and in the evening I cook something simple. It is a quiet routine, but it keeps "
        "the week balanced.",
        ScriptCategory.LONG_PARAGRAPH,
        18.0,
    ),
]

# ---------------------------------------------------------------------------
# C. Hinglish — natural code-switching
# ---------------------------------------------------------------------------
_HINGLISH_SCRIPTS: list[ScriptEntry] = [
    _he(
        "he-01",
        "Hi, kaise ho? Main abhi ghar par hoon aur thodi der mein call karunga.",
        "Hi, how are you? I'm at home and I'll call in a bit.",
    ),
    _he(
        "he-02",
        "Yaar, mujhe lagta hai ki humein is idea ke baare mein thoda aur sochna chahiye.",
        "Man, I think we should think about this idea a bit more.",
        ScriptCategory.CODE_SWITCH,
        7.0,
    ),
    _he(
        "he-03",
        "Actually, main tumhari baat samajh raha hoon, but mujhe ek cheez clear karni hai.",
        "Actually, I understand what you're saying, but I want to clarify one thing.",
        ScriptCategory.CODE_SWITCH,
        7.0,
    ),
    _he(
        "he-04",
        "Aaj ka plan kya hai? Agar time mile toh hum shaam ko mil sakte hain.",
        "What's the plan today? If we have time we can meet in the evening.",
    ),
    _he(
        "he-05",
        "Maine message dekh liya hai, bas abhi reply karne ka time nahi mila.",
        "I've seen the message, I just didn't get time to reply.",
    ),
    _he(
        "he-06",
        "Honestly, mujhe ye option kaafi achha lag raha hai, but let's compare both options first.",
        "Honestly, I'm liking this option quite a lot, but let's compare both first.",
        ScriptCategory.CODE_SWITCH,
        7.0,
    ),
    _he(
        "he-07",
        "Koi baat nahi, hum aaram se baat kar lenge. There is no need to rush.",
        "No problem, we'll sort it out calmly. There's no need to rush.",
        ScriptCategory.CODE_SWITCH,
        7.0,
    ),
    _he(
        "he-08",
        "Ek minute rukna, main abhi check karke batata hoon.",
        "Wait a minute, let me check and tell you.",
    ),
    _he(
        "he-09",
        "Haan, that makes sense. Ab dekhte hain ki next step kya hona chahiye.",
        "Yes, that makes sense. Now let's see what the next step should be.",
    ),
    _he(
        "he-10",
        "Mujhe exact time bata dena, so that I can plan everything properly.",
        "Tell me the exact time so I can plan everything properly.",
    ),
    _he(
        "he-numbers",
        "Ek, do, teen, chaar, paanch, chhah, saat, aath, nau, das.",
        "One, two, three, four, five, six, seven, eight, nine, ten.",
        ScriptCategory.NUMBERS,
    ),
    _he(
        "he-question-plan",
        "Kal ka kya scene hai? Tum free ho ya kuch kaam hai?",
        "What's the scene tomorrow? Are you free or do you have work?",
        ScriptCategory.QUESTION,
    ),
    _he(
        "he-question-msg",
        "Tumne woh message padha? Mujhe uska jawab dena hai urgently.",
        "Did you read that message? I need to reply to it urgently.",
        ScriptCategory.QUESTION,
    ),
    _he(
        "he-aspirate",
        "Khana kha liya? Ghar pe sab theek hai, chhota bhai just bahar gaya hai.",
        "Had food? All good at home, little brother has just gone out.",
        ScriptCategory.PHONETIC_HI_ASPIRATION,
        7.0,
    ),
    _he(
        "he-nasal",
        "Chandni raat mein long drive ka apna hi maza hai. Kaam kal karenge, that is a promise.",
        "A long drive on a moonlit night has its own charm. We'll work tomorrow, that is a promise.",
        ScriptCategory.PHONETIC_HI_NASALS_VOWELS,
        7.0,
    ),
    _he(
        "he-ack",
        "Haan haan, samajh gaya. Done, kal subah bhej dunga pakka.",
        "Yes yes, understood. Done, I'll definitely send it tomorrow morning.",
        ScriptCategory.ACKNOWLEDGEMENT,
    ),
    _he(
        "he-short-vm",
        "Haan, pahunch gaya. paanch minute mein aata hoon.",
        "Yes, got it. Coming in five minutes.",
        ScriptCategory.SHORT_VOICE_NOTE,
        5.0,
    ),
    _he(
        "he-reassure",
        "Tension mat lo yaar, sab sort ho jayega. Main hoon na tumhare saath, I promise.",
        "Don't stress buddy, everything will get sorted. I'm with you, I promise.",
        ScriptCategory.EMOTION_REASSURANCE,
        7.0,
    ),
    _he(
        "he-explain",
        "Dekho, pehle hum saari details nikaalte hain, phir aaram se decide karte hain.",
        "Look, first we pull out all the details, then decide calmly.",
        ScriptCategory.EXPLANATION,
        7.0,
    ),
    _he(
        "he-contrast",
        "Idea toh achha hai, but budget thoda zyada lag raha hai mujhe.",
        "The idea is good, but the budget feels a bit high to me.",
        ScriptCategory.EMPHASIS_CONTRAST,
        7.0,
    ),
    _he(
        "he-datetime",
        "Meeting kal dopahar teen baje hai, aur Friday ko chhutti hai.",
        "The meeting is tomorrow at 3pm, and Friday is a holiday.",
        ScriptCategory.DATES_TIMES,
    ),
    _he(
        "he-long-para",
        "Aaj subah main jaldi utha, chai banayi, aur balcony mein baithkar Mausam "
        "enjoy kiya. Dopahar mein doston se mila, shaam ko thoda walk kiya, aur "
        "raat mein ghar ka khana khaya. Din kaafi achha tha overall.",
        "This morning I woke early, made tea, and enjoyed the weather on the "
        "balcony. Met friends in the afternoon, walked in the evening, and ate "
        "home food at night. Overall a pretty good day.",
        ScriptCategory.LONG_PARAGRAPH,
        18.0,
    ),
    _he(
        "he-thanks",
        "Thank you yaar, tumne meri bahut help ki. Kal milte hain!",
        "Thank you buddy, you helped me a lot. See you tomorrow!",
        ScriptCategory.ACKNOWLEDGEMENT,
    ),
]

SCRIPT_LIBRARY: list[ScriptEntry] = [
    *_HINDI_STARTER,
    *_HINDI_ADDITIONAL,
    *_ENGLISH_STARTER,
    *_ENGLISH_ADDITIONAL,
    *_HINGLISH_SCRIPTS,
]

_BY_ID = {s.id: s for s in SCRIPT_LIBRARY}
if len(_BY_ID) != len(SCRIPT_LIBRARY):  # fail fast on duplicate ids
    raise RuntimeError("Duplicate script ids in library")


def all_scripts() -> list[ScriptEntry]:
    return list(SCRIPT_LIBRARY)


def get_script(script_id: str) -> ScriptEntry:
    if script_id not in _BY_ID:
        raise KeyError(f"Unknown script id {script_id!r}")
    return _BY_ID[script_id]


def scripts_by_language(language: str) -> list[ScriptEntry]:
    return [s for s in SCRIPT_LIBRARY if s.language.value == language]


def scripts_by_category(category: str) -> list[ScriptEntry]:
    return [s for s in SCRIPT_LIBRARY if s.category.value == category]


def render(entry: ScriptEntry, values: dict[str, str] | None = None) -> str:
    """Render placeholders. Missing values render as editable blanks ('______').

    We never fabricate a user's name, a date or a time; an unset placeholder
    becomes an explicit blank the user fills in.
    """
    values = values or {}
    out = entry.text
    for key in ("[NAME]", "[DATE]", "[TIME]"):
        val = values.get(key.strip("[]"))
        out = out.replace(key, val) if val else out.replace(key, "__________")
    return out


def total_estimated_seconds(entries: list[ScriptEntry]) -> float:
    return round(sum(e.est_seconds for e in entries), 1)
