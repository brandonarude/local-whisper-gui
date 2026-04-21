"""Static data referenced across the UI and core: model catalogue,
language list, chunking defaults. VRAM hints are rough float16 estimates
surfaced next to the model dropdown (SPEC §3.4)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    name: str
    vram_hint_gb: float
    description: str


# Order drives the dropdown in the Settings panel.
MODELS: tuple[ModelInfo, ...] = (
    ModelInfo("tiny", 0.5, "Fastest, lowest accuracy"),
    ModelInfo("base", 1.0, "Fast, low accuracy"),
    ModelInfo("small", 2.0, "Balanced speed and accuracy"),
    ModelInfo("medium", 5.0, "Good accuracy, moderate speed"),
    ModelInfo("large-v2", 10.0, "High accuracy (previous generation)"),
    ModelInfo("large-v3", 10.0, "Highest accuracy"),
    ModelInfo("distil-large-v3", 6.0, "Distilled large-v3, ~2x faster"),
)

DEFAULT_MODEL = "large-v3"


AUTO_DETECT_LANGUAGE = "auto"

# Whisper's supported languages. Key is the ISO code faster-whisper expects;
# value is the display name. `auto` is our sentinel for language detection.
LANGUAGES: dict[str, str] = {
    "auto": "Auto-detect",
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "ca": "Catalan",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "id": "Indonesian",
    "hi": "Hindi",
    "fi": "Finnish",
    "vi": "Vietnamese",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "th": "Thai",
    "ur": "Urdu",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "la": "Latin",
    "mi": "Maori",
    "ml": "Malayalam",
    "cy": "Welsh",
    "sk": "Slovak",
    "te": "Telugu",
    "fa": "Persian",
    "lv": "Latvian",
    "bn": "Bengali",
    "sr": "Serbian",
    "az": "Azerbaijani",
    "sl": "Slovenian",
    "kn": "Kannada",
    "et": "Estonian",
    "mk": "Macedonian",
    "br": "Breton",
    "eu": "Basque",
    "is": "Icelandic",
    "hy": "Armenian",
    "ne": "Nepali",
    "mn": "Mongolian",
    "bs": "Bosnian",
    "kk": "Kazakh",
    "sq": "Albanian",
    "sw": "Swahili",
    "gl": "Galician",
    "mr": "Marathi",
    "pa": "Punjabi",
    "si": "Sinhala",
    "km": "Khmer",
    "sn": "Shona",
    "yo": "Yoruba",
    "so": "Somali",
    "af": "Afrikaans",
    "oc": "Occitan",
    "ka": "Georgian",
    "be": "Belarusian",
    "tg": "Tajik",
    "sd": "Sindhi",
    "gu": "Gujarati",
    "am": "Amharic",
    "yi": "Yiddish",
    "lo": "Lao",
    "uz": "Uzbek",
    "fo": "Faroese",
    "ht": "Haitian Creole",
    "ps": "Pashto",
    "tk": "Turkmen",
    "nn": "Nynorsk",
    "mt": "Maltese",
    "sa": "Sanskrit",
    "lb": "Luxembourgish",
    "my": "Burmese",
    "bo": "Tibetan",
    "tl": "Tagalog",
    "mg": "Malagasy",
    "as": "Assamese",
    "tt": "Tatar",
    "haw": "Hawaiian",
    "ln": "Lingala",
    "ha": "Hausa",
    "ba": "Bashkir",
    "jw": "Javanese",
    "su": "Sundanese",
    "yue": "Cantonese",
}

DEFAULT_LANGUAGE = "en"


# Chunking defaults — SPEC §3.3 and §6.
DEFAULT_MIN_SILENCE_MS: int = 700
DEFAULT_SILENCE_THRESHOLD_DBFS: int = -40
DEFAULT_MIN_CHUNK_MINUTES: int = 5
DEFAULT_MAX_CHUNK_MINUTES: int = 30
CHUNK_OVERLAP_SECONDS: float = 2.0
LONG_FILE_PROMPT_THRESHOLD_SECONDS: int = 60 * 60  # 1 hour (SPEC §3.3)


# Output formats the exporter writes (SPEC §3.7).
OUTPUT_FORMATS: tuple[str, ...] = ("txt", "srt", "vtt", "json")
DEFAULT_OUTPUT_FORMATS: tuple[str, ...] = ("txt", "srt")

# Plain-text timestamp cadence (issue #5): insert a ``[M:SS]`` marker
# every N seconds of audio. The settings spinbox is reckoned in seconds
# to match the expected range (a few seconds up to a couple of minutes);
# longer cadences than the loaded audio trigger a UI warning.
DEFAULT_TIMESTAMP_CADENCE_S: int = 30
MIN_TIMESTAMP_CADENCE_S: int = 5
MAX_TIMESTAMP_CADENCE_S: int = 600
