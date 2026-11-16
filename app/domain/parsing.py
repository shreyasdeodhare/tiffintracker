import re
import unicodedata
from dataclasses import dataclass

from .meal_items import MealItem, parse_meal_request


@dataclass(frozen=True)
class Interpretation:
    category: str
    confidence: str
    quantity: int | None
    reason: str
    day_offset: int = 0
    items: tuple[MealItem, ...] = ()
    descriptions: tuple[tuple[int, str], ...] = ()


NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "ek": 1,
    "do": 2,
    "teen": 3,
    "char": 4,
    "paanch": 5,
    "एक": 1,
    "दो": 2,
    "तीन": 3,
    "चार": 4,
    "पांच": 5,
    "पाँच": 5,
}
FOOD = r"(?:tiff[ia]ns?|tifins?|lunch(?:es)?|dinners?|dab[b]?as?|टिफिन|टिफ़िन|डब्बे|डब्बा|लंच|खाना)"


def classify(body: str) -> Interpretation:
    raw = unicodedata.normalize("NFC", body).strip().casefold()
    for word, value in NUMBERS.items():
        raw = re.sub(r"(?<!\w)" + re.escape(word) + r"(?!\w)", str(value), raw)
    text = re.sub(r"\s+", " ", raw)

    def answer(category, confidence="low", quantity=None, reason="", offset=0):
        return Interpretation(category, confidence, quantity, reason or category, offset)

    if re.search(
        r"\[media:|media omitted|attached:|\.opus\b|\.jpg\b|\.mp4\b|image omitted|audio omitted|मीडिया",
        text,
    ):
        return answer("media", "medium", reason="Media requires administrator interpretation")
    if re.search(r"\b(paid|payment|upi|gpay|transferred|received money)\b|भुगतान|पैसे", text):
        return answer(
            "payment", "medium", reason="Reconcile payment manually; no automatic payment entry"
        )
    if re.search(r"\b(holiday|holidays|closed|office off|no office)\b|छुट्टी|अवकाश", text):
        return answer("announcement", reason="Holiday announcement; no order")
    if re.search(r"deleted|edited|message was edited|संपादित", text):
        return answer(
            "ambiguous", "medium", reason="Export cannot reliably recover edit/deletion identity"
        )
    if "?" in text or re.search(r"\b(maybe|perhaps|anyone|who|if|unless|kal)\b|शायद|कल", text):
        return answer("ambiguous", "medium", reason="Question, condition or ambiguous date")
    offset = 0
    if re.search(r"\b(tomorrow|yesterday)\b", text):
        offset = 1 if "tomorrow" in text else -1
        text = re.sub(r"\b(?:for )?(tomorrow|yesterday)\b", "", text).strip()
    # Only simple, complete self-order phrases qualify for automation.
    text = re.sub(r"\b(?:for today|today)\b|आज", "", text).strip(" .!")
    if re.fullmatch(
        r"(?:please |pls |plz )?(?:cancel (?:my )?(?:lunch|dinner|tiffin|order)|mera (?:lunch|tiffin|order) cancel(?: karo)?|मेरी? (?:टिफिन|लंच|ऑर्डर) रद्द(?: करो)?)",
        text,
    ):
        return answer(
            "cancel",
            "high",
            reason="Explicit self cancellation; unique target required",
            offset=offset,
        )
    changed = re.fullmatch(
        r"(?:make it|change (?:it )?to|instead make it|अब|ab) (\d{1,2})(?: instead| tiffins?| टिफिन)?",
        text,
    )
    if changed:
        quantity = int(changed[1])
        return answer(
            "modify",
            "high" if 1 <= quantity <= 20 else "medium",
            quantity if 1 <= quantity <= 20 else None,
            "Explicit quantity replacement; unique target required",
            offset,
        )
    meal_text = re.sub(r"\b(?:for )?(?:today|tomorrow|yesterday)\b|आज", "", raw).strip()
    request = parse_meal_request(meal_text)
    if request is not None:
        return Interpretation(
            "order",
            "high",
            request.quantity,
            "Explicit meal quantities; dish descriptions are not extra meals",
            offset,
            request.items,
            request.descriptions,
        )
    order = re.fullmatch(
        rf"(?:(?:i need|i want|please|pls|plz|send|for me|mere liye|mujhe|मेरे लिए|मुझे) )?(\d{{1,2}})\s*{FOOD}(?: (?:for (?:lunch|dinner)|for me|please|pls|chahiye|bhej do|चाहिए))?",
        text,
    )
    if order:
        quantity = int(order[1])
        if 1 <= quantity <= 20:
            return answer("order", "high", quantity, "Explicit self quantity and meal", offset)
        return answer("ambiguous", "medium", reason="Quantity outside 1–20")
    if re.search(FOOD, text) or re.search(r"\b(cancel|instead|order|make it)\b|रद्द", text):
        return answer(
            "ambiguous",
            "medium",
            reason="Meal-related wording or multiple-person request requires review",
        )
    if re.fullmatch(
        r"(?:good (?:morning|evening|night)|thanks?(?: you)?|ok(?:ay)?|hello|hi|नमस्ते|धन्यवाद)[! .]*",
        text,
    ):
        return answer("conversation", reason="General conversation")
    return answer("ambiguous", "low", reason="Unrecognized wording; never assumed to be an order")
