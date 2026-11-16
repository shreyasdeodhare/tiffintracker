"""Explicit meal lines and prices; dish descriptions never create extra meals."""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .orders import BusinessError, validate_quantity

FOOD = r"(?:tiff[ia]ns?|tifins?|tififins?|lunch(?:es)?|dinners?|dab[b]?as?|टिफिन|टिफ़िन|डब्बे|डब्बा|लंच|खाना)"
LINE = re.compile(
    rf"(?:(?:i need|i want|please|pls|plz|send|for me|mere liye|mujhe|मेरे लिए|मुझे)\s+)?"
    rf"(?P<quantity>\d{{1,2}})\s*(?:(?P<portion>full|half)\s+)?{FOOD}"
    r"(?:\s+(?:for\s+)?(?P<meal>lunch|dinner))?"
    r"(?:\s*[-:]?\s*option\s*[-:]?\s*(?P<option>[ab12]))?"
    r"(?:\s+(?:for me|please|pls|chahiye|bhej do|चाहिए))?",
    re.IGNORECASE,
)
DISH = re.compile(r"(?P<quantity>\d{1,2})\s+(?P<dish>rajma|methi\s+aloo)\Z", re.IGNORECASE)
SEPARATOR = re.compile(r"\s*(?:\+|,|;|\n|\band\b)\s*", re.IGNORECASE)


@dataclass(frozen=True)
class MealItem:
    quantity: int
    portion: str = "standard"
    meal: str = "unspecified"
    option: str | None = None


@dataclass(frozen=True)
class MealRequest:
    items: tuple[MealItem, ...]
    descriptions: tuple[tuple[int, str], ...] = ()

    @property
    def quantity(self):
        return sum(item.quantity for item in self.items)


def parse_meal_request(text):
    """Require complete consumption; unknown text, people or uncertain layouts stay in review."""
    text = text.strip(" .!\n\r")
    items, descriptions = [], []
    for line in SEPARATOR.split(text):
        line = line.strip(" .!\r")
        if not line:
            return None
        dish = DISH.fullmatch(line)
        if dish:
            descriptions.append((int(dish["quantity"]), dish["dish"].casefold()))
            continue
        # Space-separated explicit meal lines are accepted only if every character matches.
        position = 0
        while position < len(line):
            match = LINE.match(line, position)
            if not match:
                return None
            items.append(
                MealItem(
                    int(match["quantity"]),
                    (match["portion"] or "standard").lower(),
                    match["meal"] or "unspecified",
                    match["option"],
                )
            )
            position = match.end()
            if position < len(line):
                if not line[position].isspace():
                    return None
                position += len(line[position:]) - len(line[position:].lstrip())
    if not items:
        return None
    # An explicit first-line total followed by a complete option/portion breakdown
    # describes the same meals, not an additional set of meals.
    first_line = LINE.fullmatch(text.splitlines()[0].strip(" .!\r"))
    if first_line and len(items) > 1 and items[0].option is None:
        rest = items[1:]
        if items[0].quantity == sum(item.quantity for item in rest) and (
            (
                all(item.option is not None for item in rest)
                and all(item.portion == items[0].portion for item in rest)
            )
            or (items[0].portion == "standard" and all(item.portion != "standard" for item in rest))
        ):
            items = rest
    total = sum(item.quantity for item in items)
    if not 1 <= total <= 20 or any(not 1 <= item.quantity <= 20 for item in items):
        return None
    if descriptions and (
        any(not 1 <= quantity <= 20 for quantity, _ in descriptions)
        or sum(quantity for quantity, _ in descriptions) != total
        or any(item.portion == "standard" for item in items)
    ):
        return None
    # Leave the original single-line parser/price semantics intact for existing wording.
    if (
        len(items) == 1
        and items[0].portion == "standard"
        and not descriptions
        and not items[0].option
    ):
        return None
    return MealRequest(tuple(items), tuple(descriptions))


def portion_price(portion, settings):
    key = {
        "standard": "TIFFIN_PRICE",
        "full": "FULL_TIFFIN_PRICE",
        "half": "HALF_TIFFIN_PRICE",
    }.get(portion)
    if key is None:
        raise BusinessError("Choose standard, full or half tiffin")
    defaults = {"FULL_TIFFIN_PRICE": "80.00", "HALF_TIFFIN_PRICE": "60.00"}
    try:
        price = Decimal(str(settings.get(key, defaults.get(key))))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BusinessError("Invalid tiffin price") from exc
    if not price.is_finite() or price <= 0 or price.as_tuple().exponent < -2:
        raise BusinessError("Tiffin prices must be positive with at most two decimals")
    return price.quantize(Decimal("0.01"))


def grouped_items(items):
    groups = {}
    for item in items:
        validate_quantity(item.quantity)
        key = (item.portion, item.meal)
        groups.setdefault(key, {"quantity": 0, "options": []})
        groups[key]["quantity"] += item.quantity
        if item.option:
            groups[key]["options"].append(
                {"option": item.option.upper(), "quantity": item.quantity}
            )
    validate_quantity(sum(group["quantity"] for group in groups.values()))
    return groups
