# reprocess_scent_emotion.py
# Purpose: Applies extended scent & emotion lexicons to extract keywords.
# Input: YouTube_Perfume_Reviews_FullyFormatted.csv
# Output: YouTube_Perfume_Reviews_With_ScentEmotion.csv
# Execution Order: Optional (if enriching transcripts with scent/emotion labels)

import pandas as pd
import re

# Load cleaned transcripts
df = pd.read_csv("csv files/YouTube_Perfume_Reviews_FullyFormatted.csv")

# --- Re-clean Final Transcript Column ---
def fix_transcript_format(text):
    text = str(text)
    text = re.sub(r'\s{3,}', ' ', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

df['Final Transcript'] = df['Final Transcript'].apply(fix_transcript_format)

# (Optional) tokenization if needed later
df['Tokenized'] = df['Final Transcript'].apply(lambda x: re.findall(r'\b\w+\b', str(x).lower()))

# ✅ Expanded scent & emotion terms (unchanged content)
scent_terms = {
    # Core Families
    "amber", "vanilla", "woody", "citrus", "fresh", "rose", "patchouli", "spicy", "musky",
    "floral", "fruity", "green", "powdery", "smoky", "leathery", "balsamic", "resinous",
    "aldehydic", "aromatic", "animalic", "earthy", "herbal", "ozonic", "aquatic", "gourmand",

    # Floral Subtypes
    "indolic", "heady", "narcotic", "dewy", "watery", "buttery", "solar", "white floral",
    "yellow floral", "blue floral", "sweet floral", "green floral", "metallic floral",
    "opulent", "lush", "succulent", "crisp floral", "transparent floral",

    # Specific Florals
    "jasmine", "tuberose", "ylang ylang", "lily", "lily of the valley", "freesia", "violet",
    "iris", "gardenia", "narcissus", "hyacinth", "honeysuckle", "orange blossom", "osmanthus",
    "lavender", "mimosa", "chamomile", "hibiscus", "carnation", "peony", "lotus", "frangipani",
    "petals", "stemmy", "leafy", "pollen", "nectarous",

    # Woods & Moss
    "cedar", "sandalwood", "oud", "vetiver", "cypress", "pine", "oakmoss", "guaiacwood",
    "rosewood", "cashmeran", "palisander", "amberwood", "iso e super", "bark", "earth", "roots",
    "smoked wood", "burnt wood", "wet wood", "forest", "mossy", "dry wood", "resinous wood",

    # Animalics
    "indole", "civet", "castoreum", "ambergris", "musk", "hyraceum", "leather", "suede", "fur",
    "skin-like", "sweaty", "barnyard", "sensual", "intimate", "salty skin", "raunchy", "dirty",
    "raw", "fleshy", "pheromonal",

    # Resins & Incense
    "myrrh", "frankincense", "labdanum", "benzoin", "opoponax", "elemi", "styrax", "copal",
    "peru balsam", "tolu balsam", "storax", "incense", "church incense", "temple smoke",

    # Citruses
    "bergamot", "orange", "lemon", "lime", "grapefruit", "mandarin", "tangerine", "yuzu",
    "kumquat", "pomelo", "citron", "lemon zest", "peel", "pithy", "acidic", "sour", "sparkling",

    # Spices
    "cardamom", "clove", "nutmeg", "pepper", "pink pepper", "cinnamon", "saffron", "ginger",
    "coriander", "anise", "fennel", "cumin", "allspice", "pimento", "turmeric", "bay leaf",
    "herbaceous", "metallic spice", "hot spice", "cool spice",

    # Gourmand
    "honey", "chocolate", "toffee", "caramel", "praline", "milk", "cream", "butter", "sugar",
    "brown sugar", "maple", "marzipan", "chestnut", "hazelnut", "pistachio", "biscuit",
    "pastry", "dough", "coffee", "café latte", "mocha", "tonka bean", "coumarin", "bread",
    "yeast", "rum", "boozy", "liqueur", "whiskey", "brandy",

    # Fruity
    "apple", "pear", "peach", "plum", "grape", "blackcurrant", "raspberry", "strawberry",
    "cherry", "fig", "pomegranate", "melon", "banana", "apricot", "lychee", "dragonfruit",
    "passionfruit", "tropical fruit", "juicy", "syrupy", "candied", "jammy", "cassis",

    # Marine & Atmospheric
    "sea salt", "ocean breeze", "ozone", "mineral", "chalk", "stone", "rain", "wet earth",
    "petrichor", "cool air", "metallic", "urban", "concrete", "steel", "ice", "arctic", "snow",

    # Smoky
    "charred", "ash", "tar", "birch tar", "campfire", "burnt", "smoldering", "coal",
    "gunpowder", "singe", "cinders", "embers",

    # Abstract
    "clean linen", "silky", "velvety", "powdery", "creamy", "crispy", "juicy", "sweetened",
    "dry", "wet", "airy", "translucent", "dense", "ethereal", "heavy", "transparent", "thick",
    "cloying", "light", "fluffy", "sharp", "rounded", "metallic", "green tea", "maté", "hay",
    "ambery", "crystalline", "crunchy",

    # Synthetic & Modern
    "iso e super", "ambroxan", "cashmeran", "timbersilk", "calone", "aldehydes", "norlimbanol",
    "vetyverol", "vertiver acetate", "geosmin", "rose oxide", "vert de fleur", "javanol",
    "exaltolide", "muscenone", "serenolide"
}

emotion_terms = {
    # Passion
    "love", "adore", "worship", "obsessed", "devoted", "intoxicated", "infatuated", "captivated",
    "mesmerized", "seduced", "passion", "burning", "soulful", "spellbound",

    # Beauty
    "beautiful", "sublime", "exquisite", "magnificent", "elegant", "regal", "majestic",
    "graceful", "luxurious", "divine", "heavenly", "angelic", "ethereal", "breathtaking",

    # Joy
    "joyful", "cheerful", "uplifting", "playful", "sparkling", "radiant", "delightful",
    "vivacious", "sunny", "luminous", "bright", "fresh", "youthful", "effervescent",
    "elated", "happy",

    # Peace
    "calming", "soothing", "relaxing", "serene", "tranquil", "peaceful", "cozy", "gentle",
    "soft", "meditative", "zen", "comforting", "harmonious", "whispering",

    # Dark
    "mysterious", "enigmatic", "dark", "brooding", "moody", "haunting", "melancholic",
    "deep", "sultry", "intense", "hypnotic", "noir", "gothic", "shadowy", "voluptuous",

    # Nostalgia
    "nostalgic", "sentimental", "timeless", "vintage", "classic", "old-school", "familiar",
    "heartwarming", "homey", "reminiscent", "childhood", "memory-laden", "retro",

    # Uniqueness
    "exotic", "rare", "unusual", "eccentric", "avant-garde", "whimsical", "curious", "quirky",
    "adventurous", "daring", "bold", "unexpected", "surprising", "innovative", "visionary",

    # Strength
    "strong", "bold", "assertive", "confident", "empowering", "fearless", "intrepid",
    "commanding", "dominant", "decisive", "striking", "ferocious", "heroic", "vigorous",
    "relentless",

    # Critique
    "hate", "disappointed", "repulsive", "cloying", "overwhelming", "harsh", "chemical",
    "synthetic", "awkward", "too sweet", "too sharp", "boring", "flat", "uninspired", "weird"
}

# Build regex giving precedence to longer phrases
def build_pattern(terms):
    ordered = sorted(terms, key=len, reverse=True)
    return re.compile(r'\b(?:' + '|'.join(re.escape(t) for t in ordered) + r')\b', re.IGNORECASE)

scent_pattern = build_pattern(scent_terms)
emotion_pattern = build_pattern(emotion_terms)

# Processing function
def extract_terms(text, pattern):
    matches = pattern.findall(text)
    return list(set(m.lower() for m in matches)) if matches else []

# ✅ Use the CLEANED column for extraction
df["Scent Terms Found"] = df["Final Transcript"].fillna("").apply(lambda x: extract_terms(x, scent_pattern))
df["Emotion Terms Found"] = df["Final Transcript"].fillna("").apply(lambda x: extract_terms(x, emotion_pattern))

# Save enriched dataset
df.to_csv("csv files/YouTube_Perfume_Reviews_With_ScentEmotion.csv", index=False)
print("✅ Reprocessed data saved to YouTube_Perfume_Reviews_With_ScentEmotion.csv")

