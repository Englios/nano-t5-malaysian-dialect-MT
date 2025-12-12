DIALECT_PREFIXES = [
    "terjemah ke johor:",
    "terjemah ke kelantan:",
    "terjemah ke sabah:",
    "terjemah ke sarawak:",
    "terjemah ke kedah:",
    "terjemah ke pahang:",
    "terjemah ke perak:",
    "terjemah ke terengganu:",
    "terjemah ke melaka:",
    "terjemah ke negeri sembilan:",
]

STANDARD_PREFIXES = [
    "terjemah ke Melayu:",
    "terjemah ke Inggeris:",
    # "terjemah ke Mandarin:",
    # "terjemah ke Tamil:",
    "terjemah ke Manglish:",
    # "terjemah ke Cantonese:",
]

ALLOWED_PREFIXES = DIALECT_PREFIXES + STANDARD_PREFIXES

MODEL_NAME = "mesolitica/nanot5-small-malaysian-cased"