SYMPTOM_KEYWORDS = {
    "psoriasis": [
        "scaling", "silvery", "plaques", "elbow", "knee", "itch", "red patches",
        "thick skin", "flaky"
    ],
    "eczema": [
        "itching", "dry", "red", "rash", "irritation", "cracked",
        "inflammation", "sensitive skin"
    ],
    "lichen_planus": [
        "purple", "polygonal", "papules", "itch", "rash", "flat bumps",
        "mouth sores"
    ],
    "dermatitis": [
        "redness", "irritation", "inflammation", "swelling", "tender"
    ],
    "fungal_infection": [
        "ring", "circular", "itch", "red border", "flaky", "ringworm"
    ],
    "urticaria": [
        "hives", "welts", "itching", "swelling", "allergic reaction"
    ],
    "melanoma": [
        "dark patch", "irregular border", "mole change", "bleeding", 
        "growing patch", "asymmetry"
    ]
}


def rough_predict(text: str, top_k=3):
    """
    Returns top_k possible diseases based on keyword matching.
    """
    text = text.lower()
    scores = {}

    for disease, keywords in SYMPTOM_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in text)
        scores[disease] = matches

    # Sort by match count
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_scores[:top_k]
