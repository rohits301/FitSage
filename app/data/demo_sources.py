"""Small, traceable corpus used only for offline demonstrations.

Production ingestion should replace this with versioned source chunks and preserve
the same metadata fields. The summaries below are deliberately short so reviewers
can inspect every answer used in demo mode.
"""

SOURCES = [
    {
        "id": "nih-vitamin-d",
        "organization": "NIH Office of Dietary Supplements",
        "title": "Vitamin D — Fact Sheet for Health Professionals",
        "url": "https://ods.od.nih.gov/factsheets/VitaminD-HealthProfessional/",
        "text": (
            "For adults ages 19–70 years, the recommended dietary allowance for "
            "vitamin D is 15 mcg (600 IU) daily. Needs and safe intake limits can "
            "differ with age, health conditions, medicines, and clinician guidance."
        ),
        "topics": {"vitamin", "d", "sun", "supplement", "bone", "iu", "mcg"},
    },
    {
        "id": "usda-myplate",
        "organization": "USDA MyPlate",
        "title": "What Is MyPlate?",
        "url": "https://www.myplate.gov/eat-healthy/what-is-myplate",
        "text": (
            "MyPlate is a visual guide for building balanced meals. It emphasizes "
            "making half the plate fruits and vegetables, including grains and "
            "protein foods, and choosing nutrient-dense options that fit preferences "
            "and budget."
        ),
        "topics": {"meal", "plate", "vegetable", "fruit", "diet", "balanced", "eat"},
    },
    {
        "id": "pubmed-protein-training",
        "organization": "PubMed",
        "title": "Protein supplementation and resistance training in healthy adults",
        "url": "https://pubmed.ncbi.nlm.nih.gov/28698222/",
        "text": (
            "A systematic review and meta-analysis examined protein supplementation "
            "alongside resistance training in healthy adults. It reported that benefits "
            "to resistance-training adaptations were not unlimited and varied by total "
            "protein intake, training, and participant characteristics."
        ),
        "topics": {"protein", "muscle", "strength", "lifting", "resistance", "workout", "gym"},
    },
    {
        "id": "nih-magnesium",
        "organization": "NIH Office of Dietary Supplements",
        "title": "Magnesium — Fact Sheet for Consumers",
        "url": "https://ods.od.nih.gov/factsheets/Magnesium-Consumer/",
        "text": (
            "Magnesium is involved in many body processes. Food sources include "
            "legumes, nuts, seeds, whole grains, and leafy green vegetables. People "
            "considering supplements should account for medicines and health conditions."
        ),
        "topics": {"magnesium", "cramp", "cramps", "supplement", "sleep", "nuts", "greens"},
    },
]

