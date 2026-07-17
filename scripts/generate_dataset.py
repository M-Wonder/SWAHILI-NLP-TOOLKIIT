"""
Generate a small, original, synthetic Swahili news-headline dataset for
demonstration purposes.

The dataset is built from hand-written category-specific vocabulary banks
combined via templates, so every sentence is original text (not scraped or
copied from any publication) and can be freely redistributed with this
project.

Usage
-----
    python scripts/generate_dataset.py

This writes ``data/swahili_news_dataset.csv`` with columns ``text,category``.

Note
----
This is a *demonstration* dataset meant to showcase the pipeline end to
end. For a production system, replace it with a real, properly licensed
Swahili corpus (see README.md for suggested public datasets).
"""

from __future__ import annotations

import csv
import itertools
import random
from pathlib import path

random.seed(42)

SUBJECTS = {
    "siasa": [
        "Rais", "Waziri Mkuu", "Bunge", "Waziri wa Mambo ya Ndani",
        "Spika wa Bunge", "Gavana", "Kiongozi wa upinzani", "Baraza la Mawaziri",
        "Tume ya Uchaguzi", "Mbunge wa jimbo",
    ],
    "michezo": [
        "Timu ya taifa", "Kocha mkuu", "Klabu ya mpira", "Mchezaji nyota",
        "Shirikisho la mpira wa miguu", "Timu ya vijana", "Bingwa wa riadha",
        "Klabu ya Simba", "Klabu ya Yanga", "Mwanariadha",
    ],
    "biashara": [
        "Benki kuu", "Soko la hisa", "Kampuni kubwa", "Waziri wa Fedha",
        "Wafanyabiashara", "Shirika la kodi", "Sekta ya kilimo",
        "Wawekezaji wa kigeni", "Mfumo wa fedha", "Soko la bidhaa",
    ],
    "afya": [
        "Wizara ya Afya", "Hospitali kuu", "Madaktari", "Wataalamu wa afya",
        "Shirika la afya duniani", "Kituo cha afya", "Wagonjwa",
        "Wauguzi", "Idara ya chanjo", "Taasisi ya utafiti wa afya",
    ],
    "burudani": [
        "Msanii maarufu", "Kampuni ya filamu", "Mwigizaji", "Bendi ya muziki",
        "Tamasha la muziki", "Studio ya sanaa", "Mwandishi wa vitabu",
        "Kipindi cha televisheni", "Msanii wa filamu", "Kikundi cha ngoma",
    ], 
}

ACTIONS = {
    "siasa": [
        "ametangaza sera mpya ya elimu",
        "amefanya ziara ya kikazi mkoani",
        "ameunda kamati ya kuchunguza rushwa",
        "amesaini sheria mpya bungeni",
        "ametoa hotuba kuhusu maendeleo ya taifa",
        "amewaita viongozi kwa mkutano wa dharura",
        "ameahidi kuboresha miundombinu",
        "amepitisha bajeti ya serikali",
        "amezindua mradi wa maji vijijini",
        "ametoa tamko kuhusu usalama wa nchi",
    ],
    "michezo": [
        "imeshinda mechi ya kirafiki",
        "amevunja rekodi ya taifa",
        "wamejiandaa kwa fainali za bara",
        "imetangaza wachezaji wapya",
        "ameteuliwa kuwa nahodha wa timu",
        "imefunga msimu kwa ushindi mkubwa",
        "amepata jeraha kabla ya mechi kubwa",
        "wameanza maandalizi ya kombe la dunia",
        "imepanda daraja baada ya ushindi",
        "amepewa tuzo ya mchezaji bora",
    ],
    "biashara": [
        "imeongeza thamani ya sarafu",
        "imetangaza faida kubwa mwaka huu",
        "wameongeza bei ya mafuta",
        "imeidhinisha mkopo mpya wa maendeleo",
        "wamewekeza katika kilimo cha kisasa",
        "imepunguza riba ya benki",
        "wamefungua matawi mapya nchini",
        "imezindua huduma mpya za kidijitali",
        "imeripoti ukuaji wa uchumi",
        "wamepata mkataba mkubwa wa kimataifa",
    ],
    "afya": [
        "imeanzisha kampeni ya chanjo",
        "wamegundua matibabu mapya ya ugonjwa",
        "imeripoti ongezeko la wagonjwa",
        "wametoa tahadhari kuhusu mlipuko wa ugonjwa",
        "imeboresha huduma za dharura",
        "wamefanikiwa upasuaji mgumu",
        "imezindua kituo kipya cha afya",
        "wametoa elimu kuhusu lishe bora",
        "imepokea vifaa vipya vya kisasa",
        "wamesisitiza umuhimu wa uchunguzi wa mapema",
    ],
    "burudani": [
        "ametoa wimbo mpya wenye mafanikio",
        "ameshinda tuzo ya kimataifa",
        "wamezindua filamu mpya wiki hii",
        "atafanya tamasha jijini wiki ijayo",
        "amezinduliwa kama balozi wa utamaduni",
        "wamepanga tamasha kubwa la muziki",
        "ameandika kitabu kipya cha hadithi",
        "amerudi jukwaani baada ya mapumziko",
        "wameshiriki tamasha la kimataifa",
        "amepata mafanikio makubwa mtandaoni",
    ],
}

CONTEXT = {
     "wiki hii", "leo asubuhi", "jana jioni", "mwishoni mwa wiki",
    "mwezi huu", "katika mkutano wa hivi karibuni", "nchini Kenya",
    "nchini Tanzania", "mjini Nairobi", "mjini Dar es Salaam",
    "", "", "",  # allow some headlines without a trailing context
}

def build_dataset() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] =[]
    for category, subjects in SUBJECTS.items():
        actions = ACTIONS[category]
        combos = list(itertools.product(subjects, actions))
        random.shuffle(combos)
        for subject, action in combos[:55]:  # limit to 55 headlines per category
            context = random.choice(CONTEXT)
            headline = f"{subject} {action} {context}".strip()
            rows.append((headline, category))
    random.shuffle(rows)
    return rows

def main() -> None:
    rows = build_dataset()
    out_path = Path(__file__).resolve().parent.parent /"data"/"swahili_news_dataset.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
         writer = csv.writer(f)
         writer.writerow(["text", "category"])
         writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    counts: dict[str, int] = {}
    for _, cat in rows:
        counts[cat] =counts.get(cat, 0) + 1
        print("category distribution:", counts)

if __name__ == "__main__":
    main() 

