import csv

BG   = ("deep blue-black seamless background, cold blue rim light with "
        "energy-yellow accents, brushed steel, high contrast")
QUAL = ("shallow depth of field, cinematic 4K, hyperrealistic, "
        "vertical 9:16 portrait composition")
NEG  = ("blurry, distorted, low quality, watermark, text, logo, "
        "extra holes, irregular spacing, warped surface, deformed grid")

THICK_MIN = 20   # soglia: la clip [THICK] si attiva solo da 20mm in su

def grid(s):
    return ("fine precision Ø16 mm hole grid" if str(s) == "16"
            else "heavy-duty Ø28 mm hole grid")

def hero(r):
    return (f"wide shot, a {r['spessore']}mm thick S355 steel modular welding table "
            f"{r['misura']} with a {grid(r['sistema'])}, {r['portata']} kg capacity, "
            f"slow orbit shot around the subject, low angle, {BG}, {QUAL}.")

def macro(r):
    return (f"extreme close-up macro, the {grid(r['sistema'])} surface of a steel "
            f"welding table, rows of precision bore holes, smooth tracking shot, "
            f"overhead top-down, {BG}, {QUAL}.")

def spessore(r):
    return (f"extreme close-up macro, the thick machined edge of a {r['spessore']}mm "
            f"S355 steel welding table top, showing structural mass and robustness, "
            f"slow pan right, eye level, {BG}, {QUAL}.")

with open("tavoli.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        print(f"### {r['codice']} — Ø{r['sistema']} · {r['misura']} · "
              f"{r['spessore']}mm · {r['portata']}kg")
        print(f"[HERO]  {hero(r)}")
        print(f"[MACRO] {macro(r)}")
        if int(r['spessore']) >= THICK_MIN:
            print(f"[THICK] {spessore(r)}")   # solo tavoli pesanti
        print(f"[NEG]   {NEG}\n")
