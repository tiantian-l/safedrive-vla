"""E19a: the [28] red-light and [29] speed sentence libraries must not reproduce any evaluation instruction.

Collects every <text> of the route XMLs in benchmark/data (carla_f, b2d_c, b2d_adv, bench2drive) and the
track3 routes folder (if given) plus the agent's instruction tables, normalizes (lowercase, strip punctuation,
collapse spaces), expands each template for speeds 1..30 (whole and .5) and reports exact duplicates.
Also prints near matches (same text once the number is removed) for information. Exit code 1 on a duplicate.

    python3 scripts/data/check_sentences.py [--track3-routes ../track3/routes]
"""
import argparse, glob, json, re, sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def norm(t):
    t = re.sub(r"[^\w\s./]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip(" .")


def eval_texts(track3):
    texts = set()
    dirs = [REPO / "benchmark" / "data"] + ([Path(track3)] if track3 else [])
    for d in dirs:
        for f in glob.glob(f"{d}/**/*.xml", recursive=True):
            try:
                for e in ET.parse(f).iter("text"):
                    if e.text:
                        texts.add(e.text)
            except ET.ParseError:
                pass
    for t in ("instruction_meta_commands.json", "conflict_instruction_meta_commands.json"):
        p = REPO / "team_code" / t
        if p.exists():
            texts.update(json.load(open(p)))
    return texts


def speeds():
    return [str(v) for v in range(1, 31)] + [f"{v}.5" for v in range(0, 30)]


def expand(lib):
    out = []
    if "templates" in lib:
        for t in lib["templates"]:
            out += [t.format(v=v) for v in speeds()]
    for key in ("generic", "S", "L", "R"):
        for t in lib.get(key, []):
            out += [t.format(v=v) for v in speeds()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track3-routes", default="")
    a = ap.parse_args()
    ev = eval_texts(a.track3_routes)
    evn = {norm(t) for t in ev}
    nonum = lambda t: re.sub(r"\d+(\.\d+)?", "#", norm(t))
    ev_nonum = {nonum(t): t for t in ev}
    print(f"{len(ev)} evaluation instruction texts")
    bad = 0
    for name in ("speed_sentences.json", "red_light_sentences.json"):
        lib = json.load(open(REPO / "safedrive_vla" / "assets" / name))
        sents = expand(lib)
        dups = sorted({s for s in sents if norm(s) in evn})
        near = sorted({ev_nonum[nonum(s)] for s in sents if nonum(s) in ev_nonum})
        print(f"{name}: {len(sents)} expanded sentences, exact duplicates {len(dups)}, same-except-number {len(near)}")
        for s in dups:
            print("   DUP ", s)
        for s in near[:20]:
            print("   near", s)
        bad += len(dups)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
