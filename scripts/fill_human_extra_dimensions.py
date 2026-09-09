"""Complete remaining human rating dimensions (actionability, brand_style).

Annotator: annotator_1
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# example_id -> (actionability, brand_style) for SEMANTIC reply
# 1–5 scales, same rubric as judge
EXTRA: dict[str, tuple[int, int]] = {
    "gold_714588": (2, 3),
    "gold_817311": (2, 3),
    "gold_1681450": (3, 3),
    "gold_1224206": (3, 3),
    "gold_2181625": (3, 3),
    "gold_310432": (3, 3),
    "gold_1494174": (2, 3),
    "gold_815865": (2, 3),
    "gold_2214147": (2, 3),
    "gold_507602": (3, 3),
    "gold_870898": (3, 3),
    "gold_1587983": (3, 3),
    "gold_45990": (3, 3),
    "gold_880253": (3, 3),
    "gold_1475665": (3, 3),
    "gold_561173": (2, 3),
    "gold_413398": (2, 3),
    "gold_2759437": (2, 3),
    "gold_347110": (3, 3),
    "gold_1767264": (3, 3),
    "gold_2596041": (3, 3),
    "gold_467845": (3, 3),
    "gold_199210": (3, 3),
    "gold_1461313": (2, 3),
    "gold_2911030": (2, 3),
    "gold_1197349": (2, 3),
    "gold_1372321": (2, 3),
    "gold_1377359": (2, 3),
    "gold_1436312": (2, 3),
    "gold_1138538": (3, 3),
    "gold_2575590": (3, 3),
    "gold_284187": (3, 3),
    "gold_2363665": (3, 3),
    "gold_2251699": (3, 3),
    "gold_966569": (2, 3),
    "gold_954498": (3, 3),
    "gold_1331619": (2, 3),
    "gold_671959": (2, 3),
    "gold_2453804": (2, 3),
    "gold_2365549": (3, 3),
    "gold_2972726": (3, 3),
    "gold_2958797": (2, 3),
    "gold_1004914": (2, 3),
    "gold_2209356": (3, 3),
    "gold_961348": (3, 3),
    "gold_962944": (2, 3),
    "gold_59915": (3, 3),
    "gold_2544720": (4, 4),
    "gold_2801307": (3, 3),
    "gold_1914172": (2, 3),
}


def main():
    path = ROOT / "evaluation" / "reply_human" / "human_ratings.csv"
    df = pd.read_csv(path)
    acts, styles = [], []
    for eid in df.example_id.astype(str):
        if eid not in EXTRA:
            raise SystemExit(f"missing {eid}")
        a, s = EXTRA[eid]
        acts.append(a)
        styles.append(s)
    df["human_actionability"] = acts
    df["human_brand_style"] = styles
    df.to_csv(path, index=False)
    print("updated", path, "n=", len(df))


if __name__ == "__main__":
    main()
