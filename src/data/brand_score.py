"""Brand suitability score: exact formula, reproducibility, sensitivity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SuitabilityWeights:
    """Default weights from Phase 1 lightweight_brand_metrics."""

    inbound_cap: float = 0.25
    pairs_cap: float = 0.30
    coverage: float = 0.15
    lexical: float = 0.15
    golden_gate: float = 0.15
    # Soft penalties (subtracted), not weights that sum with the above.
    penalty_gt_100k: float = 0.08
    penalty_gt_80k: float = 0.04
    cap_n: float = 5000.0


DEFAULT_WEIGHTS = SuitabilityWeights()


def compute_suitability_score(
    *,
    inbound_count: int,
    n_pairs: int,
    coverage: float,
    lexical: float,
    outbound_n: int,
    weights: SuitabilityWeights = DEFAULT_WEIGHTS,
) -> dict[str, Any]:
    """Exact Phase-1 suitability formula.

    score =
        w_in  * min(inbound, CAP)/CAP
      + w_pr  * min(pairs, CAP)/CAP
      + w_cov * min(coverage, 1.0)          # coverage clipped to 1.0 for scoring
      + w_lex * min(lexical, 1.0)
      + w_g   * 1[inbound>=250 and pairs>=200]
      - mega_brand_penalty

    mega_brand_penalty:
      0.08 if outbound > 100_000
      0.04 if outbound > 80_000
      else 0

    Rationale:
    - inbound/pairs: enough volume for taxonomy + retrieval, capped so mega-brands
      do not dominate purely by size.
    - coverage: reply linkage quality (clipped because @mention undercount can
      make raw coverage > 1).
    - lexical: issue diversity proxy (unique texts / inbound).
    - golden_gate: binary feasibility for 150–250 labeled examples.
    - mega-brand penalty: keep taxonomy focused for a take-home (6–12 intents).
    """
    golden_ok = inbound_count >= 250 and n_pairs >= 200
    if outbound_n > 100_000:
        penalty = weights.penalty_gt_100k
    elif outbound_n > 80_000:
        penalty = weights.penalty_gt_80k
    else:
        penalty = 0.0

    components = {
        "inbound_term": weights.inbound_cap * min(inbound_count, weights.cap_n) / weights.cap_n,
        "pairs_term": weights.pairs_cap * min(n_pairs, weights.cap_n) / weights.cap_n,
        "coverage_term": weights.coverage * min(float(coverage), 1.0),
        "lexical_term": weights.lexical * min(float(lexical), 1.0),
        "golden_gate_term": weights.golden_gate * (1.0 if golden_ok else 0.0),
        "volume_penalty": penalty,
    }
    score = (
        components["inbound_term"]
        + components["pairs_term"]
        + components["coverage_term"]
        + components["lexical_term"]
        + components["golden_gate_term"]
        - components["volume_penalty"]
    )
    return {
        "suitability_score": round(float(score), 6),
        "components": {k: round(float(v), 6) for k, v in components.items()},
        "golden_ok": golden_ok,
        "weights": weights.__dict__,
    }


def formula_documentation() -> str:
    return """\
## Suitability score formula (OBSERVED / IMPLEMENTED)

```
score =
  0.25 * min(inbound_mentions, 5000) / 5000
+ 0.30 * min(unique_reply_pairs, 5000) / 5000
+ 0.15 * min(coverage, 1.0)
+ 0.15 * min(lexical_diversity, 1.0)
+ 0.15 * 1[inbound>=250 AND pairs>=200]
- mega_brand_penalty
```

### Components
| component | weight | normalization | rationale |
|---|---|---|---|
| inbound @brand mentions | 0.25 | min(n,5000)/5000 | enough customer volume |
| unique reply pairs | 0.30 | min(n,5000)/5000 | retrieval + supervision fuel |
| response coverage | 0.15 | clip to [0,1] | linkage quality |
| lexical diversity | 0.15 | unique/inbound clipped to 1 | issue variety proxy |
| golden feasibility gate | 0.15 | binary | can support 150–250 labels |
| mega-brand penalty | −0.04 / −0.08 | outbound>80k / >100k | avoid Amazon-scale taxonomy sprawl |

Coverage may be >1 vs @mentions when brand replies to customers without a literal
@brand string; scoring clips coverage at 1.0.
"""
