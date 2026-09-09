# Top 5 failure modes (frozen golden N=199)

Source: `reports/final/metric_audit_recompute.json` + `reports/phase2/failure_analysis.json`.

**Overlap:** flag categories **co-occur**. Counts are **not mutually exclusive**. An example can contribute to multiple modes.

## 1. Escalation disagreement — 91 / 199 (45.7%)
Includes **both** directions:
- **Over-escalation (FP):** gold auto, pred escalate = **85**
- **Under-escalation (FN):** gold escalate, pred auto = **6**

Primary safety-relevant subtype for UX/volume is over-escalation (**85**).  
**Examples (over):** `gold_2911030`, `gold_2721090`.  
**Cause:** safety-first policy.  
**Mitigation:** intent-specific auto-handle bands.

## 2. Intent confusion — 61 / 199 (30.7%)
pred intent ≠ gold intent.  
**Examples:** `gold_1372321` (playback↔live_tv), `gold_2623567`.  
**Cause:** lexical overlap.  
**Mitigation:** hierarchical intents / features.

## 3. Rare intents — 34 / 199 (17.1%)
gold ∈ {`how_to_feature`,`service_outage`}.  
**Examples:** `gold_45990`, `gold_1138538`.  
**Cause:** low support + lexical bleed into billing.  
**Mitigation:** more rare-class labels.

## 4. Ambiguous messages — 25 / 199 (12.6%)
gold = `other_ambiguous`.  
**Cause:** insufficient information.  
**Mitigation:** clarify questions (product).

## 5. Long messages — 7 / 199 (3.5%)
customer_message length > 220.  
**Cause:** multi-issue tweets.  
**Mitigation:** multi-intent detection.

Do **not** label mode #1 as “Over-escalation 91” — that number is total disagreement (85+6).
