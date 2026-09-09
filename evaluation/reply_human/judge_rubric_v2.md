# Judge rubric v2 (calibrated)

## What changed vs v1

1. **Explicit escalation grounding rule:** A response must **not** be penalized on groundedness for omitting unsupported details when the correct behavior is escalation.
2. Clarified groundedness for: direct answers, evidence-backed answers, safe escalation, insufficient-evidence responses, account-specific issues, unsupported claims.
3. Separated “ungrounded invention” (bad) from “abstention / escalate” (can be highly grounded).

## Why

Phase 3B data: groundedness within±1 = 0.12 overall, but **1.00 on non-escalated** vs **0.02 on escalated**.  
Hypothesis “judge treats safe escalation as ungrounded” covers **~70%** of |diff|>1 disagreements — supported by evidence, not assumed a priori.

## Rubric text

```
Fixed rubric (1–5 ordinal scales):
1 = clearly wrong / invents unsupported facts / unsafe auto-handle
2 = major issues
3 = mixed / partial
4 = mostly good with minor issues
5 = excellent for the dimension

correctness: factual alignment with the customer issue; no invented policy/pricing/account state
groundedness:
  - Direct / evidence-backed answers: claims must be supported by retrieved evidence or customer text
  - Safe escalation / insufficient-evidence responses: score HIGH if the reply abstains from unsupported claims and states escalation/uncertainty clearly
  - CRITICAL RULE: Do NOT penalize groundedness for failing to provide unsupported information when the correct behavior is escalation
  - Account-specific / billing / security: escalating without fabricating account actions is grounded
  - Unsupported claims (prices, refunds, timelines, password resets not in evidence): score LOW
helpfulness: useful next step or clear explanation (escalation can be helpful if it is the right next step)
actionability: customer knows what to do next OR why a specialist will take over
brand_style: polite, concise, brand-appropriate support tone
escalation_appropriateness: escalate when evidence/risk weak; avoid unnecessary escalate when strong evidence supports a safe reply
```
