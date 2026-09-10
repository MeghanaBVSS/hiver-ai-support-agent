# Validation enhancement report

Terminology: **human-annotated golden evaluation set** (taxonomy-guided). Independently validated on a subset using inter-annotator agreement.

## 1. Second-annotator inter-annotator agreement

- Subset N = **50**
- Intent exact agreement = **0.620**
- Intent Cohen's κ = **0.575**
- Escalation exact agreement = **0.720**
- Escalation Cohen's κ = **0.435**
- Intent disagreements = **19**
- Escalation disagreements = **14**

Top intents causing disagreement (annotator_1 label):

- `other_ambiguous`: 6
- `how_to_feature`: 3
- `billing_subscription`: 2
- `content_availability`: 2
- `feedback_complaint`: 2
- `playback_error`: 2
- `app_device_issue`: 1
- `service_outage`: 1

## 2. Golden-set leakage check

```
Golden-set leakage check
------------------------
Training overlap:       0
Retrieval overlap:      0
Prompt-example overlap: 0
Status: PASS
```


## 3. Rule-aid vs baseline stack

| System | Intent macro-F1 | Escalation F1 | Unsafe auto-handle |
|---|---:|---:|---:|
| rule_aid_only | 0.986 | 0.972 | 0.011 |
| tfidf_logreg_plus_simple_policy | 0.668 | 0.700 | 0.023 |
| logreg_plus_rule_aid | 0.981 | 0.966 | 0.023 |

> **Important:** High rule-aid ↔ golden agreement is partly **circular** (golden labeling used rule-aid). Use it to show rule consistency with the annotation process, not as proof that rules beat the learned model. The fair ML comparison is TF-IDF+LR / final semantic+policy vs the human-annotated evaluation set.


## 4. Final system (semantic retrieval + policy)

### Intent classification
- Accuracy: **0.693**
- Macro F1: **0.668**
- Weighted F1: **0.675**

### Escalation (separate decision)
- Accuracy: **0.543**
- Precision: **0.488**
- Recall: **0.931**
- F1: **0.640**
- Unsafe auto-handle (FN rate): **0.069**
- Confusion [[TN,FP],[FN,TP]]: `[[27, 85], [6, 81]]`

### By difficulty

| Difficulty | N | Intent macro-F1 | Escalation F1 |
|---|---:|---:|---:|
| easy | 94 | 0.662 | 0.519 |
| medium | 96 | 0.600 | 0.698 |
| hard | 9 | 0.214 | 1.000 |

### other_ambiguous
- Gold rate: **0.126**
- Pred rate: **0.261**
- False ambiguous: **34**

### Model confidence → intent accuracy

| Confidence | # examples | Intent accuracy |
|---|---:|---:|
| high | 35 | 0.943 |
| medium | 28 | 1.000 |
| low | 136 | 0.566 |

### Escalation false negatives (dangerous errors)
- N = **6**

- `other`: 1
- `unsafe_automation`: 5

### Error analysis (tagged mistakes)

1. **escalation policy ambiguity** — 59.9%
   - `gold_2911030`: gold=playback_error/False pred=playback_error/True
   - `gold_2721090`: gold=playback_error/False pred=playback_error/True
1. **ambiguous customer wording** — 27.0%
   - `gold_2623567`: gold=playback_error/False pred=other_ambiguous/True
   - `gold_2147146`: gold=playback_error/False pred=other_ambiguous/True
1. **intent boundary confusion** — 9.9%
   - `gold_1372321`: gold=playback_error/False pred=live_tv_issues/True
   - `gold_1197349`: gold=playback_error/False pred=app_device_issue/True
1. **multiple intents** — 2.0%
   - `gold_652356`: gold=login_account/True pred=app_device_issue/True
   - `gold_2363665`: gold=how_to_feature/True pred=billing_subscription/True
1. **device context confusion** — 1.3%
   - `gold_1355492`: gold=live_tv_issues/False pred=app_device_issue/False
   - `gold_1057041`: gold=content_availability/False pred=app_device_issue/True

## 5. Decision policy (explicit)

```
IF account-specific action     → ESCALATE
IF billing/refund dispute      → ESCALATE
IF security/privacy            → ESCALATE
IF ambiguous / weak evidence   → ESCALATE
IF unsafe / unsupported op     → ESCALATE
ELSE IF strong evidence gates  → auto-handle (copy historical reply)
ELSE                           → ESCALATE (conservative default)
```

Intent classification ≠ escalation decision. Policy is final authority.

## 6. Architecture

```
Customer Message
       │
       ▼
Preprocessing / normalize
       │
       ▼
Intent Classification (TF-IDF+LR)
       │
       ├──────────────────┐
       ▼                  ▼
Semantic Retrieval   Escalation Policy
       │                  │
       └────────┬─────────┘
                ▼
         Final Prediction
        ┌───────┴────────┐
        ▼                ▼
     Intent          Escalate?
        │
        ▼
    Confidence
```

## 7. Limitations

- Golden labels are taxonomy-guided human annotations; IAA measured on a 50-example subset.
- Rule-aid may introduce confirmation bias during initial labeling.
- Evaluation set is relatively small (N=199).
- Taxonomy boundaries contain subjective cases (playback vs live_tv).
- Performance may not generalize to unseen customer language or current Hulu policy.
- Historical Hulu tweet replies are evidence, not escalation ground truth.
- Semantic+LLM golden reply quality remains NOT MEASURED without API key.

