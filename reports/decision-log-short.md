# Decision log (final) — 12 decisions

1. **Decision:** Explicit TWCS acquisition (no silent full download).  
   **Why:** Auditability / offline safety.  
   **Alternative:** Auto-download on import.  
   **Trade-off:** Extra setup step.  
   **Consequence:** Reproducible installs without surprise 500MB pulls.

2. **Decision:** Brand = `hulu_support` by suitability, not max volume.  
   **Why:** Cleaner 6–12 intent taxonomy.  
   **Alternative:** AmazonHelp.  
   **Trade-off:** Less data than mega-brands.  
   **Consequence:** Focused evaluation narrative.

3. **Decision:** Conservative conversation IDs for huge multi-author components.  
   **Why:** Undirected graphs merged unrelated customers.  
   **Alternative:** Always trust components.  
   **Trade-off:** Possible over-fragmentation.  
   **Consequence:** Safer splits.

4. **Decision:** Conversation-level split + exact duplicate-text purge.  
   **Why:** Prevent thread/text leakage.  
   **Alternative:** Tweet-level random split.  
   **Trade-off:** Slightly smaller eval pools.  
   **Consequence:** Honest retrieval/intent metrics.

5. **Decision:** 10 human-reviewed intents from train-only discovery.  
   **Why:** Fits Hulu themes without Banking77 sprawl.  
   **Alternative:** Raw cluster IDs or 77 banking intents.  
   **Trade-off:** Boundary fuzz (playback vs live).  
   **Consequence:** Interpretable taxonomy.

6. **Decision:** Keep TF-IDF+LR as primary intent classifier.  
   **Why:** Measured baseline; embeddings ≠ automatic upgrade.  
   **Alternative:** Embed+linear or LLM classify.  
   **Trade-off:** Lexical limits.  
   **Consequence:** Semantic agent intent = 0.693/0.668 matches LR.

7. **Decision:** Default embeddings = `tfidf_svd`.  
   **Why:** CI/demo without model downloads.  
   **Alternative:** Force sentence-transformers.  
   **Trade-off:** Weaker paraphrase retrieval.  
   **Consequence:** <15 min headline path.

8. **Decision:** Deterministic escalation policy is final authority.  
   **Why:** LLM must not override safety.  
   **Alternative:** Model-decided escalate.  
   **Trade-off:** Over-escalation.  
   **Consequence:** Unsafe auto-handle 0.069 at coverage 0.166.

9. **Decision:** Tune thresholds on validation only.  
   **Why:** Golden integrity.  
   **Alternative:** Tune on golden.  
   **Trade-off:** Noisy validation labels.  
   **Consequence:** Honest held-out safety metric.

10. **Decision:** Prefer unsafe auto-handle minimization over coverage.  
    **Why:** Trust > automation rate.  
    **Alternative:** Maximize auto-handle.  
    **Trade-off:** Helpfulness lower on human pack.  
    **Consequence:** Headline = unsafe 0.069.

11. **Decision:** LLM judge = evaluator only; not GT.  
    **Why:** Groundedness within±1 = 0.12 under v1; esc bias.  
    **Alternative:** Use judge scores as labels.  
    **Trade-off:** Need human pack for reply quality.  
    **Consequence:** Reply headlines from annotator_1.

12. **Decision:** Freeze brand/taxonomy/splits/policy for submission.  
    **Why:** Stop metric chasing.  
    **Alternative:** Keep iterating thresholds on golden.  
    **Trade-off:** Known over-escalation remains.  
    **Consequence:** Stable submission package.
