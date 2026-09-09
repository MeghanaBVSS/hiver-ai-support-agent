# Case studies (10)

Generated from frozen deterministic agent.

## normal_successful_auto_or_copy (`gold_2544720`)
- **input:** @115940 when will the new episode of @1760  and the @167381 be on hulu
- **intent:** pred=content_availability (conf=0.780) gold=content_availability
- **evidence_strength:** 0.599; top_sim=0.781
- **reply:** @221742 Sorry for the delay! The new Empire is now up on all supported devices: https://t.co/3hYIXFIT24. Happy streaming!
- **escalation:** False reason=auto_handle_strong_evidence
- **evaluation:** gold_esc=False; pred_esc=False; intent_match=True

## escalation_case (`gold_1224206`)
- **input:** Why do I have to pay for Hulu and have to watch commercials? 🤔😭 @115940
- **intent:** pred=billing_subscription (conf=0.689) gold=billing_subscription
- **evidence_strength:** 0.628; top_sim=0.766
- **reply:** I'm escalating this to a Hulu specialist because: billing_refund. A human agent can take a closer look with your account context.
- **escalation:** True reason=billing_refund
- **evaluation:** gold_esc=True; pred_esc=True; intent_match=True

## weak_evidence_or_over_escalation (`gold_2911030`)
- **input:** @hulu_support why is the app constantly pausing and buffering lately?  It's not connection because it is happening at several different locations and devices for me.  From ps4 to Android phone.
- **intent:** pred=playback_error (conf=0.415) gold=playback_error
- **evidence_strength:** 0.481; top_sim=0.480
- **reply:** I'm escalating this to a Hulu specialist because: low_intent_confidence. A human agent can take a closer look with your account context.
- **escalation:** True reason=low_intent_confidence
- **evaluation:** gold_esc=False; pred_esc=True; intent_match=True

## ambiguous_case (`gold_199210`)
- **input:** @hulu_support The Purge: Election Year is showing up on my Hulu app but not my tv. It doesn’t say it’s part of the HBO preview weekend. Help
- **intent:** pred=app_device_issue (conf=0.265) gold=other_ambiguous
- **evidence_strength:** 0.465; top_sim=0.543
- **reply:** I'm escalating this to a Hulu specialist because: low_intent_confidence. A human agent can take a closer look with your account context.
- **escalation:** True reason=low_intent_confidence
- **evaluation:** gold_esc=True; pred_esc=True; intent_match=False

## billing (`gold_310432`)
- **input:** @hulu_support I signed up for a free trial at the $5.99 rate and switched to $11.99 an hour later and now I'm getting charged $4. Why?
- **intent:** pred=billing_subscription (conf=0.816) gold=billing_subscription
- **evidence_strength:** 0.606; top_sim=0.741
- **reply:** I'm escalating this to a Hulu specialist because: billing_refund. A human agent can take a closer look with your account context.
- **escalation:** True reason=billing_refund
- **evaluation:** gold_esc=True; pred_esc=True; intent_match=True

## login_account (`gold_954498`)
- **input:** @hulu_support wait time today to try and figure out why I haven’t been able to log in for five days https://t.co/lgt47x96X7
- **intent:** pred=login_account (conf=0.645) gold=login_account
- **evidence_strength:** 0.448; top_sim=0.588
- **reply:** I'm escalating this to a Hulu specialist because: account_specific. A human agent can take a closer look with your account context.
- **escalation:** True reason=account_specific
- **evaluation:** gold_esc=True; pred_esc=True; intent_match=True

## playback_live_boundary (`gold_1372321`)
- **input:** @hulu_support ok...I give up. Hulu TV "live" is constantly buffering and is now a solid half inning behind the WS. What do I do?
- **intent:** pred=live_tv_issues (conf=0.342) gold=playback_error
- **evidence_strength:** 0.476; top_sim=0.562
- **reply:** I'm escalating this to a Hulu specialist because: conflicting_evidence. A human agent can take a closer look with your account context.
- **escalation:** True reason=conflicting_evidence
- **evaluation:** gold_esc=False; pred_esc=True; intent_match=False

## successful_grounded_reply (`gold_1587983`)
- **input:** @488406 @hulu_support Agreed. I hate the new setup. I hardly ever use it anymore. Almost no reason for me to have it
- **intent:** pred=feedback_complaint (conf=0.787) gold=feedback_complaint
- **evidence_strength:** 0.516; top_sim=0.552
- **reply:** @143922 Sorry for the frustration! On most devices, you can press the back button to get rid of the overlay. We'll share your feedback.
- **escalation:** False reason=auto_handle_strong_evidence
- **evaluation:** gold_esc=True; pred_esc=False; intent_match=True

## content_availability (`gold_815865`)
- **input:** @hulu_support Hi! I'm not seeing episode 5, season 6 of @19606 yet. Shouldn't it be available already? Past midnight on a Tuesday.
- **intent:** pred=content_availability (conf=0.833) gold=content_availability
- **evidence_strength:** 0.465; top_sim=0.663
- **reply:** I'm escalating this to a Hulu specialist because: conflicting_evidence. A human agent can take a closer look with your account context.
- **escalation:** True reason=conflicting_evidence
- **evaluation:** gold_esc=False; pred_esc=True; intent_match=True

## worst_observed_policy_mismatch (`gold_45990`)
- **input:** @hulu_support I just wanted to know, with the basic package (limited commercials), how many screens can you watch at the same time?
- **intent:** pred=billing_subscription (conf=0.470) gold=how_to_feature
- **evidence_strength:** 0.466; top_sim=0.557
- **reply:** I'm escalating this to a Hulu specialist because: billing_refund. A human agent can take a closer look with your account context.
- **escalation:** True reason=billing_refund
- **evaluation:** gold_esc=False; pred_esc=True; intent_match=False
