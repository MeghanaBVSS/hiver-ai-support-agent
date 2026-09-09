# Escalation annotation rubric

Status: **IMPLEMENTED** policy rubric.

Escalation labels are **policy judgments** for whether an AI agent
should auto-handle. They are **not** historical Hulu escalation labels.

| reason | escalate? | description |
|---|---|---|
| `generic_informational` | False | Clear FAQ/how-to with no account mutation required. |
| `troubleshooting_strong_evidence` | False | Routine playback/Live TV/device issue with a safe generic fix path. |
| `account_specific_action` | True | Login/password/profile/account access or account-bound actions. |
| `billing_refund_dispute` | True | Charges, refunds, cancellations, payment disputes. |
| `security_privacy` | True | Hacked/fraud/unauthorized access/privacy concerns. |
| `ambiguous_request` | True | Unclear ask; cannot safely choose an action. |
| `insufficient_evidence` | True | Missing details needed for a safe answer. |
| `potentially_harmful_automation` | True | Auto-reply could cause harm, false promises, or unsafe instructions. |
| `unsupported_operational_action` | True | Requires ops capability the agent cannot perform. |

## Distinctions

- **generic informational support** → usually auto-handle
- **troubleshooting with strong evidence** → usually auto-handle
- **account-specific actions** → escalate
- **billing/refund disputes** → escalate
- **security/privacy** → escalate
- **ambiguous requests** → escalate
- **insufficient evidence** → escalate
- **potentially harmful/unsafe automation** → escalate