"""
decision_engine.py
Rule-based clean/no-action/insufficient-evidence decision layer.

Deliberately NOT a calibrated optimizer — see project brief Section 1.
This module combines:
  - classifier output (label + softmax confidence)
  - a confidence threshold for "insufficient evidence"
  - a soft rain signal (NOT treated as automatic soiling removal)
  - a cited, region-approximate daily soiling loss rate assumption

...to produce a recommendation plus a *range* (not a point estimate) of
estimated energy lost per day by deferring cleaning.

TWO ENTRY POINTS:
  - decide(): clean_now / no_action / insufficient_evidence. No cost
    awareness — dust detected means clean now, full stop.
  - decide_with_cost_sensitivity(): adds a genuine "defer" action, gated
    on comparing accumulated estimated loss (PKR) against a cleaning-cost
    assumption. Validated via simulation in scenario_cost_analysis.ipynb
    (Week 3) across two real net-metered systems (5.2kW, 8kW) and a
    soiling-rate x cleaning-cost sensitivity grid. Requires the caller to
    track accumulated loss since the last cleaning — this module is
    stateless and does not persist that itself.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


# --- Regional soiling-rate assumptions (cited, not invented) ---
# Ghazi et al., Lahore, Pakistan (30° tilt): ~0.8%/day — primary assumption,
# used as a regional placeholder, NOT a Karachi-specific measurement.
# Sensitivity range below brackets NREL's low end and the Cyprus arid
# test-bench's high end around that point estimate, so the output is
# presented as a range rather than false precision.
SOILING_RATE_LOW = 0.003    # 0.3%/day  (NREL typical low end)
SOILING_RATE_MID = 0.008    # 0.8%/day  (Ghazi et al., Lahore — primary)
SOILING_RATE_HIGH = 0.019   # 1.9%/day  (Cyprus arid seasonal average)

# Confidence threshold for "insufficient evidence" — empirically calibrated
# against the real validation set (n=511) for the canonical best.pt
# checkpoint: 0.70 -> 96.3% coverage, 98.4% accuracy on the covered subset.
# See weather_decision_layer.ipynb, Section 3, for the calibration run.
DEFAULT_CONFIDENCE_THRESHOLD = 0.70


@dataclass
class Recommendation:
    action: str                                    # "clean_now" | "no_action" | "insufficient_evidence"
    reason: str
    est_daily_loss_pct_range: Optional[Tuple[float, float, float]]  # (low, mid, high) or None
    inputs_used: dict


def decide(
    classifier_label: str,
    classifier_confidence: float,
    irradiance_kwh_m2: float,
    rain_expected_soon: bool,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Recommendation:
    """
    Args:
        classifier_label: "clean" or "dusty" (raw top-1 prediction)
        classifier_confidence: softmax value for the predicted class, 0-1
        irradiance_kwh_m2: from weather_client.get_irradiance_climatology()
        rain_expected_soon: from weather_client.get_rain_forecast()
        confidence_threshold: override the module default once calibrated
    """

    inputs_used = {
        "classifier_label": classifier_label,
        "classifier_confidence": classifier_confidence,
        "irradiance_kwh_m2": irradiance_kwh_m2,
        "rain_expected_soon": rain_expected_soon,
        "confidence_threshold": confidence_threshold,
    }

    # 1. Confidence gate — "insufficient evidence" is a threshold on the
    #    existing binary softmax output, not a trained third class.
    if classifier_confidence < confidence_threshold:
        return Recommendation(
            action="insufficient_evidence",
            reason=(
                f"Classifier confidence ({classifier_confidence:.2f}) is below "
                f"the {confidence_threshold:.2f} threshold. Retake the photo "
                f"with better lighting or a closer, more direct angle on the "
                f"panel before deciding."
            ),
            est_daily_loss_pct_range=None,
            inputs_used=inputs_used,
        )

    # 2. Clean panel — nothing to decide
    if classifier_label == "clean":
        return Recommendation(
            action="no_action",
            reason="Classifier detected a clean panel with sufficient confidence.",
            est_daily_loss_pct_range=None,
            inputs_used=inputs_used,
        )

    # 3. Dusty panel, confident — estimate what leaving it costs, as a
    #    range rather than a single invented number.
    loss_range_pct = (
        SOILING_RATE_LOW * 100,
        SOILING_RATE_MID * 100,
        SOILING_RATE_HIGH * 100,
    )

    # Rain is a SOFT signal only — cited evidence shows adherent dust often
    # survives rainfall, so this nudges the explanation text, not the action.
    rain_note = (
        " Rain is forecast soon, but adherent dust often survives rainfall "
        "per the cited literature — this is not treated as a reason to wait."
        if rain_expected_soon
        else ""
    )

    reason = (
        f"Classifier detected dust with {classifier_confidence:.2f} confidence. "
        f"Estimated output loss ranges {loss_range_pct[0]:.2f}-{loss_range_pct[2]:.2f}%/day "
        f"(region-approximate estimate from Lahore, PK data — not Karachi-specific)."
        + rain_note
    )

    return Recommendation(
        action="clean_now",
        reason=reason,
        est_daily_loss_pct_range=loss_range_pct,
        inputs_used=inputs_used,
    )


def decide_with_cost_sensitivity(
    classifier_label: str,
    classifier_confidence: float,
    irradiance_kwh_m2: float,
    rain_expected_soon: bool,
    accumulated_loss_pkr_since_last_clean: float,
    cleaning_cost_pkr: float,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Recommendation:
    """
    Cost-aware extension of decide() — adds a genuine "defer" action.

    Validated in scenario_cost_analysis.ipynb: across a grid of 3 soiling
    rates x 4 cleaning-cost brackets (24 scenarios), on two real systems
    (5.2kW and 8kW, both net-metered, Karachi), this strategy had the
    lowest total cost in 15/24 scenarios (62.5%) under import-rate economic
    valuation and 20/24 scenarios (83.3%) under export-rate valuation. The
    one consistent counter-case: high soiling rate + high cleaning cost,
    where a rigid fixed-interval schedule can outperform by capping
    worst-case accumulated loss. Report that exception, don't hide it.

    IMPORTANT — state tracking: this function is stateless by design. The
    caller is responsible for tracking `accumulated_loss_pkr_since_last_clean`
    across repeated checks (e.g. a simple running log keyed by
    installation/panel ID, reset to 0 whenever a cleaning actually happens).
    This module does not persist that state itself.

    Args:
        classifier_label, classifier_confidence, irradiance_kwh_m2,
            rain_expected_soon: same as decide()
        accumulated_loss_pkr_since_last_clean: running total of estimated
            PKR lost to soiling since the panel was last cleaned (caller-tracked)
        cleaning_cost_pkr: the cost of a single cleaning event. For the
            validated DIY pressure-washer case (<15 min), illustrative
            brackets were Rs 50 (low) / 150 (mid) / 300 (high) — see
            scenario_cost_analysis.ipynb. Use a real quote if/when available.
        confidence_threshold: override the module default once calibrated
    """

    inputs_used = {
        "classifier_label": classifier_label,
        "classifier_confidence": classifier_confidence,
        "irradiance_kwh_m2": irradiance_kwh_m2,
        "rain_expected_soon": rain_expected_soon,
        "accumulated_loss_pkr_since_last_clean": accumulated_loss_pkr_since_last_clean,
        "cleaning_cost_pkr": cleaning_cost_pkr,
        "confidence_threshold": confidence_threshold,
    }

    # 1. Confidence gate — same as decide()
    if classifier_confidence < confidence_threshold:
        return Recommendation(
            action="insufficient_evidence",
            reason=(
                f"Classifier confidence ({classifier_confidence:.2f}) is below "
                f"the {confidence_threshold:.2f} threshold. Retake the photo "
                f"with better lighting or a closer, more direct angle."
            ),
            est_daily_loss_pct_range=None,
            inputs_used=inputs_used,
        )

    # 2. Clean panel — nothing to decide
    if classifier_label == "clean":
        return Recommendation(
            action="no_action",
            reason="Classifier detected a clean panel with sufficient confidence.",
            est_daily_loss_pct_range=None,
            inputs_used=inputs_used,
        )

    # 3. Dusty panel, confident — now compare accumulated loss vs. cleaning cost
    loss_range_pct = (SOILING_RATE_LOW * 100, SOILING_RATE_MID * 100, SOILING_RATE_HIGH * 100)

    rain_note = (
        " Rain is forecast soon, but adherent dust often survives rainfall "
        "per the cited literature — this is not treated as a reason to wait."
        if rain_expected_soon
        else ""
    )

    if accumulated_loss_pkr_since_last_clean >= cleaning_cost_pkr:
        reason = (
            f"Classifier detected dust with {classifier_confidence:.2f} confidence. "
            f"Estimated accumulated loss since last cleaning (Rs {accumulated_loss_pkr_since_last_clean:.0f}) "
            f"has met or exceeded the cleaning cost (Rs {cleaning_cost_pkr:.0f})."
            + rain_note
        )
        return Recommendation(
            action="clean_now",
            reason=reason,
            est_daily_loss_pct_range=loss_range_pct,
            inputs_used=inputs_used,
        )
    else:
        shortfall = cleaning_cost_pkr - accumulated_loss_pkr_since_last_clean
        reason = (
            f"Classifier detected dust with {classifier_confidence:.2f} confidence, "
            f"but accumulated loss since last cleaning (Rs {accumulated_loss_pkr_since_last_clean:.0f}) "
            f"has not yet met the cleaning cost (Rs {cleaning_cost_pkr:.0f}); "
            f"roughly Rs {shortfall:.0f} more in estimated loss before cleaning is worthwhile."
            + rain_note
        )
        return Recommendation(
            action="defer",
            reason=reason,
            est_daily_loss_pct_range=loss_range_pct,
            inputs_used=inputs_used,
        )


if __name__ == "__main__":
    # Manual smoke test with made-up inputs (no network calls needed)
    examples = [
        dict(classifier_label="dusty", classifier_confidence=0.91,
             irradiance_kwh_m2=5.8, rain_expected_soon=False),
        dict(classifier_label="dusty", classifier_confidence=0.91,
             irradiance_kwh_m2=5.8, rain_expected_soon=True),
        dict(classifier_label="clean", classifier_confidence=0.95,
             irradiance_kwh_m2=5.8, rain_expected_soon=False),
        dict(classifier_label="dusty", classifier_confidence=0.52,
             irradiance_kwh_m2=5.8, rain_expected_soon=False),
    ]

    for ex in examples:
        rec = decide(**ex)
        print(f"\nInput: {ex}")
        print(f"  -> action: {rec.action}")
        print(f"  -> reason: {rec.reason}")

    print("\n" + "=" * 60)
    print("decide_with_cost_sensitivity() smoke test")
    print("=" * 60)

    cost_examples = [
        # dusty, confident, plenty of accumulated loss -> should clean_now
        dict(classifier_label="dusty", classifier_confidence=0.91, irradiance_kwh_m2=5.8,
             rain_expected_soon=False, accumulated_loss_pkr_since_last_clean=400, cleaning_cost_pkr=150),
        # dusty, confident, loss hasn't caught up to cost yet -> should defer
        dict(classifier_label="dusty", classifier_confidence=0.91, irradiance_kwh_m2=5.8,
             rain_expected_soon=False, accumulated_loss_pkr_since_last_clean=40, cleaning_cost_pkr=150),
        # clean panel -> no_action regardless of cost inputs
        dict(classifier_label="clean", classifier_confidence=0.95, irradiance_kwh_m2=5.8,
             rain_expected_soon=False, accumulated_loss_pkr_since_last_clean=0, cleaning_cost_pkr=150),
    ]

    for ex in cost_examples:
        rec = decide_with_cost_sensitivity(**ex)
        print(f"\nInput: {ex}")
        print(f"  -> action: {rec.action}")
        print(f"  -> reason: {rec.reason}")
