"""Existing measurement validity checks; no changes to the GSD estimate."""
import math


REQUIRED_XMP = ("GpsLatitude", "GpsLongitude", "AbsoluteAltitude", "GimbalYawDegree",
                "GimbalPitchDegree", "GimbalRollDegree", "CalibratedFocalLength",
                "CalibratedOpticalCenterX", "CalibratedOpticalCenterY")
GSD_REVIEW_RATIO = 10.0


def review_measurement(spatial: dict, focal_px: float) -> dict:
    """Conservative display gate, NOT a new GSD estimate or accuracy claim.

    For the user's roughly frontal imagery assumption, a local scale over 10x
    camera-to-hit distance / focal length needs manual review. Raw legacy
    measurements are preserved even when omitted from the public CSV.
    """
    reasons = []
    if not spatial.get("xyz_valid"):
        reasons.append(spatial.get("xyz_miss_reason") or "representative_ray_no_hit")
    if not spatial.get("measurement_valid"):
        reasons.append(spatial.get("measurement_miss_reason") or "local_gsd_unavailable")
    distance = spatial.get("mesh_ray_t_m")
    reference = float(distance) / float(focal_px) if distance is not None and distance > 0 else None
    scales = [spatial.get("gsd_length_m_per_px"), spatial.get("gsd_width_m_per_px")]
    scale = max((float(value) for value in scales if value is not None), default=None)
    ratio = scale / reference if reference and scale is not None else None
    if ratio is not None and (not math.isfinite(ratio) or ratio > GSD_REVIEW_RATIO):
        reasons.append("local_gsd_exceeds_10x_frontal_reference_manual_review")
    return {"physical_values_exported": not reasons,
            "reasons": reasons, "frontal_reference_m_per_px": reference,
            "max_directional_gsd_to_frontal_ratio": ratio,
            "review_ratio_threshold": GSD_REVIEW_RATIO}
