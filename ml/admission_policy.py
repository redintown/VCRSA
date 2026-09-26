from enum import Enum


class AdmissionTier(Enum):
    FULL = "FULL"
    PROVISIONAL = "PROVISIONAL"
    REJECT = "REJECT"


def classify_admission(
    predicted_residency_seconds: float,
    x_threshold: float,
    tmin_threshold: float,
) -> AdmissionTier:
    """
    Admission policy based on predicted remaining residency.

    FULL:
        predicted lifetime >= X

    PROVISIONAL:
        Tmin <= predicted lifetime < X

    REJECT:
        predicted lifetime < Tmin
    """

    if predicted_residency_seconds >= x_threshold:
        return AdmissionTier.FULL

    if predicted_residency_seconds >= tmin_threshold:
        return AdmissionTier.PROVISIONAL

    return AdmissionTier.REJECT


def explain_admission(
    predicted_residency_seconds: float,
    x_threshold: float,
    tmin_threshold: float,
) -> str:

    tier = classify_admission(
        predicted_residency_seconds,
        x_threshold,
        tmin_threshold,
    )

    if tier == AdmissionTier.FULL:
        return (
            "FULL: predicted residency is sufficient "
            "for normal admission."
        )

    if tier == AdmissionTier.PROVISIONAL:
        return (
            "PROVISIONAL: admission is allowed with "
            "shorter/limited tenancy."
        )

    return (
        "REJECT: predicted residency is below the "
        "minimum acceptable threshold."
    )


if __name__ == "__main__":

    # ------------------------------------------------------------
    # Experiment parameters
    #
    # These are configurable because the architecture specifies
    # X and Tmin symbolically rather than giving fixed values.
    # ------------------------------------------------------------

    X = 25.0
    TMIN = 10.0

    test_predictions = [
        35.0,
        28.007,
        20.0,
        12.0,
        7.0,
    ]

    print("=" * 60)
    print("             ADMISSION POLICY TEST")
    print("=" * 60)

    print(
        f"X threshold    : {X:.1f} seconds"
    )

    print(
        f"Tmin threshold : {TMIN:.1f} seconds"
    )

    print()

    for prediction in test_predictions:

        tier = classify_admission(
            prediction,
            X,
            TMIN,
        )

        print(
            f"Predicted residency: "
            f"{prediction:6.3f}s"
            f"  →  {tier.value}"
        )

    print("=" * 60)
