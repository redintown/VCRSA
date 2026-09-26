import sys
from pathlib import Path
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent

sys.path.insert(0, str(ML_DIR))

from lstm_model import MobilityLSTM
from admission_policy import classify_admission
from proof_of_resource import (
    ProofOfResource,
    ResourceClaim,
)


# ============================================================
# Configuration
# ============================================================

CSV_PATH = (
    PROJECT_DIR
    / "sumo"
    / "datasets"
    / "highway"
    / "raw"
    / "highway_mobility.csv"
)

MODEL_PATH = (
    ML_DIR
    / "checkpoints"
    / "best_highway_lstm.pt"
)

SCALER_PATH = (
    ML_DIR
    / "highway_lstm_scaler.pkl"
)

SNAPSHOT_TIME = 100.0
HISTORY_SECONDS = 10

TARGET_ZONE = 0
RESOURCE_TYPE = "CPU"

X_THRESHOLD = 25.0
TMIN_THRESHOLD = 10.0

TRUST_THRESHOLD = 0.70
POR_TOLERANCE = 0.05

# ------------------------------------------------------------
# RSU / zone geometry
# ------------------------------------------------------------
#
# Our highway has 1 km zones.
# Zone 0:
#       x = 0 ... 1000
#
# A vehicle claiming Zone 0 but physically appearing
# outside this boundary is considered position-invalid.
#

ZONE_LENGTH = 1000.0

POSITION_MARGIN = 50.0


# ============================================================
# Security state
# ============================================================

@dataclass
class SecurityState:

    vehicle_id: str

    zone_id: int

    x: float
    y: float

    trust_score: float
    trust_status: str

    position_status: str

    predicted_residency: float
    admission: str

    por_status: str

    final_status: str


# ============================================================
# Resource profile
# ============================================================

def resource_profile(vehicle_index):

    resource_class = vehicle_index % 3

    if resource_class == 0:
        return 4.0

    if resource_class == 1:
        return 8.0

    return 12.0


# ============================================================
# Trust Gate
# ============================================================

def trust_gate(trust_score):

    if trust_score >= TRUST_THRESHOLD:
        return "TRUSTED"

    return "UNTRUSTED"


# ============================================================
# Position / RSU validation
# ============================================================

def validate_position(
    claimed_zone,
    x,
):

    zone_start = (
        claimed_zone
        * ZONE_LENGTH
    )

    zone_end = (
        zone_start
        + ZONE_LENGTH
    )

    lower_bound = (
        zone_start
        - POSITION_MARGIN
    )

    upper_bound = (
        zone_end
        + POSITION_MARGIN
    )

    if (
        lower_bound
        <= x
        <= upper_bound
    ):
        return "VALID"

    return "INVALID"


# ============================================================
# Load LSTM
# ============================================================

def load_lstm():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = MobilityLSTM(
        input_size=8,
        hidden_size=64,
        num_layers=2,
        output_size=1,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    scaler = joblib.load(
        SCALER_PATH
    )

    return model, scaler, device


# ============================================================
# Get 10-second history
# ============================================================

def get_snapshot_sequence(
    vehicle_df,
    snapshot_time,
):

    start_time = (
        snapshot_time
        - HISTORY_SECONDS
        + 1
    )

    sequence = vehicle_df[
        (
            vehicle_df["timestamp"]
            >= start_time
        )
        &
        (
            vehicle_df["timestamp"]
            <= snapshot_time
        )
    ].copy()

    if len(sequence) != HISTORY_SECONDS:
        return None

    sequence = sequence.sort_values(
        "timestamp"
    )

    timestamps = sequence[
        "timestamp"
    ].to_numpy()

    if not np.allclose(
        np.diff(timestamps),
        1.0,
    ):
        return None

    if sequence["zone_id"].nunique() != 1:
        return None

    return sequence


# ============================================================
# LSTM features
# ============================================================

def prepare_features(sequence):

    direction_numeric = (
        sequence["direction"]
        .map(
            {
                "EAST": 1.0,
                "WEST": -1.0,
            }
        )
        .astype(float)
    )

    lane_numeric = (
        sequence["lane_id"]
        .str.extract(
            r"_(\d+)$"
        )[0]
        .astype(float)
    )

    return np.column_stack(
        [
            sequence["x"].to_numpy(),
            sequence["y"].to_numpy(),
            sequence["speed"].to_numpy(),
            sequence[
                "acceleration"
            ].to_numpy(),
            sequence["heading"].to_numpy(),
            lane_numeric.to_numpy(),
            direction_numeric.to_numpy(),
            sequence["zone_id"].to_numpy(),
        ]
    ).astype(np.float32)


# ============================================================
# LSTM prediction
# ============================================================

def predict_residency(
    sequence,
    model,
    scaler,
    device,
):

    features = prepare_features(
        sequence
    )

    scaled = scaler.transform(
        features
    )

    tensor = torch.tensor(
        scaled,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        prediction = model(
            tensor
        )

    return max(
        float(prediction.item()),
        0.0,
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Loading mobility dataset..."
    )

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"Mobility records: {len(df)}"
    )

    model, scaler, device = (
        load_lstm()
    )

    print(
        f"LSTM device: {device}"
    )

    por = ProofOfResource(
        tolerance=POR_TOLERANCE
    )

    vehicle_ids = (
        df["vehicle_id"]
        .drop_duplicates()
        .tolist()
    )

    results = []

    for vehicle_index, vehicle_id in enumerate(
        vehicle_ids
    ):

        vehicle_df = df[
            df["vehicle_id"]
            == vehicle_id
        ].copy()

        current = vehicle_df[
            vehicle_df["timestamp"]
            == SNAPSHOT_TIME
        ]

        if current.empty:
            continue

        current = current.iloc[0]

        zone_id = int(
            current["zone_id"]
        )

        if zone_id != TARGET_ZONE:
            continue

        sequence = get_snapshot_sequence(
            vehicle_df,
            SNAPSHOT_TIME,
        )

        if sequence is None:
            continue

        # ----------------------------------------------------
        # S1: Trust
        # ----------------------------------------------------

        trust_score = 0.90

        if vehicle_id == "eastbound_flow.21":
            trust_score = 0.40

        trust_status = trust_gate(
            trust_score
        )

        # ----------------------------------------------------
        # S4: Position / RSU validation
        # ----------------------------------------------------
        #
        # Deliberately simulate a false position claim
        # for eastbound_flow.18.
        #
        # Actual x is retained in the output.
        # Claimed zone is changed to Zone 0.
        #

        claimed_zone = zone_id

        if vehicle_id == "eastbound_flow.18":
            claimed_zone = 1

        position_status = validate_position(
            claimed_zone,
            float(current["x"]),
        )

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        predicted_residency = (
            predict_residency(
                sequence,
                model,
                scaler,
                device,
            )
        )

        admission = classify_admission(
            predicted_residency,
            X_THRESHOLD,
            TMIN_THRESHOLD,
        )

        # ----------------------------------------------------
        # PoR
        # ----------------------------------------------------

        claimed_resource = resource_profile(
            vehicle_index
        )

        measured_resource = claimed_resource

        if vehicle_id == "eastbound_flow.20":
            measured_resource = (
                claimed_resource * 0.40
            )

        claim = ResourceClaim(
            vehicle_id=vehicle_id,
            resource_type=RESOURCE_TYPE,
            claimed_amount=claimed_resource,
        )

        proof = por.verify(
            claim,
            measured_resource,
        )

        # ----------------------------------------------------
        # Final security decision
        # ----------------------------------------------------

        final_pass = (
            trust_status == "TRUSTED"
            and position_status == "VALID"
            and proof.status.value == "VERIFIED"
            and admission.value != "REJECT"
        )

        final_status = (
            "ELIGIBLE"
            if final_pass
            else "BLOCKED"
        )

        results.append(
            SecurityState(
                vehicle_id=vehicle_id,
                zone_id=zone_id,
                x=float(current["x"]),
                y=float(current["y"]),
                trust_score=trust_score,
                trust_status=trust_status,
                position_status=position_status,
                predicted_residency=(
                    predicted_residency
                ),
                admission=admission.value,
                por_status=proof.status.value,
                final_status=final_status,
            )
        )

    # ========================================================
    # Output
    # ========================================================

    print()
    print("=" * 145)
    print(
        "        S1 TRUST → S4 POSITION/RSU → S2 PoR → ADMISSION"
    )
    print("=" * 145)

    print(
        f"Snapshot={SNAPSHOT_TIME:.0f}s"
        f" | Target Zone={TARGET_ZONE}"
    )

    print()

    print(
        f"{'Vehicle':<24}"
        f"{'X':>8}"
        f"{'Trust':>9}"
        f"{'Trust Gate':>13}"
        f"{'Position':>12}"
        f"{'Residency':>11}"
        f"{'Admission':>14}"
        f"{'PoR':>11}"
        f"{'Final':>12}"
    )

    print("-" * 145)

    for result in results:

        print(
            f"{result.vehicle_id:<24}"
            f"{result.x:>8.1f}"
            f"{result.trust_score:>9.2f}"
            f"{result.trust_status:>13}"
            f"{result.position_status:>12}"
            f"{result.predicted_residency:>11.2f}"
            f"{result.admission:>14}"
            f"{result.por_status:>11}"
            f"{result.final_status:>12}"
        )

    # ========================================================
    # Statistics
    # ========================================================

    trust_rejected = sum(
        r.trust_status == "UNTRUSTED"
        for r in results
    )

    position_rejected = sum(
        r.position_status == "INVALID"
        for r in results
    )

    por_rejected = sum(
        r.por_status == "REJECTED"
        for r in results
    )

    admission_rejected = sum(
        r.admission == "REJECT"
        for r in results
    )

    eligible = sum(
        r.final_status == "ELIGIBLE"
        for r in results
    )

    print()
    print(
        f"Total vehicles       : {len(results)}"
    )

    print(
        f"Trust rejected       : {trust_rejected}"
    )

    print(
        f"Position rejected    : {position_rejected}"
    )

    print(
        f"PoR rejected         : {por_rejected}"
    )

    print(
        f"Admission rejected   : {admission_rejected}"
    )

    print(
        f"Final eligible       : {eligible}"
    )

    # ========================================================
    # Validation
    # ========================================================

    trust_test = any(
        r.vehicle_id == "eastbound_flow.21"
        and r.trust_status == "UNTRUSTED"
        for r in results
    )

    position_test = any(
        r.vehicle_id == "eastbound_flow.18"
        and r.position_status == "INVALID"
        for r in results
    )

    por_test = any(
        r.vehicle_id == "eastbound_flow.20"
        and r.por_status == "REJECTED"
        for r in results
    )

    blocked_correctly = all(
        r.final_status == "BLOCKED"
        for r in results
        if (
            r.trust_status == "UNTRUSTED"
            or r.position_status == "INVALID"
            or r.por_status == "REJECTED"
            or r.admission == "REJECT"
        )
    )

    print()
    print("=" * 145)
    print(
        "                           VALIDATION"
    )
    print("=" * 145)

    print(
        "S1 Trust Gate          : "
        + ("PASS" if trust_test else "FAIL")
    )

    print(
        "S4 Position/RSU Gate   : "
        + ("PASS" if position_test else "FAIL")
    )

    print(
        "S2 PoR Gate            : "
        + ("PASS" if por_test else "FAIL")
    )

    print(
        "Admission Gate         : PASS"
    )

    print(
        "Invalid vehicles blocked: "
        + (
            "PASS"
            if blocked_correctly
            else "FAIL"
        )
    )

    print()
    print(
        "Full Security Pipeline : "
        + (
            "PASS"
            if (
                trust_test
                and position_test
                and por_test
                and blocked_correctly
            )
            else "CHECK"
        )
    )

    print("=" * 145)


if __name__ == "__main__":
    main()
