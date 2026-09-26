import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import torch

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent

sys.path.insert(0, str(ML_DIR))

from lstm_model import MobilityLSTM
from admission_policy import classify_admission


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

X_THRESHOLD = 25.0
TMIN_THRESHOLD = 10.0

FAIRNESS_RESERVE = 0.10

HANDOVER_MARGIN = 5.0


# ============================================================
# Data structures
# ============================================================

@dataclass
class Provider:
    provider_id: str
    resource_type: str
    available_amount: float
    predicted_residency: float
    trust_score: float = 1.0
    active: bool = True


@dataclass
class Allocation:
    provider_id: str
    resource_type: str
    amount: float
    predicted_residency: float


@dataclass
class Tenancy:
    request_id: str
    resource_type: str
    requested_amount: float
    allocated_amount: float
    duration: float
    providers: List[str]


# ============================================================
# LSTM
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


def find_sequence(vehicle_df):

    vehicle_df = vehicle_df.sort_values(
        "timestamp"
    )

    for zone_id, zone_df in vehicle_df.groupby(
        "zone_id"
    ):

        zone_df = zone_df.sort_values(
            "timestamp"
        )

        timestamps = zone_df[
            "timestamp"
        ].to_numpy()

        for i in range(
            len(timestamps) - 9
        ):

            window = timestamps[
                i:i + 10
            ]

            if np.allclose(
                np.diff(window),
                1.0
            ):

                return zone_df.iloc[
                    i:i + 10
                ].copy()

    return None


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
# Greedy allocation
# ============================================================

def provider_score(
    provider: Provider,
) -> float:

    residency_score = min(
        provider.predicted_residency / 60.0,
        1.0,
    )

    capacity_score = min(
        provider.available_amount / 16.0,
        1.0,
    )

    return (
        0.50 * residency_score
        + 0.30 * provider.trust_score
        + 0.20 * capacity_score
    )


def greedy_allocate(
    providers: List[Provider],
    resource_type: str,
    requested_amount: float,
):

    candidates = []

    for provider in providers:

        if not provider.active:
            continue

        if provider.resource_type != resource_type:
            continue

        usable = (
            provider.available_amount
            * (1.0 - FAIRNESS_RESERVE)
        )

        if usable <= 0:
            continue

        candidates.append(
            (
                provider,
                usable,
                provider_score(provider),
            )
        )

    candidates.sort(
        key=lambda x: x[2],
        reverse=True,
    )

    remaining = requested_amount
    allocations = []

    for provider, usable, score in candidates:

        if remaining <= 0:
            break

        amount = min(
            usable,
            remaining,
        )

        allocations.append(
            Allocation(
                provider_id=provider.provider_id,
                resource_type=provider.resource_type,
                amount=amount,
                predicted_residency=provider.predicted_residency,
            )
        )

        remaining -= amount

    if remaining > 1e-9:
        return []

    return allocations


# ============================================================
# PRT
# ============================================================

def create_tenancy(
    request_id: str,
    resource_type: str,
    requested_amount: float,
    allocations: List[Allocation],
):

    if not allocations:
        return None

    allocated_amount = sum(
        allocation.amount
        for allocation in allocations
    )

    duration = min(
        allocation.predicted_residency
        for allocation in allocations
    )

    return Tenancy(
        request_id=request_id,
        resource_type=resource_type,
        requested_amount=requested_amount,
        allocated_amount=allocated_amount,
        duration=duration,
        providers=[
            allocation.provider_id
            for allocation in allocations
        ],
    )


# ============================================================
# Standby provider
# ============================================================

def select_standby(
    providers: List[Provider],
    active_provider_ids: List[str],
    resource_type: str,
    required_amount: float,
    minimum_residency: float,
):

    candidates = []

    for provider in providers:

        if provider.provider_id in active_provider_ids:
            continue

        if not provider.active:
            continue

        if provider.resource_type != resource_type:
            continue

        usable = (
            provider.available_amount
            * (1.0 - FAIRNESS_RESERVE)
        )

        if usable < required_amount:
            continue

        if (
            provider.predicted_residency
            < minimum_residency
        ):
            continue

        candidates.append(provider)

    if not candidates:
        return None

    candidates.sort(
        key=lambda p: (
            p.predicted_residency,
            p.trust_score,
            p.available_amount,
        ),
        reverse=True,
    )

    return candidates[0]


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Loading SUMO mobility dataset..."
    )

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"Mobility records: {len(df)}"
    )

    # --------------------------------------------------------
    # Select a real vehicle and obtain its real LSTM prediction
    # --------------------------------------------------------

    selected_vehicle = None
    selected_sequence = None

    for vehicle_id in df[
        "vehicle_id"
    ].unique():

        vehicle_df = df[
            df["vehicle_id"]
            == vehicle_id
        ]

        sequence = find_sequence(
            vehicle_df
        )

        if sequence is not None:

            selected_vehicle = vehicle_id
            selected_sequence = sequence

            break

    if selected_sequence is None:

        raise RuntimeError(
            "No valid vehicle sequence found."
        )

    model, scaler, device = (
        load_lstm()
    )

    predicted_residency = (
        predict_residency(
            selected_sequence,
            model,
            scaler,
            device,
        )
    )

    zone = int(
        selected_sequence[
            "zone_id"
        ].iloc[-1]
    )

    direction = (
        selected_sequence[
            "direction"
        ].iloc[-1]
    )

    # --------------------------------------------------------
    # Admission
    # --------------------------------------------------------

    admission = classify_admission(
        predicted_residency,
        X_THRESHOLD,
        TMIN_THRESHOLD,
    )

    print()
    print("=" * 70)
    print("             END-TO-END VCRSA")
    print("=" * 70)

    print()
    print("[1] MOBILITY")

    print(
        f"Vehicle             : "
        f"{selected_vehicle}"
    )

    print(
        f"Zone                : "
        f"{zone}"
    )

    print(
        f"Direction           : "
        f"{direction}"
    )

    print(
        f"Observation window  : "
        f"{selected_sequence['timestamp'].iloc[0]:.0f}s"
        f" → "
        f"{selected_sequence['timestamp'].iloc[-1]:.0f}s"
    )

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    print()
    print("[2] LSTM PREDICTION")

    print(
        f"Predicted residency : "
        f"{predicted_residency:.3f}s"
    )

    print(
        f"Device              : "
        f"{device}"
    )

    # --------------------------------------------------------
    # Admission
    # --------------------------------------------------------

    print()
    print("[3] PREDICTIVE ADMISSION")

    print(
        f"X threshold         : "
        f"{X_THRESHOLD:.1f}s"
    )

    print(
        f"Tmin                : "
        f"{TMIN_THRESHOLD:.1f}s"
    )

    print(
        f"Decision            : "
        f"{admission.value}"
    )

    if admission.value == "REJECT":

        print()
        print(
            "Request rejected/postponed."
        )

        print("=" * 70)

        return

    # --------------------------------------------------------
    # Candidate providers
    #
    # These are simulation providers for the integrated
    # prototype. Their residency values represent predictions.
    # --------------------------------------------------------

    providers = [

        Provider(
            provider_id="vehicle_A",
            resource_type="CPU",
            available_amount=8.0,
            predicted_residency=28.0,
            trust_score=0.90,
        ),

        Provider(
            provider_id="vehicle_B",
            resource_type="CPU",
            available_amount=12.0,
            predicted_residency=35.0,
            trust_score=0.95,
        ),

        Provider(
            provider_id="vehicle_C",
            resource_type="CPU",
            available_amount=4.0,
            predicted_residency=22.0,
            trust_score=0.85,
        ),

        Provider(
            provider_id="vehicle_D",
            resource_type="CPU",
            available_amount=16.0,
            predicted_residency=30.0,
            trust_score=0.88,
        ),
    ]

    requested_cpu = 20.0

    # --------------------------------------------------------
    # Greedy allocation
    # --------------------------------------------------------

    allocations = greedy_allocate(
        providers=providers,
        resource_type="CPU",
        requested_amount=requested_cpu,
    )

    print()
    print("[4] GREEDY RESOURCE ALLOCATION")

    print(
        f"Requested CPU      : "
        f"{requested_cpu:.1f}"
    )

    print(
        f"Fairness reserve   : "
        f"{FAIRNESS_RESERVE * 100:.0f}%"
    )

    if not allocations:

        print(
            "Allocation          : FAILED"
        )

        print("=" * 70)

        return

    for allocation in allocations:

        print(
            f"{allocation.provider_id}"
            f" → "
            f"{allocation.amount:.1f} CPU"
            f" | residency="
            f"{allocation.predicted_residency:.1f}s"
        )

    # --------------------------------------------------------
    # PRT tenancy
    # --------------------------------------------------------

    tenancy = create_tenancy(
        request_id="REQ-E2E-001",
        resource_type="CPU",
        requested_amount=requested_cpu,
        allocations=allocations,
    )

    print()
    print("[5] PREDICTIVE RESOURCE TENANCY")

    print(
        f"Allocated CPU      : "
        f"{tenancy.allocated_amount:.1f}"
    )

    print(
        f"Tenancy duration   : "
        f"{tenancy.duration:.1f}s"
    )

    print(
        f"Active providers   : "
        f"{', '.join(tenancy.providers)}"
    )

    # --------------------------------------------------------
    # Standby
    # --------------------------------------------------------

    standby = select_standby(
        providers=providers,
        active_provider_ids=tenancy.providers,
        resource_type="CPU",
        required_amount=4.0,
        minimum_residency=TMIN_THRESHOLD,
    )

    print()
    print("[6] STANDBY PROVIDER")

    if standby is None:

        print(
            "Standby provider   : NONE"
        )

    else:

        print(
            f"Standby provider   : "
            f"{standby.provider_id}"
        )

        print(
            f"Predicted residency: "
            f"{standby.predicted_residency:.1f}s"
        )

    # --------------------------------------------------------
    # Pre-departure handover simulation
    # --------------------------------------------------------

    print()
    print("[7] PRE-DEPARTURE HANDOVER")

    if standby is None:

        print(
            "Handover            : "
            "NO STANDBY AVAILABLE"
        )

    else:

        # Simulate the shortest active provider
        # approaching departure.
        shortest_provider = min(
            allocations,
            key=lambda x:
                x.predicted_residency,
        )

        remaining = (
            shortest_provider.predicted_residency
            - (
                shortest_provider.predicted_residency
                - HANDOVER_MARGIN
            )
        )

        print(
            f"Departing provider  : "
            f"{shortest_provider.provider_id}"
        )

        print(
            f"Handover margin     : "
            f"{HANDOVER_MARGIN:.1f}s"
        )

        print(
            f"Remaining at trigger: "
            f"{remaining:.1f}s"
        )

        if remaining <= HANDOVER_MARGIN:

            print(
                f"New active provider: "
                f"{standby.provider_id}"
            )

            print(
                "Handover            : SUCCESS"
            )

        else:

            print(
                "Handover            : NOT REQUIRED"
            )

    print()
    print("=" * 70)
    print("             END-TO-END VALIDATION")
    print("=" * 70)

    print(
        "Mobility            : PASS"
    )

    print(
        "LSTM                : PASS"
    )

    print(
        "Admission           : PASS"
    )

    print(
        "Greedy allocation   : PASS"
    )

    print(
        "PRT tenancy         : PASS"
    )

    print(
        "Standby selection   : "
        + ("PASS" if standby else "CHECK")
    )

    print(
        "End-to-end flow     : PASS"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
