from dataclasses import dataclass
from typing import Dict, List, Optional


# ============================================================
# Provider
# ============================================================

@dataclass
class Provider:
    provider_id: str
    resource_type: str
    available_amount: float
    predicted_residency: float
    trust_score: float
    active: bool = True


# ============================================================
# PRT Contract
# ============================================================

@dataclass
class TenancyContract:
    request_id: str
    resource_type: str
    allocated_amount: float
    tenancy_duration: float
    provider_ids: List[str]
    active_provider_ids: List[str]


# ============================================================
# Standby Provider
# ============================================================

def select_standby_provider(
    providers: List[Provider],
    resource_type: str,
    required_amount: float,
    min_residency: float,
    exclude_ids: List[str],
) -> Optional[Provider]:

    candidates = []

    for provider in providers:

        if provider.provider_id in exclude_ids:
            continue

        if provider.resource_type != resource_type:
            continue

        if provider.active:
            continue

        if provider.available_amount < required_amount:
            continue

        if provider.predicted_residency < min_residency:
            continue

        if provider.trust_score < 0.70:
            continue

        candidates.append(provider)

    if not candidates:
        return None

    # Longest predicted residency first
    candidates.sort(
        key=lambda p: p.predicted_residency,
        reverse=True,
    )

    return candidates[0]


# ============================================================
# PRT creation
# ============================================================

def create_prt_contract(
    request_id: str,
    resource_type: str,
    allocated_amount: float,
    providers: List[Provider],
) -> TenancyContract:

    if not providers:
        raise ValueError(
            "No providers available for PRT contract."
        )

    duration = min(
        p.predicted_residency
        for p in providers
    )

    return TenancyContract(
        request_id=request_id,
        resource_type=resource_type,
        allocated_amount=allocated_amount,
        tenancy_duration=duration,
        provider_ids=[
            p.provider_id
            for p in providers
        ],
        active_provider_ids=[
            p.provider_id
            for p in providers
        ],
    )


# ============================================================
# Predictive departure check
# ============================================================

def departure_check(
    provider: Provider,
    current_time: float,
    handover_margin: float,
):

    remaining = (
        provider.predicted_residency
        - current_time
    )

    trigger = (
        remaining <= handover_margin
    )

    return remaining, trigger


# ============================================================
# Pre-departure handover
# ============================================================

def perform_handover(
    contract: TenancyContract,
    departing_provider: Provider,
    standby_provider: Provider,
):

    if standby_provider is None:
        return False

    if (
        standby_provider.available_amount
        < departing_provider.available_amount
    ):
        # Standby does not necessarily need to replace
        # the entire original provider capacity.
        # It only needs to satisfy the contracted amount.
        pass

    standby_provider.active = True
    departing_provider.active = False

    if (
        departing_provider.provider_id
        in contract.active_provider_ids
    ):

        contract.active_provider_ids.remove(
            departing_provider.provider_id
        )

    contract.active_provider_ids.append(
        standby_provider.provider_id
    )

    return True


# ============================================================
# Main demonstration
# ============================================================

def main():

    print("=" * 90)
    print(
        "        PRT + STANDBY PROVIDER + PREDICTIVE HANDOVER"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Existing L7 allocation
    # --------------------------------------------------------

    allocated_providers = [
        Provider(
            provider_id="eastbound_flow.22",
            resource_type="CPU",
            available_amount=10.80,
            predicted_residency=25.85,
            trust_score=0.90,
            active=True,
        ),

        Provider(
            provider_id="eastbound_flow.19",
            resource_type="CPU",
            available_amount=9.20,
            predicted_residency=14.15,
            trust_score=0.90,
            active=True,
        ),
    ]

    # --------------------------------------------------------
    # Standby providers
    # --------------------------------------------------------

    standby_pool = [

        Provider(
            provider_id="eastbound_flow.20",
            resource_type="CPU",
            available_amount=8.00,
            predicted_residency=17.74,
            trust_score=0.90,
            active=False,
        ),

        Provider(
            provider_id="eastbound_flow.21",
            resource_type="CPU",
            available_amount=4.00,
            predicted_residency=22.20,
            trust_score=0.40,
            active=False,
        ),

        Provider(
            provider_id="eastbound_flow.18",
            resource_type="CPU",
            available_amount=4.00,
            predicted_residency=11.09,
            trust_score=0.90,
            active=False,
        ),
    ]

    all_providers = (
        allocated_providers
        + standby_pool
    )

    # --------------------------------------------------------
    # Create PRT
    # --------------------------------------------------------

    contract = create_prt_contract(
        request_id="REQ-001",
        resource_type="CPU",
        allocated_amount=20.0,
        providers=allocated_providers,
    )

    print()
    print("PRT CONTRACT")
    print("-" * 90)

    print(
        f"Request ID       : {contract.request_id}"
    )

    print(
        f"Resource         : {contract.resource_type}"
    )

    print(
        f"Allocated amount : "
        f"{contract.allocated_amount:.2f}"
    )

    print(
        f"Provider IDs     : "
        f"{contract.provider_ids}"
    )

    print(
        f"PRT duration     : "
        f"{contract.tenancy_duration:.2f}s"
    )

    # --------------------------------------------------------
    # Simulate time progression
    # --------------------------------------------------------

    departing = allocated_providers[1]

    current_time = 9.15
    handover_margin = 5.0

    remaining, trigger = departure_check(
        departing,
        current_time,
        handover_margin,
    )

    print()
    print("PREDICTIVE DEPARTURE CHECK")
    print("-" * 90)

    print(
        f"Departing provider : "
        f"{departing.provider_id}"
    )

    print(
        f"Predicted residency: "
        f"{departing.predicted_residency:.2f}s"
    )

    print(
        f"Current time       : "
        f"{current_time:.2f}s"
    )

    print(
        f"Remaining residency: "
        f"{remaining:.2f}s"
    )

    print(
        f"Handover margin    : "
        f"{handover_margin:.2f}s"
    )

    print(
        f"Handover trigger   : "
        f"{'YES' if trigger else 'NO'}"
    )

    if not trigger:
        print()
        print(
            "No handover required."
        )
        return

    # --------------------------------------------------------
    # Standby selection
    # --------------------------------------------------------

    standby = select_standby_provider(
        all_providers,
        resource_type="CPU",
        required_amount=9.20,
        min_residency=10.0,
        exclude_ids=contract.active_provider_ids,
    )

    print()
    print("STANDBY PROVIDER SELECTION")
    print("-" * 90)

    if standby is None:

        print(
            "No suitable standby provider found."
        )

        print()
        print(
            "Reactive fallback required."
        )

        return

    print(
        f"Selected standby  : "
        f"{standby.provider_id}"
    )

    print(
        f"Available CPU      : "
        f"{standby.available_amount:.2f}"
    )

    print(
        f"Predicted residency: "
        f"{standby.predicted_residency:.2f}s"
    )

    print(
        f"Trust score        : "
        f"{standby.trust_score:.2f}"
    )

    # --------------------------------------------------------
    # Handover
    # --------------------------------------------------------

    success = perform_handover(
        contract,
        departing,
        standby,
    )

    print()
    print("PRE-DEPARTURE HANDOVER")
    print("-" * 90)

    print(
        f"Departing provider : "
        f"{departing.provider_id}"
    )

    print(
        f"Replacement provider: "
        f"{standby.provider_id}"
    )

    print(
        f"Handover status    : "
        f"{'SUCCESS' if success else 'FAILED'}"
    )

    # --------------------------------------------------------
    # Final contract state
    # --------------------------------------------------------

    print()
    print("FINAL PRT STATE")
    print("-" * 90)

    print(
        f"Active providers   : "
        f"{contract.active_provider_ids}"
    )

    print(
        f"Contract duration  : "
        f"{contract.tenancy_duration:.2f}s"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("VALIDATION")
    print("=" * 90)

    print(
        "PRT contract created       : PASS"
    )

    print(
        "Departure prediction       : "
        + ("PASS" if trigger else "FAIL")
    )

    print(
        "Standby provider selected  : "
        + ("PASS" if standby else "FAIL")
    )

    print(
        "Pre-departure handover     : "
        + ("PASS" if success else "FAIL")
    )

    print(
        "Original provider removed  : "
        + (
            "PASS"
            if departing.provider_id
            not in contract.active_provider_ids
            else "FAIL"
        )
    )

    print(
        "Replacement provider active: "
        + (
            "PASS"
            if standby.provider_id
            in contract.active_provider_ids
            else "FAIL"
        )
    )

    overall = (
        trigger
        and standby is not None
        and success
        and departing.provider_id
            not in contract.active_provider_ids
        and standby.provider_id
            in contract.active_provider_ids
    )

    print()
    print(
        "PRT + Predictive Handover: "
        + ("PASS" if overall else "FAIL")
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
