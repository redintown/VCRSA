from dataclasses import dataclass
from typing import List


@dataclass
class ResourceProvider:
    provider_id: str
    resource_type: str
    available_amount: float
    predicted_residency: float


@dataclass
class TenancyContract:
    request_id: str
    resource_type: str
    allocated_amount: float
    tenancy_duration: float
    providers: List[str]


def calculate_tenancy_duration(
    providers: List[ResourceProvider],
) -> float:
    """
    PRT rule:

    Tenancy duration is bounded by the shortest
    predicted residency among participating providers.
    """

    if not providers:
        return 0.0

    return min(
        provider.predicted_residency
        for provider in providers
    )


def create_tenancy_contract(
    request_id: str,
    providers: List[ResourceProvider],
) -> TenancyContract:

    if not providers:
        raise ValueError(
            "At least one provider is required."
        )

    resource_type = providers[0].resource_type

    if any(
        provider.resource_type != resource_type
        for provider in providers
    ):
        raise ValueError(
            "All providers must provide the same "
            "resource type in this prototype."
        )

    allocated_amount = sum(
        provider.available_amount
        for provider in providers
    )

    duration = calculate_tenancy_duration(
        providers
    )

    return TenancyContract(
        request_id=request_id,
        resource_type=resource_type,
        allocated_amount=allocated_amount,
        tenancy_duration=duration,
        providers=[
            provider.provider_id
            for provider in providers
        ],
    )


def check_pre_departure_handover(
    current_time: float,
    contract: TenancyContract,
    provider: ResourceProvider,
    handover_margin: float = 5.0,
) -> bool:
    """
    Returns True when pre-departure handover
    should be triggered.

    Handover is triggered when the provider's
    predicted remaining residency is within
    the configured margin.
    """

    remaining_time = (
        provider.predicted_residency
        - current_time
    )

    return remaining_time <= handover_margin


if __name__ == "__main__":

    # ------------------------------------------------------------
    # Example providers
    # ------------------------------------------------------------

    providers = [
        ResourceProvider(
            provider_id="vehicle_A",
            resource_type="CPU",
            available_amount=8.0,
            predicted_residency=28.0,
        ),
        ResourceProvider(
            provider_id="vehicle_B",
            resource_type="CPU",
            available_amount=12.0,
            predicted_residency=35.0,
        ),
        ResourceProvider(
            provider_id="vehicle_C",
            resource_type="CPU",
            available_amount=4.0,
            predicted_residency=22.0,
        ),
    ]

    # ------------------------------------------------------------
    # Create tenancy
    # ------------------------------------------------------------

    contract = create_tenancy_contract(
        request_id="REQ-001",
        providers=providers,
    )

    print("=" * 60)
    print("          PREDICTIVE RESOURCE TENANCY")
    print("=" * 60)

    print(
        f"Request ID       : "
        f"{contract.request_id}"
    )

    print(
        f"Resource type    : "
        f"{contract.resource_type}"
    )

    print(
        f"Allocated amount : "
        f"{contract.allocated_amount:.1f}"
    )

    print(
        f"Providers        : "
        f"{', '.join(contract.providers)}"
    )

    print(
        f"Tenancy duration : "
        f"{contract.tenancy_duration:.1f} seconds"
    )

    print()

    print("Provider predictions:")

    for provider in providers:

        print(
            f"  {provider.provider_id}"
            f" → {provider.predicted_residency:.1f}s"
        )

    # ------------------------------------------------------------
    # Verify shortest-residency rule
    # ------------------------------------------------------------

    expected_duration = 22.0

    if (
        contract.tenancy_duration
        == expected_duration
    ):
        print()
        print(
            "Tenancy duration validation: PASS"
        )
    else:
        print()
        print(
            "Tenancy duration validation: CHECK"
        )

    # ------------------------------------------------------------
    # Handover test
    # ------------------------------------------------------------

    current_time = 18.0

    departing_provider = providers[2]

    handover_required = (
        check_pre_departure_handover(
            current_time=current_time,
            contract=contract,
            provider=departing_provider,
            handover_margin=5.0,
        )
    )

    print()
    print(
        f"Current time     : "
        f"{current_time:.1f}s"
    )

    print(
        f"Provider          : "
        f"{departing_provider.provider_id}"
    )

    print(
        f"Handover required : "
        f"{'YES' if handover_required else 'NO'}"
    )

    print("=" * 60)
