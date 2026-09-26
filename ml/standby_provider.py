from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ResourceProvider:
    provider_id: str
    resource_type: str
    available_amount: float
    predicted_residency: float
    active: bool = True


def select_standby_provider(
    providers: List[ResourceProvider],
    active_provider_id: str,
    required_amount: float,
    minimum_residency: float,
) -> Optional[ResourceProvider]:
    """
    Select a standby provider for predictive handover.

    Eligibility:
      1. Same resource type as active provider.
      2. Not the active provider.
      3. Enough available resource.
      4. Predicted residency >= minimum required residency.

    Selection:
      Highest predicted residency among eligible providers.
    """

    active_provider = None

    for provider in providers:
        if provider.provider_id == active_provider_id:
            active_provider = provider
            break

    if active_provider is None:
        raise ValueError(
            f"Active provider '{active_provider_id}' not found."
        )

    candidates = []

    for provider in providers:

        if provider.provider_id == active_provider_id:
            continue

        if provider.resource_type != active_provider.resource_type:
            continue

        if provider.available_amount < required_amount:
            continue

        if provider.predicted_residency < minimum_residency:
            continue

        candidates.append(provider)

    if not candidates:
        return None

    candidates.sort(
        key=lambda provider: provider.predicted_residency,
        reverse=True,
    )

    return candidates[0]


def perform_handover(
    active_provider: ResourceProvider,
    standby_provider: ResourceProvider,
) -> tuple[ResourceProvider, ResourceProvider]:

    print()
    print("PRE-DEPARTURE HANDOVER")
    print("-" * 50)

    print(
        f"Old provider : "
        f"{active_provider.provider_id}"
    )

    print(
        f"Standby      : "
        f"{standby_provider.provider_id}"
    )

    # Standby becomes active.
    standby_provider.active = True

    # Old provider leaves the active service path.
    active_provider.active = False

    print(
        f"New active provider : "
        f"{standby_provider.provider_id}"
    )

    print(
        "Handover status     : SUCCESS"
    )

    return active_provider, standby_provider


if __name__ == "__main__":

    # ------------------------------------------------------------
    # Example provider set
    # ------------------------------------------------------------

    providers = [

        ResourceProvider(
            provider_id="vehicle_A",
            resource_type="CPU",
            available_amount=8.0,
            predicted_residency=12.0,
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

        ResourceProvider(
            provider_id="vehicle_D",
            resource_type="CPU",
            available_amount=16.0,
            predicted_residency=30.0,
        ),

        ResourceProvider(
            provider_id="vehicle_E",
            resource_type="MEMORY",
            available_amount=32.0,
            predicted_residency=50.0,
        ),
    ]

    active_provider_id = "vehicle_A"

    required_amount = 4.0

    minimum_residency = 10.0

    # ------------------------------------------------------------
    # Standby selection
    # ------------------------------------------------------------

    standby = select_standby_provider(
        providers=providers,
        active_provider_id=active_provider_id,
        required_amount=required_amount,
        minimum_residency=minimum_residency,
    )

    print("=" * 60)
    print("          STANDBY PROVIDER SELECTION")
    print("=" * 60)

    print(
        f"Active provider       : "
        f"{active_provider_id}"
    )

    print(
        f"Required CPU          : "
        f"{required_amount}"
    )

    print(
        f"Minimum residency     : "
        f"{minimum_residency}s"
    )

    if standby is None:

        print()
        print(
            "No eligible standby provider found."
        )

    else:

        print()
        print(
            "Selected standby provider:"
        )

        print(
            f"  Provider            : "
            f"{standby.provider_id}"
        )

        print(
            f"  Resource            : "
            f"{standby.resource_type}"
        )

        print(
            f"  Available resource  : "
            f"{standby.available_amount}"
        )

        print(
            f"  Predicted residency : "
            f"{standby.predicted_residency}s"
        )

        # --------------------------------------------------------
        # Handover
        # --------------------------------------------------------

        old_provider, new_provider = perform_handover(
            active_provider=providers[0],
            standby_provider=standby,
        )

        if (
            not old_provider.active
            and new_provider.active
        ):
            print()
            print(
                "Standby handover validation: PASS"
            )
        else:
            print()
            print(
                "Standby handover validation: CHECK"
            )

    print("=" * 60)
