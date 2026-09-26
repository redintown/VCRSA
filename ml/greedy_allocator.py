from dataclasses import dataclass
from typing import List


@dataclass
class Provider:
    provider_id: str
    resource_type: str
    available_amount: float
    predicted_residency: float
    trust_score: float


@dataclass
class Allocation:
    provider_id: str
    resource_type: str
    allocated_amount: float
    predicted_residency: float


def provider_score(provider: Provider) -> float:
    """
    Greedy ranking score.

    Higher residency and higher trust are preferred.
    Resource capacity is also considered.
    """

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
    reserve_fraction: float = 0.10,
) -> List[Allocation]:

    eligible = []

    for provider in providers:

        if provider.resource_type != resource_type:
            continue

        usable_capacity = (
            provider.available_amount
            * (1.0 - reserve_fraction)
        )

        if usable_capacity <= 0:
            continue

        eligible.append(
            (
                provider,
                usable_capacity,
                provider_score(provider),
            )
        )

    # Highest score first
    eligible.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    remaining = requested_amount

    allocations = []

    for provider, usable_capacity, score in eligible:

        if remaining <= 0:
            break

        amount = min(
            usable_capacity,
            remaining,
        )

        allocations.append(
            Allocation(
                provider_id=provider.provider_id,
                resource_type=provider.resource_type,
                allocated_amount=amount,
                predicted_residency=provider.predicted_residency,
            )
        )

        remaining -= amount

    if remaining > 1e-9:
        return []

    return allocations


def calculate_tenancy(
    allocations: List[Allocation],
) -> float:

    if not allocations:
        return 0.0

    return min(
        allocation.predicted_residency
        for allocation in allocations
    )


if __name__ == "__main__":

    providers = [

        Provider(
            "vehicle_A",
            "CPU",
            8.0,
            28.0,
            0.90,
        ),

        Provider(
            "vehicle_B",
            "CPU",
            12.0,
            35.0,
            0.95,
        ),

        Provider(
            "vehicle_C",
            "CPU",
            4.0,
            22.0,
            0.85,
        ),

        Provider(
            "vehicle_D",
            "CPU",
            16.0,
            30.0,
            0.88,
        ),

        Provider(
            "vehicle_E",
            "MEMORY",
            32.0,
            45.0,
            0.92,
        ),
    ]

    requested_cpu = 20.0

    reserve_fraction = 0.10

    allocations = greedy_allocate(
        providers=providers,
        resource_type="CPU",
        requested_amount=requested_cpu,
        reserve_fraction=reserve_fraction,
    )

    print("=" * 65)
    print("          GREEDY RESOURCE ALLOCATION")
    print("=" * 65)

    print(
        f"Requested CPU      : {requested_cpu:.1f}"
    )

    print(
        f"Fairness reserve   : "
        f"{reserve_fraction * 100:.0f}%"
    )

    print()

    if not allocations:

        print(
            "Allocation status  : FAILED"
        )

    else:

        total = sum(
            allocation.allocated_amount
            for allocation in allocations
        )

        for allocation in allocations:

            print(
                f"{allocation.provider_id:10s}"
                f" → "
                f"{allocation.allocated_amount:5.1f} CPU"
                f" | residency="
                f"{allocation.predicted_residency:.1f}s"
            )

        tenancy = calculate_tenancy(
            allocations
        )

        print()
        print(
            f"Total allocated    : {total:.1f}"
        )

        print(
            f"PRT tenancy        : {tenancy:.1f}s"
        )

        if abs(
            total - requested_cpu
        ) < 1e-9:

            print(
                "Allocation validation: PASS"
            )

        else:

            print(
                "Allocation validation: CHECK"
            )

    print("=" * 65)
