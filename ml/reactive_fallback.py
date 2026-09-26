from dataclasses import dataclass
from typing import List, Optional


# ============================================================
# Secure Provider
# ============================================================

@dataclass
class SecureProvider:
    provider_id: str
    zone: int
    resource_type: str
    available_amount: float
    predicted_residency: float
    trust_score: float
    position_valid: bool
    por_verified: bool
    admission_allowed: bool


# ============================================================
# Security qualification
# ============================================================

def is_secure(provider: SecureProvider) -> bool:

    return (
        provider.trust_score >= 0.70
        and provider.position_valid
        and provider.por_verified
        and provider.admission_allowed
    )


# ============================================================
# Reactive fallback search
# ============================================================

def reactive_search(
    providers: List[SecureProvider],
    source_zone: int,
    resource_type: str,
    required_amount: float,
    max_hops: int = 3,
):

    candidates = []

    for provider in providers:

        if not is_secure(provider):
            continue

        if provider.resource_type != resource_type:
            continue

        if provider.available_amount <= 0:
            continue

        if provider.zone == source_zone:
            hop = 0

        else:
            hop = abs(
                provider.zone - source_zone
            )

        if hop > max_hops:
            continue

        candidates.append(
            (
                provider,
                hop,
            )
        )

    # Prefer lower hop count,
    # then longer predicted residency,
    # then larger capacity.
    candidates.sort(
        key=lambda item: (
            item[1],
            -item[0].predicted_residency,
            -item[0].available_amount,
        )
    )

    selected = []

    remaining = required_amount

    for provider, hop in candidates:

        if remaining <= 0:
            break

        amount = min(
            provider.available_amount,
            remaining,
        )

        selected.append(
            {
                "provider": provider,
                "hop": hop,
                "amount": amount,
            }
        )

        remaining -= amount

    return selected, remaining


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 95)
    print(
        "             REACTIVE FALLBACK RESOURCE SEARCH"
    )
    print("=" * 95)

    # --------------------------------------------------------
    # Failed standby situation
    # --------------------------------------------------------

    print()
    print("PRE-DEPARTURE HANDOVER")
    print("-" * 95)

    print(
        "Departing provider : eastbound_flow.19"
    )

    print(
        "Standby search     : FAILED"
    )

    print(
        "Triggering reactive fallback..."
    )

    # --------------------------------------------------------
    # Secure providers from multiple zones
    #
    # These values represent the already security-qualified
    # provider state from the previous pipeline.
    # --------------------------------------------------------

    providers = [

        # Zone 0
        SecureProvider(
            provider_id="eastbound_flow.22",
            zone=0,
            resource_type="CPU",
            available_amount=1.60,
            predicted_residency=25.85,
            trust_score=0.90,
            position_valid=True,
            por_verified=True,
            admission_allowed=True,
        ),

        # Zone 1
        SecureProvider(
            provider_id="eastbound_flow.10",
            zone=1,
            resource_type="CPU",
            available_amount=8.40,
            predicted_residency=14.83,
            trust_score=0.90,
            position_valid=True,
            por_verified=True,
            admission_allowed=True,
        ),

        SecureProvider(
            provider_id="eastbound_flow.11",
            zone=1,
            resource_type="CPU",
            available_amount=6.00,
            predicted_residency=18.20,
            trust_score=0.90,
            position_valid=True,
            por_verified=True,
            admission_allowed=True,
        ),

        # Zone 2
        SecureProvider(
            provider_id="eastbound_flow.5",
            zone=2,
            resource_type="CPU",
            available_amount=8.00,
            predicted_residency=22.40,
            trust_score=0.90,
            position_valid=True,
            por_verified=True,
            admission_allowed=True,
        ),

        # Deliberately insecure provider
        SecureProvider(
            provider_id="attacker.01",
            zone=1,
            resource_type="CPU",
            available_amount=20.0,
            predicted_residency=40.0,
            trust_score=0.20,
            position_valid=True,
            por_verified=False,
            admission_allowed=True,
        ),
    ]

    # --------------------------------------------------------
    # Required replacement resource
    # --------------------------------------------------------

    required_cpu = 9.20

    print()
    print("FALLBACK REQUEST")
    print("-" * 95)

    print(
        f"Source zone       : 0"
    )

    print(
        f"Resource type     : CPU"
    )

    print(
        f"Required CPU      : {required_cpu:.2f}"
    )

    print(
        f"Maximum hops      : 3"
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    selected, remaining = reactive_search(
        providers=providers,
        source_zone=0,
        resource_type="CPU",
        required_amount=required_cpu,
        max_hops=3,
    )

    print()
    print("REACTIVE SEARCH RESULTS")
    print("-" * 95)

    for item in selected:

        provider = item["provider"]

        print(
            f"Provider: "
            f"{provider.provider_id:<24}"
            f"Zone: {provider.zone}  "
            f"Hop: {item['hop']}  "
            f"CPU: {item['amount']:.2f}  "
            f"Residency: "
            f"{provider.predicted_residency:.2f}s"
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    allocated = (
        required_cpu
        - remaining
    )

    print()
    print("FALLBACK RESULT")
    print("-" * 95)

    print(
        f"Allocated CPU     : "
        f"{allocated:.2f}"
    )

    print(
        f"Remaining CPU     : "
        f"{remaining:.2f}"
    )

    if selected:

        prt = min(
            item["provider"].predicted_residency
            for item in selected
        )

        print(
            f"New PRT duration  : "
            f"{prt:.2f}s"
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    fallback_success = (
        remaining <= 1e-9
    )

    only_secure = all(
        is_secure(
            item["provider"]
        )
        for item in selected
    )

    hop_valid = all(
        item["hop"] <= 3
        for item in selected
    )

    attacker_blocked = not any(
        item["provider"].provider_id
        == "attacker.01"
        for item in selected
    )

    print()
    print("=" * 95)
    print("VALIDATION")
    print("=" * 95)

    print(
        "Standby unavailable       : PASS"
    )

    print(
        "Reactive fallback started : PASS"
    )

    print(
        "Fallback resource found   : "
        + (
            "PASS"
            if fallback_success
            else "FAIL"
        )
    )

    print(
        "Only secure providers     : "
        + (
            "PASS"
            if only_secure
            else "FAIL"
        )
    )

    print(
        "Hop budget <= 3           : "
        + (
            "PASS"
            if hop_valid
            else "FAIL"
        )
    )

    print(
        "Insecure provider blocked : "
        + (
            "PASS"
            if attacker_blocked
            else "FAIL"
        )
    )

    print()
    print(
        "Reactive Fallback: "
        + (
            "PASS"
            if (
                fallback_success
                and only_secure
                and hop_valid
                and attacker_blocked
            )
            else "FAIL"
        )
    )

    print("=" * 95)


if __name__ == "__main__":
    main()
