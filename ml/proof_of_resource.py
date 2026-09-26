from dataclasses import dataclass
from enum import Enum


class PoRStatus(Enum):
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"


@dataclass
class ResourceClaim:
    vehicle_id: str
    resource_type: str
    claimed_amount: float


@dataclass
class ResourceProof:
    vehicle_id: str
    resource_type: str
    claimed_amount: float
    measured_amount: float
    status: PoRStatus


class ProofOfResource:

    def __init__(self, tolerance=0.05):
        # 5% measurement tolerance
        self.tolerance = tolerance

    def verify(
        self,
        claim: ResourceClaim,
        measured_amount: float,
    ):

        lower_bound = (
            claim.claimed_amount
            * (1.0 - self.tolerance)
        )

        if measured_amount >= lower_bound:

            return ResourceProof(
                vehicle_id=claim.vehicle_id,
                resource_type=claim.resource_type,
                claimed_amount=claim.claimed_amount,
                measured_amount=measured_amount,
                status=PoRStatus.VERIFIED,
            )

        return ResourceProof(
            vehicle_id=claim.vehicle_id,
            resource_type=claim.resource_type,
            claimed_amount=claim.claimed_amount,
            measured_amount=measured_amount,
            status=PoRStatus.REJECTED,
        )


def demo():

    por = ProofOfResource(
        tolerance=0.05
    )

    claims = [
        ResourceClaim(
            vehicle_id="vehicle_001",
            resource_type="CPU",
            claimed_amount=12.0,
        ),
        ResourceClaim(
            vehicle_id="vehicle_002",
            resource_type="CPU",
            claimed_amount=8.0,
        ),
        ResourceClaim(
            vehicle_id="vehicle_003",
            resource_type="CPU",
            claimed_amount=12.0,
        ),
    ]

    measured = {
        "vehicle_001": 12.0,
        "vehicle_002": 8.2,
        "vehicle_003": 5.0,
    }

    print()
    print("=" * 75)
    print("                 PROOF-OF-RESOURCE")
    print("=" * 75)

    for claim in claims:

        result = por.verify(
            claim,
            measured[claim.vehicle_id],
        )

        print()
        print(
            f"Vehicle          : "
            f"{result.vehicle_id}"
        )

        print(
            f"Resource         : "
            f"{result.resource_type}"
        )

        print(
            f"Claimed          : "
            f"{result.claimed_amount:.2f}"
        )

        print(
            f"Measured         : "
            f"{result.measured_amount:.2f}"
        )

        print(
            f"PoR status       : "
            f"{result.status.value}"
        )

    print()
    print("=" * 75)


if __name__ == "__main__":
    demo()
