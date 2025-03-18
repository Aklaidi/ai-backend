from typing import List

from django.db import transaction

from prototype.api.schemas import ContributionsSchemaInput
from prototype.models import Contributions
from prototype.utils.choices import ContributionStatus


class ContributionService:

    @staticmethod
    @transaction.atomic
    def update_contribution_status(contribution: Contributions, status: ContributionStatus) -> Contributions:
        setattr(contribution, "status", status)
        contribution.save()
        return contribution


    @transaction.atomic
    @staticmethod
    def bulk_update_contributions(contributions: List[ContributionsSchemaInput]) -> Contributions:
        updated_contributions = []
        for contribution in contributions:
            con_obj = Contributions.objects.get(id=str(contribution.contribution_id))
            con_obj.status = contribution.contribution_status
            updated_contributions.append(con_obj)

        bulk_updated_contributions = Contributions.objects.bulk_update(updated_contributions, ["status"])
        return bulk_updated_contributions
