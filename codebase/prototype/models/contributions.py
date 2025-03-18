from prototype.utils.choices import ContributionStatus
from prototype.utils.models import GenericModel
from django.db import models
from django.db.models import JSONField


class Contributions(GenericModel):
    first_name = models.CharField(max_length=255, db_index=True)
    last_name = models.CharField(max_length=255, db_index=True)
    amount = models.DecimalField(decimal_places=2, max_digits=100)

    recipient = models.CharField(max_length=255, db_index=True)
    address = models.CharField(max_length=255, db_index=True)
    employer = models.CharField(max_length=255, db_index=True, null=True, blank=True)
    occupation = models.CharField(max_length=255, db_index=True, null=True, blank=True)

    # Ideally this is non-nullable but for now its required for migration from search objects
    email = models.EmailField(null=True)

    status = models.CharField(max_length=50, choices=ContributionStatus, default=ContributionStatus.TO_DO)

    embedding = JSONField(null=True, blank=True)
