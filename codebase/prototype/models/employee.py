from django.db import models
from django.db.models import JSONField

from prototype.utils.models import GenericModel


class Employee(GenericModel):
    organization = models.ForeignKey(
        'Organization',
        on_delete=models.DO_NOTHING,
        related_name='_employees'
    )

    first_name = models.CharField(max_length=255, db_index=True)
    last_name = models.CharField(max_length=255, db_index=True)
    amount = models.DecimalField(decimal_places=2, max_digits=100)

    recipient = models.CharField(max_length=255, db_index=True)
    address = models.CharField(max_length=255, db_index=True)

    # Ideally this is non-nullable but for now its required for migration from search objects
    email = models.EmailField(null=True)

    custom_data = JSONField(blank=True, default=dict)
    internal_data = JSONField(blank=True, default=dict)
    # ID from external source (eg. external employee id or username etc)
    external_id = models.CharField(max_length=255, db_index=True, null=True)
    # Archived flag
    archived = models.BooleanField(default=False)
    # Archived date
    archived_date = models.DateTimeField(blank=True, null=True)

    embedding = JSONField(null=True, blank=True)


    @property
    def organization_name(self):
        return self.organization.name if self.organization else None