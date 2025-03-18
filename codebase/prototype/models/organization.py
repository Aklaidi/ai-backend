from django.db import models
from prototype.utils.models import GenericModel



class Organization(GenericModel):
    name = models.CharField(max_length=80, unique=True)

    def __str__(self):
        return self.name