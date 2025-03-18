from django.db import models
from django.utils.translation import gettext_lazy as _

class ContributionStatus(models.TextChoices):
    TO_DO = "to_do", _("To Do")
    APPROVED = "approved", _("Approved")
    NOT_APPROVED = "not_approved", _("Not Approved")
    UNDER_REVIEW = "under_review", _("Under Review")
    FALSE_POSITIVE = "false_positive", _("False Positive")