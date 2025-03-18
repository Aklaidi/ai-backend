from django.contrib import admin

from prototype.models import Organization, Contributions, Employee
from prototype.utils.training_data.training_data_generator import first_names


# Register your models here.

@admin.register(Organization)
class OrganizationAdmin(admin.ModelAdmin):
    list_display = ["id", "name",]
    search_fields = ["name"]


@admin.register(Employee)
class EmployeeAdmin(admin.ModelAdmin):
    list_display = [
        'first_name',
        'last_name',
        'amount',
        'organization',
        'email',
        'recipient',
        'internal_data',
        'external_id',
        'archived',
        'archived_date',
    ]
    search_fields = ["first_name", "last_name", "email", "organization__name"]


@admin.register(Contributions)
class ContributionsAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'first_name',
        'last_name',
        'amount',
        'email',
        'recipient',
        'address',
        'status'
    ]
    search_fields = ["first_name", "last_name", "email", "recipient"]
    list_filter = ["status", "first_name"]