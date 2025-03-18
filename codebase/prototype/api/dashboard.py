from http.client import HTTPException
from typing import List

import status
from joblib import load
from ninja import Router, Body

from prototype.api.schemas import DashboardOutput, ContributionsSchemaOutput, EmployeeSchemaOutput, \
    ContributionsSchemaInput, BulkContributionsSchemaInput, ContributionPrediction
from prototype.models import Employee
from prototype.models.contributions import Contributions
from prototype.services.ai import AiService
from prototype.services.contribution import ContributionService
from prototype.utils.api import CommonResponse
from prototype.utils.choices import ContributionStatus
from prototype.utils.management import get_csv_path

dashboard_router = Router()

ai_service = AiService()

try:
    model_path = get_csv_path(filename="incremental_model.pkl", extra_path=None)
    MATCH_MODEL = load(model_path)  # or the correct path to model.joblib
except FileNotFoundError:
    MATCH_MODEL = None

@dashboard_router.get("/", response={status.HTTP_200_OK: DashboardOutput})
def get_dashboard(request):
    return status.HTTP_200_OK, {"data" :"hello"}


@dashboard_router.get(
    "/contribution",
    response={status.HTTP_200_OK: list[ContributionsSchemaOutput], **CommonResponse}
)
def get_dashboard_contributions(request):
    contributions = Contributions.objects.all()
    return status.HTTP_200_OK, contributions


@dashboard_router.get("/employees", response={status.HTTP_200_OK: list[EmployeeSchemaOutput], **CommonResponse})
def get_dashboard_employee(request):
    contributions = Employee.objects.select_related("organization").all()
    return status.HTTP_200_OK, contributions

@dashboard_router.get(
    "/employee/{employee_email}/false_positives",
    response={status.HTTP_200_OK: list[ContributionsSchemaOutput | ContributionPrediction], **CommonResponse}
)
def check_false_positives(request, employee_email: str, show_probability: bool = False):
    try:
        employee = Employee.objects.get(email=employee_email)
    except Employee.DoesNotExist:
        raise HTTPException(status_code=404, detail="Employee not found.")

    contributions = Contributions.objects.filter(
        first_name__icontains=employee.first_name,
        # last_name__icontains=employee.last_name,
        status=ContributionStatus.TO_DO
    )

    predictions = ai_service._predict_false_positives_using_semantic_embedding(
        employee=employee,
        contributions=contributions,
        show_probability=show_probability
    )

    return predictions

@dashboard_router.put(
    "/contribution/{contribution_email}",
    response={status.HTTP_200_OK: ContributionsSchemaOutput, **CommonResponse}
)
def update_dashboard_contributions(
        request,
        contribution_email: str,
        contribution_status: ContributionStatus,
):
    contribution = Contributions.objects.get(email=contribution_email)
    updated_contribution = ContributionService.update_contribution_status(
        contribution=contribution,
        status=contribution_status
    )
    return updated_contribution


@dashboard_router.put(
    "/contribution/bulk_update/status",
    response={status.HTTP_204_NO_CONTENT: int, **CommonResponse},
)
def update_bulk_status_contributions(
    request,
    payload: BulkContributionsSchemaInput = Body(...),
):
    updated_contributions = ContributionService.bulk_update_contributions(
        contributions=payload.contributions,
    )
    return status.HTTP_204_NO_CONTENT, updated_contributions
