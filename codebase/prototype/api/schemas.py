from typing import Optional, List
from uuid import UUID

from ninja import ModelSchema, Schema
from pydantic import BaseModel
from pydantic.v1.schema import schema

from prototype.models import Contributions, Employee
from prototype.models.contributions import Contributions
from prototype.utils.choices import ContributionStatus

DashboardResponse = str


class DashboardOutput(Schema):
    data: DashboardResponse


class ContributionsSchemaOutput(ModelSchema):
    prediction_label: Optional[int] = None
    class Meta:
        model = Contributions
        fields = (
            'id',
            'first_name',
            'last_name',
            'email',
            'amount',
            'recipient',
            'status',
            'address'
        )

class EmployeeSchemaOutput(ModelSchema):
    organization_name: str
    class Meta:
        model = Employee
        fields = (
            'first_name',
            'last_name',
            'email',
            'amount',
            'recipient',
            "address"
        )

class ContributionPrediction(BaseModel):
    contribution_id: UUID
    contributor_name: str
    contributor_address: str
    predicted_label: int
    prob_of_true_match: float


# Define your input schema
class ContributionsSchemaInput(BaseModel):
    contribution_id: UUID
    contribution_status: ContributionStatus

class BulkContributionsSchemaInput(BaseModel):
    contributions: List[ContributionsSchemaInput]