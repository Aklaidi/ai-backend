import status
from ninja import Schema


class DetailSchema(Schema):
    detail: str

BadRequestResponse = {status.HTTP_400_BAD_REQUEST: DetailSchema}
UnauthorizedResponse = {status.HTTP_401_UNAUTHORIZED: DetailSchema}
NotFoundResponse = {status.HTTP_404_NOT_FOUND: DetailSchema}

CommonResponse = {**BadRequestResponse, **UnauthorizedResponse, **NotFoundResponse}