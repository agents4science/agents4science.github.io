import uuid

from globus_sdk._testing.models import RegisteredResponse, ResponseSet

INDEX_ID = str(uuid.uuid4())

_default_display_name = "Awesome Index of Awesomeness"
_default_description = "An index so awesome that it simply cannot be described"

RESPONSES = ResponseSet(
    default=RegisteredResponse(
        service="search",
        method="PATCH",
        path=f"/v1/index/{INDEX_ID}",
        json={
            "@datatype": "GSearchIndex",
            "@version": "2017-09-01",
            "creation_date": "2021-04-05 15:05:18",
            "display_name": _default_display_name,
            "description": _default_description,
            "id": INDEX_ID,
            "is_trial": True,
            "subscription_id": None,
            "max_size_in_mb": 1,
            "num_entries": 0,
            "num_subjects": 0,
            "size_in_mb": 0,
            "status": "open",
        },
        metadata={"index_id": INDEX_ID, "display_name": _default_display_name},
    ),
    forbidden=RegisteredResponse(
        service="search",
        method="PATCH",
        path=f"/v1/index/{INDEX_ID}",
        status=403,
        json={
            "@datatype": "GError",
            "@version": "2017-09-01",
            "status": 403,
            "code": "Forbidden.Generic",
            "message": "index_update request denied by service",
            "request_id": "0e73b6a61e53468684f86c7993336a72",
            "error_data": {
                "cause": (
                    "You do not have the proper roles "
                    "to perform the index_update operation."
                ),
                "recommended_resolution": (
                    "Ensure you are making a call authenticated with "
                    "a valid Search token and that you have been granted "
                    "the required roles for this operation"
                ),
            },
        },
        metadata={"index_id": INDEX_ID},
    ),
)
