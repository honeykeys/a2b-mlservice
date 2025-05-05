"""
Main interface for sqs service.

Usage::

    ```python
    from boto3.session import Session
    from mypy_boto3_sqs import (
        Client,
        ListDeadLetterSourceQueuesPaginator,
        ListQueuesPaginator,
        SQSClient,
        SQSServiceResource,
        ServiceResource,
    )

    session = Session()
    client: SQSClient = session.client("sqs")

    resource: SQSServiceResource = session.resource("sqs")

    list_dead_letter_source_queues_paginator: ListDeadLetterSourceQueuesPaginator = client.get_paginator("list_dead_letter_source_queues")
    list_queues_paginator: ListQueuesPaginator = client.get_paginator("list_queues")
    ```

Copyright 2025 Vlad Emelianov
"""

from .client import SQSClient
from .paginator import ListDeadLetterSourceQueuesPaginator, ListQueuesPaginator

try:
    from .service_resource import SQSServiceResource
except ImportError:
    from builtins import object as SQSServiceResource  # type: ignore[assignment]


Client = SQSClient


ServiceResource = SQSServiceResource


__all__ = (
    "Client",
    "ListDeadLetterSourceQueuesPaginator",
    "ListQueuesPaginator",
    "SQSClient",
    "SQSServiceResource",
    "ServiceResource",
)
