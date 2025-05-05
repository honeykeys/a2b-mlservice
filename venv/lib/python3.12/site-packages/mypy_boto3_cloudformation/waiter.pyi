"""
Type annotations for cloudformation service client waiters.

[Documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_cloudformation/waiters/)

Usage::

    ```python
    from boto3.session import Session

    from mypy_boto3_cloudformation.client import CloudFormationClient
    from mypy_boto3_cloudformation.waiter import (
        ChangeSetCreateCompleteWaiter,
        StackCreateCompleteWaiter,
        StackDeleteCompleteWaiter,
        StackExistsWaiter,
        StackImportCompleteWaiter,
        StackRollbackCompleteWaiter,
        StackUpdateCompleteWaiter,
        TypeRegistrationCompleteWaiter,
    )

    session = Session()
    client: CloudFormationClient = session.client("cloudformation")

    change_set_create_complete_waiter: ChangeSetCreateCompleteWaiter = client.get_waiter("change_set_create_complete")
    stack_create_complete_waiter: StackCreateCompleteWaiter = client.get_waiter("stack_create_complete")
    stack_delete_complete_waiter: StackDeleteCompleteWaiter = client.get_waiter("stack_delete_complete")
    stack_exists_waiter: StackExistsWaiter = client.get_waiter("stack_exists")
    stack_import_complete_waiter: StackImportCompleteWaiter = client.get_waiter("stack_import_complete")
    stack_rollback_complete_waiter: StackRollbackCompleteWaiter = client.get_waiter("stack_rollback_complete")
    stack_update_complete_waiter: StackUpdateCompleteWaiter = client.get_waiter("stack_update_complete")
    type_registration_complete_waiter: TypeRegistrationCompleteWaiter = client.get_waiter("type_registration_complete")
    ```

Copyright 2025 Vlad Emelianov
"""

from __future__ import annotations

import sys

from botocore.waiter import Waiter

from .type_defs import (
    DescribeChangeSetInputWaitTypeDef,
    DescribeStacksInputWaitTypeDef,
    DescribeTypeRegistrationInputWaitTypeDef,
)

if sys.version_info >= (3, 12):
    from typing import Unpack
else:
    from typing_extensions import Unpack

__all__ = (
    "ChangeSetCreateCompleteWaiter",
    "StackCreateCompleteWaiter",
    "StackDeleteCompleteWaiter",
    "StackExistsWaiter",
    "StackImportCompleteWaiter",
    "StackRollbackCompleteWaiter",
    "StackUpdateCompleteWaiter",
    "TypeRegistrationCompleteWaiter",
)

class ChangeSetCreateCompleteWaiter(Waiter):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudformation/waiter/ChangeSetCreateComplete.html#CloudFormation.Waiter.ChangeSetCreateComplete)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_cloudformation/waiters/#changesetcreatecompletewaiter)
    """
    def wait(  # type: ignore[override]
        self, **kwargs: Unpack[DescribeChangeSetInputWaitTypeDef]
    ) -> None:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudformation/waiter/ChangeSetCreateComplete.html#CloudFormation.Waiter.ChangeSetCreateComplete.wait)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_cloudformation/waiters/#changesetcreatecompletewaiter)
        """

class StackCreateCompleteWaiter(Waiter):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudformation/waiter/StackCreateComplete.html#CloudFormation.Waiter.StackCreateComplete)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_cloudformation/waiters/#stackcreatecompletewaiter)
    """
    def wait(  # type: ignore[override]
        self, **kwargs: Unpack[DescribeStacksInputWaitTypeDef]
    ) -> None:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudformation/waiter/StackCreateComplete.html#CloudFormation.Waiter.StackCreateComplete.wait)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_cloudformation/waiters/#stackcreatecompletewaiter)
        """

class StackDeleteCompleteWaiter(Waiter):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudformation/waiter/StackDeleteComplete.html#CloudFormation.Waiter.StackDeleteComplete)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_cloudformation/waiters/#stackdeletecompletewaiter)
    """
    def wait(  # type: ignore[override]
        self, **kwargs: Unpack[DescribeStacksInputWaitTypeDef]
    ) -> None:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudformation/waiter/StackDeleteComplete.html#CloudFormation.Waiter.StackDeleteComplete.wait)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_cloudformation/waiters/#stackdeletecompletewaiter)
        """

class StackExistsWaiter(Waiter):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudformation/waiter/StackExists.html#CloudFormation.Waiter.StackExists)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_cloudformation/waiters/#stackexistswaiter)
    """
    def wait(  # type: ignore[override]
        self, **kwargs: Unpack[DescribeStacksInputWaitTypeDef]
    ) -> None:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudformation/waiter/StackExists.html#CloudFormation.Waiter.StackExists.wait)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_cloudformation/waiters/#stackexistswaiter)
        """

class StackImportCompleteWaiter(Waiter):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudformation/waiter/StackImportComplete.html#CloudFormation.Waiter.StackImportComplete)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_cloudformation/waiters/#stackimportcompletewaiter)
    """
    def wait(  # type: ignore[override]
        self, **kwargs: Unpack[DescribeStacksInputWaitTypeDef]
    ) -> None:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudformation/waiter/StackImportComplete.html#CloudFormation.Waiter.StackImportComplete.wait)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_cloudformation/waiters/#stackimportcompletewaiter)
        """

class StackRollbackCompleteWaiter(Waiter):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudformation/waiter/StackRollbackComplete.html#CloudFormation.Waiter.StackRollbackComplete)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_cloudformation/waiters/#stackrollbackcompletewaiter)
    """
    def wait(  # type: ignore[override]
        self, **kwargs: Unpack[DescribeStacksInputWaitTypeDef]
    ) -> None:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudformation/waiter/StackRollbackComplete.html#CloudFormation.Waiter.StackRollbackComplete.wait)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_cloudformation/waiters/#stackrollbackcompletewaiter)
        """

class StackUpdateCompleteWaiter(Waiter):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudformation/waiter/StackUpdateComplete.html#CloudFormation.Waiter.StackUpdateComplete)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_cloudformation/waiters/#stackupdatecompletewaiter)
    """
    def wait(  # type: ignore[override]
        self, **kwargs: Unpack[DescribeStacksInputWaitTypeDef]
    ) -> None:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudformation/waiter/StackUpdateComplete.html#CloudFormation.Waiter.StackUpdateComplete.wait)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_cloudformation/waiters/#stackupdatecompletewaiter)
        """

class TypeRegistrationCompleteWaiter(Waiter):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudformation/waiter/TypeRegistrationComplete.html#CloudFormation.Waiter.TypeRegistrationComplete)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_cloudformation/waiters/#typeregistrationcompletewaiter)
    """
    def wait(  # type: ignore[override]
        self, **kwargs: Unpack[DescribeTypeRegistrationInputWaitTypeDef]
    ) -> None:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudformation/waiter/TypeRegistrationComplete.html#CloudFormation.Waiter.TypeRegistrationComplete.wait)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_cloudformation/waiters/#typeregistrationcompletewaiter)
        """
