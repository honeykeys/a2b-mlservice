"""
Type annotations for lambda service Client.

[Documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/)

Usage::

    ```python
    from boto3.session import Session
    from mypy_boto3_lambda.client import LambdaClient

    session = Session()
    client: LambdaClient = session.client("lambda")
    ```

Copyright 2025 Vlad Emelianov
"""

from __future__ import annotations

import sys
from typing import Any, overload

from botocore.client import BaseClient, ClientMeta
from botocore.errorfactory import BaseClientExceptions
from botocore.exceptions import ClientError as BotocoreClientError

from .paginator import (
    ListAliasesPaginator,
    ListCodeSigningConfigsPaginator,
    ListEventSourceMappingsPaginator,
    ListFunctionEventInvokeConfigsPaginator,
    ListFunctionsByCodeSigningConfigPaginator,
    ListFunctionsPaginator,
    ListFunctionUrlConfigsPaginator,
    ListLayersPaginator,
    ListLayerVersionsPaginator,
    ListProvisionedConcurrencyConfigsPaginator,
    ListVersionsByFunctionPaginator,
)
from .type_defs import (
    AddLayerVersionPermissionRequestRequestTypeDef,
    AddLayerVersionPermissionResponseTypeDef,
    AddPermissionRequestRequestTypeDef,
    AddPermissionResponseTypeDef,
    AliasConfigurationResponseTypeDef,
    ConcurrencyResponseTypeDef,
    CreateAliasRequestRequestTypeDef,
    CreateCodeSigningConfigRequestRequestTypeDef,
    CreateCodeSigningConfigResponseTypeDef,
    CreateEventSourceMappingRequestRequestTypeDef,
    CreateFunctionRequestRequestTypeDef,
    CreateFunctionUrlConfigRequestRequestTypeDef,
    CreateFunctionUrlConfigResponseTypeDef,
    DeleteAliasRequestRequestTypeDef,
    DeleteCodeSigningConfigRequestRequestTypeDef,
    DeleteEventSourceMappingRequestRequestTypeDef,
    DeleteFunctionCodeSigningConfigRequestRequestTypeDef,
    DeleteFunctionConcurrencyRequestRequestTypeDef,
    DeleteFunctionEventInvokeConfigRequestRequestTypeDef,
    DeleteFunctionRequestRequestTypeDef,
    DeleteFunctionUrlConfigRequestRequestTypeDef,
    DeleteLayerVersionRequestRequestTypeDef,
    DeleteProvisionedConcurrencyConfigRequestRequestTypeDef,
    EmptyResponseMetadataTypeDef,
    EventSourceMappingConfigurationResponseTypeDef,
    FunctionConfigurationResponseTypeDef,
    FunctionEventInvokeConfigResponseTypeDef,
    GetAccountSettingsResponseTypeDef,
    GetAliasRequestRequestTypeDef,
    GetCodeSigningConfigRequestRequestTypeDef,
    GetCodeSigningConfigResponseTypeDef,
    GetEventSourceMappingRequestRequestTypeDef,
    GetFunctionCodeSigningConfigRequestRequestTypeDef,
    GetFunctionCodeSigningConfigResponseTypeDef,
    GetFunctionConcurrencyRequestRequestTypeDef,
    GetFunctionConcurrencyResponseTypeDef,
    GetFunctionConfigurationRequestRequestTypeDef,
    GetFunctionEventInvokeConfigRequestRequestTypeDef,
    GetFunctionRecursionConfigRequestRequestTypeDef,
    GetFunctionRecursionConfigResponseTypeDef,
    GetFunctionRequestRequestTypeDef,
    GetFunctionResponseTypeDef,
    GetFunctionUrlConfigRequestRequestTypeDef,
    GetFunctionUrlConfigResponseTypeDef,
    GetLayerVersionByArnRequestRequestTypeDef,
    GetLayerVersionPolicyRequestRequestTypeDef,
    GetLayerVersionPolicyResponseTypeDef,
    GetLayerVersionRequestRequestTypeDef,
    GetLayerVersionResponseTypeDef,
    GetPolicyRequestRequestTypeDef,
    GetPolicyResponseTypeDef,
    GetProvisionedConcurrencyConfigRequestRequestTypeDef,
    GetProvisionedConcurrencyConfigResponseTypeDef,
    GetRuntimeManagementConfigRequestRequestTypeDef,
    GetRuntimeManagementConfigResponseTypeDef,
    InvocationRequestRequestTypeDef,
    InvocationResponseTypeDef,
    InvokeAsyncRequestRequestTypeDef,
    InvokeAsyncResponseTypeDef,
    InvokeWithResponseStreamRequestRequestTypeDef,
    InvokeWithResponseStreamResponseTypeDef,
    ListAliasesRequestRequestTypeDef,
    ListAliasesResponseTypeDef,
    ListCodeSigningConfigsRequestRequestTypeDef,
    ListCodeSigningConfigsResponseTypeDef,
    ListEventSourceMappingsRequestRequestTypeDef,
    ListEventSourceMappingsResponseTypeDef,
    ListFunctionEventInvokeConfigsRequestRequestTypeDef,
    ListFunctionEventInvokeConfigsResponseTypeDef,
    ListFunctionsByCodeSigningConfigRequestRequestTypeDef,
    ListFunctionsByCodeSigningConfigResponseTypeDef,
    ListFunctionsRequestRequestTypeDef,
    ListFunctionsResponseTypeDef,
    ListFunctionUrlConfigsRequestRequestTypeDef,
    ListFunctionUrlConfigsResponseTypeDef,
    ListLayersRequestRequestTypeDef,
    ListLayersResponseTypeDef,
    ListLayerVersionsRequestRequestTypeDef,
    ListLayerVersionsResponseTypeDef,
    ListProvisionedConcurrencyConfigsRequestRequestTypeDef,
    ListProvisionedConcurrencyConfigsResponseTypeDef,
    ListTagsRequestRequestTypeDef,
    ListTagsResponseTypeDef,
    ListVersionsByFunctionRequestRequestTypeDef,
    ListVersionsByFunctionResponseTypeDef,
    PublishLayerVersionRequestRequestTypeDef,
    PublishLayerVersionResponseTypeDef,
    PublishVersionRequestRequestTypeDef,
    PutFunctionCodeSigningConfigRequestRequestTypeDef,
    PutFunctionCodeSigningConfigResponseTypeDef,
    PutFunctionConcurrencyRequestRequestTypeDef,
    PutFunctionEventInvokeConfigRequestRequestTypeDef,
    PutFunctionRecursionConfigRequestRequestTypeDef,
    PutFunctionRecursionConfigResponseTypeDef,
    PutProvisionedConcurrencyConfigRequestRequestTypeDef,
    PutProvisionedConcurrencyConfigResponseTypeDef,
    PutRuntimeManagementConfigRequestRequestTypeDef,
    PutRuntimeManagementConfigResponseTypeDef,
    RemoveLayerVersionPermissionRequestRequestTypeDef,
    RemovePermissionRequestRequestTypeDef,
    TagResourceRequestRequestTypeDef,
    UntagResourceRequestRequestTypeDef,
    UpdateAliasRequestRequestTypeDef,
    UpdateCodeSigningConfigRequestRequestTypeDef,
    UpdateCodeSigningConfigResponseTypeDef,
    UpdateEventSourceMappingRequestRequestTypeDef,
    UpdateFunctionCodeRequestRequestTypeDef,
    UpdateFunctionConfigurationRequestRequestTypeDef,
    UpdateFunctionEventInvokeConfigRequestRequestTypeDef,
    UpdateFunctionUrlConfigRequestRequestTypeDef,
    UpdateFunctionUrlConfigResponseTypeDef,
)
from .waiter import (
    FunctionActiveV2Waiter,
    FunctionActiveWaiter,
    FunctionExistsWaiter,
    FunctionUpdatedV2Waiter,
    FunctionUpdatedWaiter,
    PublishedVersionActiveWaiter,
)

if sys.version_info >= (3, 9):
    from builtins import dict as Dict
    from builtins import type as Type
    from collections.abc import Mapping
else:
    from typing import Dict, Mapping, Type
if sys.version_info >= (3, 12):
    from typing import Literal, Unpack
else:
    from typing_extensions import Literal, Unpack

__all__ = ("LambdaClient",)

class Exceptions(BaseClientExceptions):
    ClientError: Type[BotocoreClientError]
    CodeSigningConfigNotFoundException: Type[BotocoreClientError]
    CodeStorageExceededException: Type[BotocoreClientError]
    CodeVerificationFailedException: Type[BotocoreClientError]
    EC2AccessDeniedException: Type[BotocoreClientError]
    EC2ThrottledException: Type[BotocoreClientError]
    EC2UnexpectedException: Type[BotocoreClientError]
    EFSIOException: Type[BotocoreClientError]
    EFSMountConnectivityException: Type[BotocoreClientError]
    EFSMountFailureException: Type[BotocoreClientError]
    EFSMountTimeoutException: Type[BotocoreClientError]
    ENILimitReachedException: Type[BotocoreClientError]
    InvalidCodeSignatureException: Type[BotocoreClientError]
    InvalidParameterValueException: Type[BotocoreClientError]
    InvalidRequestContentException: Type[BotocoreClientError]
    InvalidRuntimeException: Type[BotocoreClientError]
    InvalidSecurityGroupIDException: Type[BotocoreClientError]
    InvalidSubnetIDException: Type[BotocoreClientError]
    InvalidZipFileException: Type[BotocoreClientError]
    KMSAccessDeniedException: Type[BotocoreClientError]
    KMSDisabledException: Type[BotocoreClientError]
    KMSInvalidStateException: Type[BotocoreClientError]
    KMSNotFoundException: Type[BotocoreClientError]
    PolicyLengthExceededException: Type[BotocoreClientError]
    PreconditionFailedException: Type[BotocoreClientError]
    ProvisionedConcurrencyConfigNotFoundException: Type[BotocoreClientError]
    RecursiveInvocationException: Type[BotocoreClientError]
    RequestTooLargeException: Type[BotocoreClientError]
    ResourceConflictException: Type[BotocoreClientError]
    ResourceInUseException: Type[BotocoreClientError]
    ResourceNotFoundException: Type[BotocoreClientError]
    ResourceNotReadyException: Type[BotocoreClientError]
    ServiceException: Type[BotocoreClientError]
    SnapStartException: Type[BotocoreClientError]
    SnapStartNotReadyException: Type[BotocoreClientError]
    SnapStartTimeoutException: Type[BotocoreClientError]
    SubnetIPAddressLimitReachedException: Type[BotocoreClientError]
    TooManyRequestsException: Type[BotocoreClientError]
    UnsupportedMediaTypeException: Type[BotocoreClientError]

class LambdaClient(BaseClient):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda.html#Lambda.Client)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/)
    """

    meta: ClientMeta

    @property
    def exceptions(self) -> Exceptions:
        """
        LambdaClient exceptions.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda.html#Lambda.Client)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#exceptions)
        """

    def can_paginate(self, operation_name: str) -> bool:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/can_paginate.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#can_paginate)
        """

    def generate_presigned_url(
        self,
        ClientMethod: str,
        Params: Mapping[str, Any] = ...,
        ExpiresIn: int = 3600,
        HttpMethod: str = ...,
    ) -> str:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/generate_presigned_url.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#generate_presigned_url)
        """

    def add_layer_version_permission(
        self, **kwargs: Unpack[AddLayerVersionPermissionRequestRequestTypeDef]
    ) -> AddLayerVersionPermissionResponseTypeDef:
        """
        Adds permissions to the resource-based policy of a version of an <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html">Lambda
        layer</a>.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/add_layer_version_permission.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#add_layer_version_permission)
        """

    def add_permission(
        self, **kwargs: Unpack[AddPermissionRequestRequestTypeDef]
    ) -> AddPermissionResponseTypeDef:
        """
        Grants a <a
        href="https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_principal.html#Principal_specifying">principal</a>
        permission to use a function.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/add_permission.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#add_permission)
        """

    def create_alias(
        self, **kwargs: Unpack[CreateAliasRequestRequestTypeDef]
    ) -> AliasConfigurationResponseTypeDef:
        """
        Creates an <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/configuration-aliases.html">alias</a>
        for a Lambda function version.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/create_alias.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#create_alias)
        """

    def create_code_signing_config(
        self, **kwargs: Unpack[CreateCodeSigningConfigRequestRequestTypeDef]
    ) -> CreateCodeSigningConfigResponseTypeDef:
        """
        Creates a code signing configuration.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/create_code_signing_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#create_code_signing_config)
        """

    def create_event_source_mapping(
        self, **kwargs: Unpack[CreateEventSourceMappingRequestRequestTypeDef]
    ) -> EventSourceMappingConfigurationResponseTypeDef:
        """
        Creates a mapping between an event source and an Lambda function.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/create_event_source_mapping.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#create_event_source_mapping)
        """

    def create_function(
        self, **kwargs: Unpack[CreateFunctionRequestRequestTypeDef]
    ) -> FunctionConfigurationResponseTypeDef:
        """
        Creates a Lambda function.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/create_function.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#create_function)
        """

    def create_function_url_config(
        self, **kwargs: Unpack[CreateFunctionUrlConfigRequestRequestTypeDef]
    ) -> CreateFunctionUrlConfigResponseTypeDef:
        """
        Creates a Lambda function URL with the specified configuration parameters.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/create_function_url_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#create_function_url_config)
        """

    def delete_alias(
        self, **kwargs: Unpack[DeleteAliasRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Deletes a Lambda function <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/configuration-aliases.html">alias</a>.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/delete_alias.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#delete_alias)
        """

    def delete_code_signing_config(
        self, **kwargs: Unpack[DeleteCodeSigningConfigRequestRequestTypeDef]
    ) -> Dict[str, Any]:
        """
        Deletes the code signing configuration.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/delete_code_signing_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#delete_code_signing_config)
        """

    def delete_event_source_mapping(
        self, **kwargs: Unpack[DeleteEventSourceMappingRequestRequestTypeDef]
    ) -> EventSourceMappingConfigurationResponseTypeDef:
        """
        Deletes an <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/intro-invocation-modes.html">event
        source mapping</a>.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/delete_event_source_mapping.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#delete_event_source_mapping)
        """

    def delete_function(
        self, **kwargs: Unpack[DeleteFunctionRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Deletes a Lambda function.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/delete_function.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#delete_function)
        """

    def delete_function_code_signing_config(
        self, **kwargs: Unpack[DeleteFunctionCodeSigningConfigRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Removes the code signing configuration from the function.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/delete_function_code_signing_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#delete_function_code_signing_config)
        """

    def delete_function_concurrency(
        self, **kwargs: Unpack[DeleteFunctionConcurrencyRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Removes a concurrent execution limit from a function.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/delete_function_concurrency.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#delete_function_concurrency)
        """

    def delete_function_event_invoke_config(
        self, **kwargs: Unpack[DeleteFunctionEventInvokeConfigRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Deletes the configuration for asynchronous invocation for a function, version,
        or alias.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/delete_function_event_invoke_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#delete_function_event_invoke_config)
        """

    def delete_function_url_config(
        self, **kwargs: Unpack[DeleteFunctionUrlConfigRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Deletes a Lambda function URL.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/delete_function_url_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#delete_function_url_config)
        """

    def delete_layer_version(
        self, **kwargs: Unpack[DeleteLayerVersionRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Deletes a version of an <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html">Lambda
        layer</a>.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/delete_layer_version.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#delete_layer_version)
        """

    def delete_provisioned_concurrency_config(
        self, **kwargs: Unpack[DeleteProvisionedConcurrencyConfigRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Deletes the provisioned concurrency configuration for a function.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/delete_provisioned_concurrency_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#delete_provisioned_concurrency_config)
        """

    def get_account_settings(self) -> GetAccountSettingsResponseTypeDef:
        """
        Retrieves details about your account's <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/limits.html">limits</a> and
        usage in an Amazon Web Services Region.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_account_settings.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_account_settings)
        """

    def get_alias(
        self, **kwargs: Unpack[GetAliasRequestRequestTypeDef]
    ) -> AliasConfigurationResponseTypeDef:
        """
        Returns details about a Lambda function <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/configuration-aliases.html">alias</a>.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_alias.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_alias)
        """

    def get_code_signing_config(
        self, **kwargs: Unpack[GetCodeSigningConfigRequestRequestTypeDef]
    ) -> GetCodeSigningConfigResponseTypeDef:
        """
        Returns information about the specified code signing configuration.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_code_signing_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_code_signing_config)
        """

    def get_event_source_mapping(
        self, **kwargs: Unpack[GetEventSourceMappingRequestRequestTypeDef]
    ) -> EventSourceMappingConfigurationResponseTypeDef:
        """
        Returns details about an event source mapping.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_event_source_mapping.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_event_source_mapping)
        """

    def get_function(
        self, **kwargs: Unpack[GetFunctionRequestRequestTypeDef]
    ) -> GetFunctionResponseTypeDef:
        """
        Returns information about the function or function version, with a link to
        download the deployment package that's valid for 10 minutes.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_function.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_function)
        """

    def get_function_code_signing_config(
        self, **kwargs: Unpack[GetFunctionCodeSigningConfigRequestRequestTypeDef]
    ) -> GetFunctionCodeSigningConfigResponseTypeDef:
        """
        Returns the code signing configuration for the specified function.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_function_code_signing_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_function_code_signing_config)
        """

    def get_function_concurrency(
        self, **kwargs: Unpack[GetFunctionConcurrencyRequestRequestTypeDef]
    ) -> GetFunctionConcurrencyResponseTypeDef:
        """
        Returns details about the reserved concurrency configuration for a function.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_function_concurrency.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_function_concurrency)
        """

    def get_function_configuration(
        self, **kwargs: Unpack[GetFunctionConfigurationRequestRequestTypeDef]
    ) -> FunctionConfigurationResponseTypeDef:
        """
        Returns the version-specific settings of a Lambda function or version.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_function_configuration.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_function_configuration)
        """

    def get_function_event_invoke_config(
        self, **kwargs: Unpack[GetFunctionEventInvokeConfigRequestRequestTypeDef]
    ) -> FunctionEventInvokeConfigResponseTypeDef:
        """
        Retrieves the configuration for asynchronous invocation for a function,
        version, or alias.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_function_event_invoke_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_function_event_invoke_config)
        """

    def get_function_recursion_config(
        self, **kwargs: Unpack[GetFunctionRecursionConfigRequestRequestTypeDef]
    ) -> GetFunctionRecursionConfigResponseTypeDef:
        """
        Returns your function's <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/invocation-recursion.html">recursive
        loop detection</a> configuration.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_function_recursion_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_function_recursion_config)
        """

    def get_function_url_config(
        self, **kwargs: Unpack[GetFunctionUrlConfigRequestRequestTypeDef]
    ) -> GetFunctionUrlConfigResponseTypeDef:
        """
        Returns details about a Lambda function URL.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_function_url_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_function_url_config)
        """

    def get_layer_version(
        self, **kwargs: Unpack[GetLayerVersionRequestRequestTypeDef]
    ) -> GetLayerVersionResponseTypeDef:
        """
        Returns information about a version of an <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html">Lambda
        layer</a>, with a link to download the layer archive that's valid for 10
        minutes.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_layer_version.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_layer_version)
        """

    def get_layer_version_by_arn(
        self, **kwargs: Unpack[GetLayerVersionByArnRequestRequestTypeDef]
    ) -> GetLayerVersionResponseTypeDef:
        """
        Returns information about a version of an <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html">Lambda
        layer</a>, with a link to download the layer archive that's valid for 10
        minutes.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_layer_version_by_arn.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_layer_version_by_arn)
        """

    def get_layer_version_policy(
        self, **kwargs: Unpack[GetLayerVersionPolicyRequestRequestTypeDef]
    ) -> GetLayerVersionPolicyResponseTypeDef:
        """
        Returns the permission policy for a version of an <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html">Lambda
        layer</a>.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_layer_version_policy.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_layer_version_policy)
        """

    def get_policy(
        self, **kwargs: Unpack[GetPolicyRequestRequestTypeDef]
    ) -> GetPolicyResponseTypeDef:
        """
        Returns the <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/access-control-resource-based.html">resource-based
        IAM policy</a> for a function, version, or alias.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_policy.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_policy)
        """

    def get_provisioned_concurrency_config(
        self, **kwargs: Unpack[GetProvisionedConcurrencyConfigRequestRequestTypeDef]
    ) -> GetProvisionedConcurrencyConfigResponseTypeDef:
        """
        Retrieves the provisioned concurrency configuration for a function's alias or
        version.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_provisioned_concurrency_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_provisioned_concurrency_config)
        """

    def get_runtime_management_config(
        self, **kwargs: Unpack[GetRuntimeManagementConfigRequestRequestTypeDef]
    ) -> GetRuntimeManagementConfigResponseTypeDef:
        """
        Retrieves the runtime management configuration for a function's version.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_runtime_management_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_runtime_management_config)
        """

    def invoke(
        self, **kwargs: Unpack[InvocationRequestRequestTypeDef]
    ) -> InvocationResponseTypeDef:
        """
        Invokes a Lambda function.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/invoke.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#invoke)
        """

    def invoke_async(
        self, **kwargs: Unpack[InvokeAsyncRequestRequestTypeDef]
    ) -> InvokeAsyncResponseTypeDef:
        """
        For asynchronous function invocation, use <a>Invoke</a>.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/invoke_async.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#invoke_async)
        """

    def invoke_with_response_stream(
        self, **kwargs: Unpack[InvokeWithResponseStreamRequestRequestTypeDef]
    ) -> InvokeWithResponseStreamResponseTypeDef:
        """
        Configure your Lambda functions to stream response payloads back to clients.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/invoke_with_response_stream.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#invoke_with_response_stream)
        """

    def list_aliases(
        self, **kwargs: Unpack[ListAliasesRequestRequestTypeDef]
    ) -> ListAliasesResponseTypeDef:
        """
        Returns a list of <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/configuration-aliases.html">aliases</a>
        for a Lambda function.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/list_aliases.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#list_aliases)
        """

    def list_code_signing_configs(
        self, **kwargs: Unpack[ListCodeSigningConfigsRequestRequestTypeDef]
    ) -> ListCodeSigningConfigsResponseTypeDef:
        """
        Returns a list of <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/configuring-codesigning.html">code
        signing configurations</a>.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/list_code_signing_configs.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#list_code_signing_configs)
        """

    def list_event_source_mappings(
        self, **kwargs: Unpack[ListEventSourceMappingsRequestRequestTypeDef]
    ) -> ListEventSourceMappingsResponseTypeDef:
        """
        Lists event source mappings.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/list_event_source_mappings.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#list_event_source_mappings)
        """

    def list_function_event_invoke_configs(
        self, **kwargs: Unpack[ListFunctionEventInvokeConfigsRequestRequestTypeDef]
    ) -> ListFunctionEventInvokeConfigsResponseTypeDef:
        """
        Retrieves a list of configurations for asynchronous invocation for a function.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/list_function_event_invoke_configs.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#list_function_event_invoke_configs)
        """

    def list_function_url_configs(
        self, **kwargs: Unpack[ListFunctionUrlConfigsRequestRequestTypeDef]
    ) -> ListFunctionUrlConfigsResponseTypeDef:
        """
        Returns a list of Lambda function URLs for the specified function.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/list_function_url_configs.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#list_function_url_configs)
        """

    def list_functions(
        self, **kwargs: Unpack[ListFunctionsRequestRequestTypeDef]
    ) -> ListFunctionsResponseTypeDef:
        """
        Returns a list of Lambda functions, with the version-specific configuration of
        each.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/list_functions.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#list_functions)
        """

    def list_functions_by_code_signing_config(
        self, **kwargs: Unpack[ListFunctionsByCodeSigningConfigRequestRequestTypeDef]
    ) -> ListFunctionsByCodeSigningConfigResponseTypeDef:
        """
        List the functions that use the specified code signing configuration.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/list_functions_by_code_signing_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#list_functions_by_code_signing_config)
        """

    def list_layer_versions(
        self, **kwargs: Unpack[ListLayerVersionsRequestRequestTypeDef]
    ) -> ListLayerVersionsResponseTypeDef:
        """
        Lists the versions of an <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html">Lambda
        layer</a>.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/list_layer_versions.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#list_layer_versions)
        """

    def list_layers(
        self, **kwargs: Unpack[ListLayersRequestRequestTypeDef]
    ) -> ListLayersResponseTypeDef:
        """
        Lists <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/invocation-layers.html">Lambda
        layers</a> and shows information about the latest version of each.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/list_layers.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#list_layers)
        """

    def list_provisioned_concurrency_configs(
        self, **kwargs: Unpack[ListProvisionedConcurrencyConfigsRequestRequestTypeDef]
    ) -> ListProvisionedConcurrencyConfigsResponseTypeDef:
        """
        Retrieves a list of provisioned concurrency configurations for a function.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/list_provisioned_concurrency_configs.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#list_provisioned_concurrency_configs)
        """

    def list_tags(self, **kwargs: Unpack[ListTagsRequestRequestTypeDef]) -> ListTagsResponseTypeDef:
        """
        Returns a function, event source mapping, or code signing configuration's <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/tagging.html">tags</a>.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/list_tags.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#list_tags)
        """

    def list_versions_by_function(
        self, **kwargs: Unpack[ListVersionsByFunctionRequestRequestTypeDef]
    ) -> ListVersionsByFunctionResponseTypeDef:
        """
        Returns a list of <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/versioning-aliases.html">versions</a>,
        with the version-specific configuration of each.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/list_versions_by_function.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#list_versions_by_function)
        """

    def publish_layer_version(
        self, **kwargs: Unpack[PublishLayerVersionRequestRequestTypeDef]
    ) -> PublishLayerVersionResponseTypeDef:
        """
        Creates an <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html">Lambda
        layer</a> from a ZIP archive.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/publish_layer_version.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#publish_layer_version)
        """

    def publish_version(
        self, **kwargs: Unpack[PublishVersionRequestRequestTypeDef]
    ) -> FunctionConfigurationResponseTypeDef:
        """
        Creates a <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/versioning-aliases.html">version</a>
        from the current code and configuration of a function.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/publish_version.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#publish_version)
        """

    def put_function_code_signing_config(
        self, **kwargs: Unpack[PutFunctionCodeSigningConfigRequestRequestTypeDef]
    ) -> PutFunctionCodeSigningConfigResponseTypeDef:
        """
        Update the code signing configuration for the function.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/put_function_code_signing_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#put_function_code_signing_config)
        """

    def put_function_concurrency(
        self, **kwargs: Unpack[PutFunctionConcurrencyRequestRequestTypeDef]
    ) -> ConcurrencyResponseTypeDef:
        """
        Sets the maximum number of simultaneous executions for a function, and reserves
        capacity for that concurrency level.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/put_function_concurrency.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#put_function_concurrency)
        """

    def put_function_event_invoke_config(
        self, **kwargs: Unpack[PutFunctionEventInvokeConfigRequestRequestTypeDef]
    ) -> FunctionEventInvokeConfigResponseTypeDef:
        """
        Configures options for <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/invocation-async.html">asynchronous
        invocation</a> on a function, version, or alias.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/put_function_event_invoke_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#put_function_event_invoke_config)
        """

    def put_function_recursion_config(
        self, **kwargs: Unpack[PutFunctionRecursionConfigRequestRequestTypeDef]
    ) -> PutFunctionRecursionConfigResponseTypeDef:
        """
        Sets your function's <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/invocation-recursion.html">recursive
        loop detection</a> configuration.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/put_function_recursion_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#put_function_recursion_config)
        """

    def put_provisioned_concurrency_config(
        self, **kwargs: Unpack[PutProvisionedConcurrencyConfigRequestRequestTypeDef]
    ) -> PutProvisionedConcurrencyConfigResponseTypeDef:
        """
        Adds a provisioned concurrency configuration to a function's alias or version.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/put_provisioned_concurrency_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#put_provisioned_concurrency_config)
        """

    def put_runtime_management_config(
        self, **kwargs: Unpack[PutRuntimeManagementConfigRequestRequestTypeDef]
    ) -> PutRuntimeManagementConfigResponseTypeDef:
        """
        Sets the runtime management configuration for a function's version.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/put_runtime_management_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#put_runtime_management_config)
        """

    def remove_layer_version_permission(
        self, **kwargs: Unpack[RemoveLayerVersionPermissionRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Removes a statement from the permissions policy for a version of an <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html">Lambda
        layer</a>.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/remove_layer_version_permission.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#remove_layer_version_permission)
        """

    def remove_permission(
        self, **kwargs: Unpack[RemovePermissionRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Revokes function-use permission from an Amazon Web Services service or another
        Amazon Web Services account.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/remove_permission.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#remove_permission)
        """

    def tag_resource(
        self, **kwargs: Unpack[TagResourceRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Adds <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/tagging.html">tags</a> to a
        function, event source mapping, or code signing configuration.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/tag_resource.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#tag_resource)
        """

    def untag_resource(
        self, **kwargs: Unpack[UntagResourceRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Removes <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/tagging.html">tags</a> from
        a function, event source mapping, or code signing configuration.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/untag_resource.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#untag_resource)
        """

    def update_alias(
        self, **kwargs: Unpack[UpdateAliasRequestRequestTypeDef]
    ) -> AliasConfigurationResponseTypeDef:
        """
        Updates the configuration of a Lambda function <a
        href="https://docs.aws.amazon.com/lambda/latest/dg/configuration-aliases.html">alias</a>.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/update_alias.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#update_alias)
        """

    def update_code_signing_config(
        self, **kwargs: Unpack[UpdateCodeSigningConfigRequestRequestTypeDef]
    ) -> UpdateCodeSigningConfigResponseTypeDef:
        """
        Update the code signing configuration.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/update_code_signing_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#update_code_signing_config)
        """

    def update_event_source_mapping(
        self, **kwargs: Unpack[UpdateEventSourceMappingRequestRequestTypeDef]
    ) -> EventSourceMappingConfigurationResponseTypeDef:
        """
        Updates an event source mapping.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/update_event_source_mapping.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#update_event_source_mapping)
        """

    def update_function_code(
        self, **kwargs: Unpack[UpdateFunctionCodeRequestRequestTypeDef]
    ) -> FunctionConfigurationResponseTypeDef:
        """
        Updates a Lambda function's code.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/update_function_code.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#update_function_code)
        """

    def update_function_configuration(
        self, **kwargs: Unpack[UpdateFunctionConfigurationRequestRequestTypeDef]
    ) -> FunctionConfigurationResponseTypeDef:
        """
        Modify the version-specific settings of a Lambda function.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/update_function_configuration.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#update_function_configuration)
        """

    def update_function_event_invoke_config(
        self, **kwargs: Unpack[UpdateFunctionEventInvokeConfigRequestRequestTypeDef]
    ) -> FunctionEventInvokeConfigResponseTypeDef:
        """
        Updates the configuration for asynchronous invocation for a function, version,
        or alias.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/update_function_event_invoke_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#update_function_event_invoke_config)
        """

    def update_function_url_config(
        self, **kwargs: Unpack[UpdateFunctionUrlConfigRequestRequestTypeDef]
    ) -> UpdateFunctionUrlConfigResponseTypeDef:
        """
        Updates the configuration for a Lambda function URL.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/update_function_url_config.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#update_function_url_config)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["list_aliases"]
    ) -> ListAliasesPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["list_code_signing_configs"]
    ) -> ListCodeSigningConfigsPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["list_event_source_mappings"]
    ) -> ListEventSourceMappingsPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["list_function_event_invoke_configs"]
    ) -> ListFunctionEventInvokeConfigsPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["list_function_url_configs"]
    ) -> ListFunctionUrlConfigsPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["list_functions_by_code_signing_config"]
    ) -> ListFunctionsByCodeSigningConfigPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["list_functions"]
    ) -> ListFunctionsPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["list_layer_versions"]
    ) -> ListLayerVersionsPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["list_layers"]
    ) -> ListLayersPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["list_provisioned_concurrency_configs"]
    ) -> ListProvisionedConcurrencyConfigsPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["list_versions_by_function"]
    ) -> ListVersionsByFunctionPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_waiter(  # type: ignore[override]
        self, waiter_name: Literal["function_active_v2"]
    ) -> FunctionActiveV2Waiter:
        """
        Returns an object that can wait for some condition.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_waiter.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_waiter)
        """

    @overload  # type: ignore[override]
    def get_waiter(  # type: ignore[override]
        self, waiter_name: Literal["function_active"]
    ) -> FunctionActiveWaiter:
        """
        Returns an object that can wait for some condition.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_waiter.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_waiter)
        """

    @overload  # type: ignore[override]
    def get_waiter(  # type: ignore[override]
        self, waiter_name: Literal["function_exists"]
    ) -> FunctionExistsWaiter:
        """
        Returns an object that can wait for some condition.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_waiter.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_waiter)
        """

    @overload  # type: ignore[override]
    def get_waiter(  # type: ignore[override]
        self, waiter_name: Literal["function_updated_v2"]
    ) -> FunctionUpdatedV2Waiter:
        """
        Returns an object that can wait for some condition.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_waiter.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_waiter)
        """

    @overload  # type: ignore[override]
    def get_waiter(  # type: ignore[override]
        self, waiter_name: Literal["function_updated"]
    ) -> FunctionUpdatedWaiter:
        """
        Returns an object that can wait for some condition.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_waiter.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_waiter)
        """

    @overload  # type: ignore[override]
    def get_waiter(  # type: ignore[override]
        self, waiter_name: Literal["published_version_active"]
    ) -> PublishedVersionActiveWaiter:
        """
        Returns an object that can wait for some condition.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/get_waiter.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/client/#get_waiter)
        """
