"""
Type annotations for schemas service Client.

[Documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/)

Usage::

    ```python
    from boto3.session import Session
    from mypy_boto3_schemas.client import SchemasClient

    session = Session()
    client: SchemasClient = session.client("schemas")
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
    ListDiscoverersPaginator,
    ListRegistriesPaginator,
    ListSchemasPaginator,
    ListSchemaVersionsPaginator,
    SearchSchemasPaginator,
)
from .type_defs import (
    CreateDiscovererRequestRequestTypeDef,
    CreateDiscovererResponseTypeDef,
    CreateRegistryRequestRequestTypeDef,
    CreateRegistryResponseTypeDef,
    CreateSchemaRequestRequestTypeDef,
    CreateSchemaResponseTypeDef,
    DeleteDiscovererRequestRequestTypeDef,
    DeleteRegistryRequestRequestTypeDef,
    DeleteResourcePolicyRequestRequestTypeDef,
    DeleteSchemaRequestRequestTypeDef,
    DeleteSchemaVersionRequestRequestTypeDef,
    DescribeCodeBindingRequestRequestTypeDef,
    DescribeCodeBindingResponseTypeDef,
    DescribeDiscovererRequestRequestTypeDef,
    DescribeDiscovererResponseTypeDef,
    DescribeRegistryRequestRequestTypeDef,
    DescribeRegistryResponseTypeDef,
    DescribeSchemaRequestRequestTypeDef,
    DescribeSchemaResponseTypeDef,
    EmptyResponseMetadataTypeDef,
    ExportSchemaRequestRequestTypeDef,
    ExportSchemaResponseTypeDef,
    GetCodeBindingSourceRequestRequestTypeDef,
    GetCodeBindingSourceResponseTypeDef,
    GetDiscoveredSchemaRequestRequestTypeDef,
    GetDiscoveredSchemaResponseTypeDef,
    GetResourcePolicyRequestRequestTypeDef,
    GetResourcePolicyResponseTypeDef,
    ListDiscoverersRequestRequestTypeDef,
    ListDiscoverersResponseTypeDef,
    ListRegistriesRequestRequestTypeDef,
    ListRegistriesResponseTypeDef,
    ListSchemasRequestRequestTypeDef,
    ListSchemasResponseTypeDef,
    ListSchemaVersionsRequestRequestTypeDef,
    ListSchemaVersionsResponseTypeDef,
    ListTagsForResourceRequestRequestTypeDef,
    ListTagsForResourceResponseTypeDef,
    PutCodeBindingRequestRequestTypeDef,
    PutCodeBindingResponseTypeDef,
    PutResourcePolicyRequestRequestTypeDef,
    PutResourcePolicyResponseTypeDef,
    SearchSchemasRequestRequestTypeDef,
    SearchSchemasResponseTypeDef,
    StartDiscovererRequestRequestTypeDef,
    StartDiscovererResponseTypeDef,
    StopDiscovererRequestRequestTypeDef,
    StopDiscovererResponseTypeDef,
    TagResourceRequestRequestTypeDef,
    UntagResourceRequestRequestTypeDef,
    UpdateDiscovererRequestRequestTypeDef,
    UpdateDiscovererResponseTypeDef,
    UpdateRegistryRequestRequestTypeDef,
    UpdateRegistryResponseTypeDef,
    UpdateSchemaRequestRequestTypeDef,
    UpdateSchemaResponseTypeDef,
)
from .waiter import CodeBindingExistsWaiter

if sys.version_info >= (3, 9):
    from builtins import type as Type
    from collections.abc import Mapping
else:
    from typing import Mapping, Type
if sys.version_info >= (3, 12):
    from typing import Literal, Unpack
else:
    from typing_extensions import Literal, Unpack


__all__ = ("SchemasClient",)


class Exceptions(BaseClientExceptions):
    BadRequestException: Type[BotocoreClientError]
    ClientError: Type[BotocoreClientError]
    ConflictException: Type[BotocoreClientError]
    ForbiddenException: Type[BotocoreClientError]
    GoneException: Type[BotocoreClientError]
    InternalServerErrorException: Type[BotocoreClientError]
    NotFoundException: Type[BotocoreClientError]
    PreconditionFailedException: Type[BotocoreClientError]
    ServiceUnavailableException: Type[BotocoreClientError]
    TooManyRequestsException: Type[BotocoreClientError]
    UnauthorizedException: Type[BotocoreClientError]


class SchemasClient(BaseClient):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas.html#Schemas.Client)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/)
    """

    meta: ClientMeta

    @property
    def exceptions(self) -> Exceptions:
        """
        SchemasClient exceptions.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas.html#Schemas.Client)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#exceptions)
        """

    def can_paginate(self, operation_name: str) -> bool:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/can_paginate.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#can_paginate)
        """

    def generate_presigned_url(
        self,
        ClientMethod: str,
        Params: Mapping[str, Any] = ...,
        ExpiresIn: int = 3600,
        HttpMethod: str = ...,
    ) -> str:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/generate_presigned_url.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#generate_presigned_url)
        """

    def create_discoverer(
        self, **kwargs: Unpack[CreateDiscovererRequestRequestTypeDef]
    ) -> CreateDiscovererResponseTypeDef:
        """
        Creates a discoverer.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/create_discoverer.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#create_discoverer)
        """

    def create_registry(
        self, **kwargs: Unpack[CreateRegistryRequestRequestTypeDef]
    ) -> CreateRegistryResponseTypeDef:
        """
        Creates a registry.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/create_registry.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#create_registry)
        """

    def create_schema(
        self, **kwargs: Unpack[CreateSchemaRequestRequestTypeDef]
    ) -> CreateSchemaResponseTypeDef:
        """
        Creates a schema definition.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/create_schema.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#create_schema)
        """

    def delete_discoverer(
        self, **kwargs: Unpack[DeleteDiscovererRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Deletes a discoverer.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/delete_discoverer.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#delete_discoverer)
        """

    def delete_registry(
        self, **kwargs: Unpack[DeleteRegistryRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Deletes a Registry.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/delete_registry.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#delete_registry)
        """

    def delete_resource_policy(
        self, **kwargs: Unpack[DeleteResourcePolicyRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Delete the resource-based policy attached to the specified registry.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/delete_resource_policy.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#delete_resource_policy)
        """

    def delete_schema(
        self, **kwargs: Unpack[DeleteSchemaRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Delete a schema definition.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/delete_schema.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#delete_schema)
        """

    def delete_schema_version(
        self, **kwargs: Unpack[DeleteSchemaVersionRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Delete the schema version definition.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/delete_schema_version.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#delete_schema_version)
        """

    def describe_code_binding(
        self, **kwargs: Unpack[DescribeCodeBindingRequestRequestTypeDef]
    ) -> DescribeCodeBindingResponseTypeDef:
        """
        Describe the code binding URI.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/describe_code_binding.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#describe_code_binding)
        """

    def describe_discoverer(
        self, **kwargs: Unpack[DescribeDiscovererRequestRequestTypeDef]
    ) -> DescribeDiscovererResponseTypeDef:
        """
        Describes the discoverer.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/describe_discoverer.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#describe_discoverer)
        """

    def describe_registry(
        self, **kwargs: Unpack[DescribeRegistryRequestRequestTypeDef]
    ) -> DescribeRegistryResponseTypeDef:
        """
        Describes the registry.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/describe_registry.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#describe_registry)
        """

    def describe_schema(
        self, **kwargs: Unpack[DescribeSchemaRequestRequestTypeDef]
    ) -> DescribeSchemaResponseTypeDef:
        """
        Retrieve the schema definition.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/describe_schema.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#describe_schema)
        """

    def export_schema(
        self, **kwargs: Unpack[ExportSchemaRequestRequestTypeDef]
    ) -> ExportSchemaResponseTypeDef:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/export_schema.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#export_schema)
        """

    def get_code_binding_source(
        self, **kwargs: Unpack[GetCodeBindingSourceRequestRequestTypeDef]
    ) -> GetCodeBindingSourceResponseTypeDef:
        """
        Get the code binding source URI.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/get_code_binding_source.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#get_code_binding_source)
        """

    def get_discovered_schema(
        self, **kwargs: Unpack[GetDiscoveredSchemaRequestRequestTypeDef]
    ) -> GetDiscoveredSchemaResponseTypeDef:
        """
        Get the discovered schema that was generated based on sampled events.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/get_discovered_schema.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#get_discovered_schema)
        """

    def get_resource_policy(
        self, **kwargs: Unpack[GetResourcePolicyRequestRequestTypeDef]
    ) -> GetResourcePolicyResponseTypeDef:
        """
        Retrieves the resource-based policy attached to a given registry.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/get_resource_policy.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#get_resource_policy)
        """

    def list_discoverers(
        self, **kwargs: Unpack[ListDiscoverersRequestRequestTypeDef]
    ) -> ListDiscoverersResponseTypeDef:
        """
        List the discoverers.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/list_discoverers.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#list_discoverers)
        """

    def list_registries(
        self, **kwargs: Unpack[ListRegistriesRequestRequestTypeDef]
    ) -> ListRegistriesResponseTypeDef:
        """
        List the registries.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/list_registries.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#list_registries)
        """

    def list_schema_versions(
        self, **kwargs: Unpack[ListSchemaVersionsRequestRequestTypeDef]
    ) -> ListSchemaVersionsResponseTypeDef:
        """
        Provides a list of the schema versions and related information.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/list_schema_versions.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#list_schema_versions)
        """

    def list_schemas(
        self, **kwargs: Unpack[ListSchemasRequestRequestTypeDef]
    ) -> ListSchemasResponseTypeDef:
        """
        List the schemas.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/list_schemas.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#list_schemas)
        """

    def list_tags_for_resource(
        self, **kwargs: Unpack[ListTagsForResourceRequestRequestTypeDef]
    ) -> ListTagsForResourceResponseTypeDef:
        """
        Get tags for resource.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/list_tags_for_resource.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#list_tags_for_resource)
        """

    def put_code_binding(
        self, **kwargs: Unpack[PutCodeBindingRequestRequestTypeDef]
    ) -> PutCodeBindingResponseTypeDef:
        """
        Put code binding URI.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/put_code_binding.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#put_code_binding)
        """

    def put_resource_policy(
        self, **kwargs: Unpack[PutResourcePolicyRequestRequestTypeDef]
    ) -> PutResourcePolicyResponseTypeDef:
        """
        The name of the policy.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/put_resource_policy.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#put_resource_policy)
        """

    def search_schemas(
        self, **kwargs: Unpack[SearchSchemasRequestRequestTypeDef]
    ) -> SearchSchemasResponseTypeDef:
        """
        Search the schemas.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/search_schemas.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#search_schemas)
        """

    def start_discoverer(
        self, **kwargs: Unpack[StartDiscovererRequestRequestTypeDef]
    ) -> StartDiscovererResponseTypeDef:
        """
        Starts the discoverer.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/start_discoverer.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#start_discoverer)
        """

    def stop_discoverer(
        self, **kwargs: Unpack[StopDiscovererRequestRequestTypeDef]
    ) -> StopDiscovererResponseTypeDef:
        """
        Stops the discoverer.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/stop_discoverer.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#stop_discoverer)
        """

    def tag_resource(
        self, **kwargs: Unpack[TagResourceRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Add tags to a resource.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/tag_resource.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#tag_resource)
        """

    def untag_resource(
        self, **kwargs: Unpack[UntagResourceRequestRequestTypeDef]
    ) -> EmptyResponseMetadataTypeDef:
        """
        Removes tags from a resource.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/untag_resource.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#untag_resource)
        """

    def update_discoverer(
        self, **kwargs: Unpack[UpdateDiscovererRequestRequestTypeDef]
    ) -> UpdateDiscovererResponseTypeDef:
        """
        Updates the discoverer.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/update_discoverer.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#update_discoverer)
        """

    def update_registry(
        self, **kwargs: Unpack[UpdateRegistryRequestRequestTypeDef]
    ) -> UpdateRegistryResponseTypeDef:
        """
        Updates a registry.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/update_registry.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#update_registry)
        """

    def update_schema(
        self, **kwargs: Unpack[UpdateSchemaRequestRequestTypeDef]
    ) -> UpdateSchemaResponseTypeDef:
        """
        Updates the schema definition.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/update_schema.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#update_schema)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["list_discoverers"]
    ) -> ListDiscoverersPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["list_registries"]
    ) -> ListRegistriesPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["list_schema_versions"]
    ) -> ListSchemaVersionsPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["list_schemas"]
    ) -> ListSchemasPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["search_schemas"]
    ) -> SearchSchemasPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#get_paginator)
        """

    def get_waiter(  # type: ignore[override]
        self, waiter_name: Literal["code_binding_exists"]
    ) -> CodeBindingExistsWaiter:
        """
        Returns an object that can wait for some condition.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/schemas/client/get_waiter.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/client/#get_waiter)
        """
