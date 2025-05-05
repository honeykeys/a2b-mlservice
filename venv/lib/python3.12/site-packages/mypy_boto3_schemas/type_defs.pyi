"""
Type annotations for schemas service type definitions.

[Documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_schemas/type_defs/)

Usage::

    ```python
    from mypy_boto3_schemas.type_defs import CreateDiscovererRequestRequestTypeDef

    data: CreateDiscovererRequestRequestTypeDef = ...
    ```

Copyright 2025 Vlad Emelianov
"""

from __future__ import annotations

import sys
from datetime import datetime

from botocore.response import StreamingBody

from .literals import CodeGenerationStatusType, DiscovererStateType, TypeType

if sys.version_info >= (3, 9):
    from builtins import dict as Dict
    from builtins import list as List
    from collections.abc import Mapping, Sequence
else:
    from typing import Dict, List, Mapping, Sequence
if sys.version_info >= (3, 12):
    from typing import NotRequired, TypedDict
else:
    from typing_extensions import NotRequired, TypedDict

__all__ = (
    "CreateDiscovererRequestRequestTypeDef",
    "CreateDiscovererResponseTypeDef",
    "CreateRegistryRequestRequestTypeDef",
    "CreateRegistryResponseTypeDef",
    "CreateSchemaRequestRequestTypeDef",
    "CreateSchemaResponseTypeDef",
    "DeleteDiscovererRequestRequestTypeDef",
    "DeleteRegistryRequestRequestTypeDef",
    "DeleteResourcePolicyRequestRequestTypeDef",
    "DeleteSchemaRequestRequestTypeDef",
    "DeleteSchemaVersionRequestRequestTypeDef",
    "DescribeCodeBindingRequestRequestTypeDef",
    "DescribeCodeBindingRequestWaitTypeDef",
    "DescribeCodeBindingResponseTypeDef",
    "DescribeDiscovererRequestRequestTypeDef",
    "DescribeDiscovererResponseTypeDef",
    "DescribeRegistryRequestRequestTypeDef",
    "DescribeRegistryResponseTypeDef",
    "DescribeSchemaRequestRequestTypeDef",
    "DescribeSchemaResponseTypeDef",
    "DiscovererSummaryTypeDef",
    "EmptyResponseMetadataTypeDef",
    "ExportSchemaRequestRequestTypeDef",
    "ExportSchemaResponseTypeDef",
    "GetCodeBindingSourceRequestRequestTypeDef",
    "GetCodeBindingSourceResponseTypeDef",
    "GetDiscoveredSchemaRequestRequestTypeDef",
    "GetDiscoveredSchemaResponseTypeDef",
    "GetResourcePolicyRequestRequestTypeDef",
    "GetResourcePolicyResponseTypeDef",
    "ListDiscoverersRequestPaginateTypeDef",
    "ListDiscoverersRequestRequestTypeDef",
    "ListDiscoverersResponseTypeDef",
    "ListRegistriesRequestPaginateTypeDef",
    "ListRegistriesRequestRequestTypeDef",
    "ListRegistriesResponseTypeDef",
    "ListSchemaVersionsRequestPaginateTypeDef",
    "ListSchemaVersionsRequestRequestTypeDef",
    "ListSchemaVersionsResponseTypeDef",
    "ListSchemasRequestPaginateTypeDef",
    "ListSchemasRequestRequestTypeDef",
    "ListSchemasResponseTypeDef",
    "ListTagsForResourceRequestRequestTypeDef",
    "ListTagsForResourceResponseTypeDef",
    "PaginatorConfigTypeDef",
    "PutCodeBindingRequestRequestTypeDef",
    "PutCodeBindingResponseTypeDef",
    "PutResourcePolicyRequestRequestTypeDef",
    "PutResourcePolicyResponseTypeDef",
    "RegistrySummaryTypeDef",
    "ResponseMetadataTypeDef",
    "SchemaSummaryTypeDef",
    "SchemaVersionSummaryTypeDef",
    "SearchSchemaSummaryTypeDef",
    "SearchSchemaVersionSummaryTypeDef",
    "SearchSchemasRequestPaginateTypeDef",
    "SearchSchemasRequestRequestTypeDef",
    "SearchSchemasResponseTypeDef",
    "StartDiscovererRequestRequestTypeDef",
    "StartDiscovererResponseTypeDef",
    "StopDiscovererRequestRequestTypeDef",
    "StopDiscovererResponseTypeDef",
    "TagResourceRequestRequestTypeDef",
    "UntagResourceRequestRequestTypeDef",
    "UpdateDiscovererRequestRequestTypeDef",
    "UpdateDiscovererResponseTypeDef",
    "UpdateRegistryRequestRequestTypeDef",
    "UpdateRegistryResponseTypeDef",
    "UpdateSchemaRequestRequestTypeDef",
    "UpdateSchemaResponseTypeDef",
    "WaiterConfigTypeDef",
)

class CreateDiscovererRequestRequestTypeDef(TypedDict):
    SourceArn: str
    Description: NotRequired[str]
    CrossAccount: NotRequired[bool]
    Tags: NotRequired[Mapping[str, str]]

class ResponseMetadataTypeDef(TypedDict):
    RequestId: str
    HTTPStatusCode: int
    HTTPHeaders: Dict[str, str]
    RetryAttempts: int
    HostId: NotRequired[str]

class CreateRegistryRequestRequestTypeDef(TypedDict):
    RegistryName: str
    Description: NotRequired[str]
    Tags: NotRequired[Mapping[str, str]]

CreateSchemaRequestRequestTypeDef = TypedDict(
    "CreateSchemaRequestRequestTypeDef",
    {
        "Content": str,
        "RegistryName": str,
        "SchemaName": str,
        "Type": TypeType,
        "Description": NotRequired[str],
        "Tags": NotRequired[Mapping[str, str]],
    },
)

class DeleteDiscovererRequestRequestTypeDef(TypedDict):
    DiscovererId: str

class DeleteRegistryRequestRequestTypeDef(TypedDict):
    RegistryName: str

class DeleteResourcePolicyRequestRequestTypeDef(TypedDict):
    RegistryName: NotRequired[str]

class DeleteSchemaRequestRequestTypeDef(TypedDict):
    RegistryName: str
    SchemaName: str

class DeleteSchemaVersionRequestRequestTypeDef(TypedDict):
    RegistryName: str
    SchemaName: str
    SchemaVersion: str

class DescribeCodeBindingRequestRequestTypeDef(TypedDict):
    Language: str
    RegistryName: str
    SchemaName: str
    SchemaVersion: NotRequired[str]

class WaiterConfigTypeDef(TypedDict):
    Delay: NotRequired[int]
    MaxAttempts: NotRequired[int]

class DescribeDiscovererRequestRequestTypeDef(TypedDict):
    DiscovererId: str

class DescribeRegistryRequestRequestTypeDef(TypedDict):
    RegistryName: str

class DescribeSchemaRequestRequestTypeDef(TypedDict):
    RegistryName: str
    SchemaName: str
    SchemaVersion: NotRequired[str]

class DiscovererSummaryTypeDef(TypedDict):
    DiscovererArn: NotRequired[str]
    DiscovererId: NotRequired[str]
    SourceArn: NotRequired[str]
    State: NotRequired[DiscovererStateType]
    CrossAccount: NotRequired[bool]
    Tags: NotRequired[Dict[str, str]]

ExportSchemaRequestRequestTypeDef = TypedDict(
    "ExportSchemaRequestRequestTypeDef",
    {
        "RegistryName": str,
        "SchemaName": str,
        "Type": str,
        "SchemaVersion": NotRequired[str],
    },
)

class GetCodeBindingSourceRequestRequestTypeDef(TypedDict):
    Language: str
    RegistryName: str
    SchemaName: str
    SchemaVersion: NotRequired[str]

GetDiscoveredSchemaRequestRequestTypeDef = TypedDict(
    "GetDiscoveredSchemaRequestRequestTypeDef",
    {
        "Events": Sequence[str],
        "Type": TypeType,
    },
)

class GetResourcePolicyRequestRequestTypeDef(TypedDict):
    RegistryName: NotRequired[str]

class PaginatorConfigTypeDef(TypedDict):
    MaxItems: NotRequired[int]
    PageSize: NotRequired[int]
    StartingToken: NotRequired[str]

class ListDiscoverersRequestRequestTypeDef(TypedDict):
    DiscovererIdPrefix: NotRequired[str]
    Limit: NotRequired[int]
    NextToken: NotRequired[str]
    SourceArnPrefix: NotRequired[str]

class ListRegistriesRequestRequestTypeDef(TypedDict):
    Limit: NotRequired[int]
    NextToken: NotRequired[str]
    RegistryNamePrefix: NotRequired[str]
    Scope: NotRequired[str]

class RegistrySummaryTypeDef(TypedDict):
    RegistryArn: NotRequired[str]
    RegistryName: NotRequired[str]
    Tags: NotRequired[Dict[str, str]]

class ListSchemaVersionsRequestRequestTypeDef(TypedDict):
    RegistryName: str
    SchemaName: str
    Limit: NotRequired[int]
    NextToken: NotRequired[str]

SchemaVersionSummaryTypeDef = TypedDict(
    "SchemaVersionSummaryTypeDef",
    {
        "SchemaArn": NotRequired[str],
        "SchemaName": NotRequired[str],
        "SchemaVersion": NotRequired[str],
        "Type": NotRequired[TypeType],
    },
)

class ListSchemasRequestRequestTypeDef(TypedDict):
    RegistryName: str
    Limit: NotRequired[int]
    NextToken: NotRequired[str]
    SchemaNamePrefix: NotRequired[str]

class SchemaSummaryTypeDef(TypedDict):
    LastModified: NotRequired[datetime]
    SchemaArn: NotRequired[str]
    SchemaName: NotRequired[str]
    Tags: NotRequired[Dict[str, str]]
    VersionCount: NotRequired[int]

class ListTagsForResourceRequestRequestTypeDef(TypedDict):
    ResourceArn: str

class PutCodeBindingRequestRequestTypeDef(TypedDict):
    Language: str
    RegistryName: str
    SchemaName: str
    SchemaVersion: NotRequired[str]

class PutResourcePolicyRequestRequestTypeDef(TypedDict):
    Policy: str
    RegistryName: NotRequired[str]
    RevisionId: NotRequired[str]

SearchSchemaVersionSummaryTypeDef = TypedDict(
    "SearchSchemaVersionSummaryTypeDef",
    {
        "CreatedDate": NotRequired[datetime],
        "SchemaVersion": NotRequired[str],
        "Type": NotRequired[TypeType],
    },
)

class SearchSchemasRequestRequestTypeDef(TypedDict):
    Keywords: str
    RegistryName: str
    Limit: NotRequired[int]
    NextToken: NotRequired[str]

class StartDiscovererRequestRequestTypeDef(TypedDict):
    DiscovererId: str

class StopDiscovererRequestRequestTypeDef(TypedDict):
    DiscovererId: str

class TagResourceRequestRequestTypeDef(TypedDict):
    ResourceArn: str
    Tags: Mapping[str, str]

class UntagResourceRequestRequestTypeDef(TypedDict):
    ResourceArn: str
    TagKeys: Sequence[str]

class UpdateDiscovererRequestRequestTypeDef(TypedDict):
    DiscovererId: str
    Description: NotRequired[str]
    CrossAccount: NotRequired[bool]

class UpdateRegistryRequestRequestTypeDef(TypedDict):
    RegistryName: str
    Description: NotRequired[str]

UpdateSchemaRequestRequestTypeDef = TypedDict(
    "UpdateSchemaRequestRequestTypeDef",
    {
        "RegistryName": str,
        "SchemaName": str,
        "ClientTokenId": NotRequired[str],
        "Content": NotRequired[str],
        "Description": NotRequired[str],
        "Type": NotRequired[TypeType],
    },
)

class CreateDiscovererResponseTypeDef(TypedDict):
    Description: str
    DiscovererArn: str
    DiscovererId: str
    SourceArn: str
    State: DiscovererStateType
    CrossAccount: bool
    Tags: Dict[str, str]
    ResponseMetadata: ResponseMetadataTypeDef

class CreateRegistryResponseTypeDef(TypedDict):
    Description: str
    RegistryArn: str
    RegistryName: str
    Tags: Dict[str, str]
    ResponseMetadata: ResponseMetadataTypeDef

CreateSchemaResponseTypeDef = TypedDict(
    "CreateSchemaResponseTypeDef",
    {
        "Description": str,
        "LastModified": datetime,
        "SchemaArn": str,
        "SchemaName": str,
        "SchemaVersion": str,
        "Tags": Dict[str, str],
        "Type": str,
        "VersionCreatedDate": datetime,
        "ResponseMetadata": ResponseMetadataTypeDef,
    },
)

class DescribeCodeBindingResponseTypeDef(TypedDict):
    CreationDate: datetime
    LastModified: datetime
    SchemaVersion: str
    Status: CodeGenerationStatusType
    ResponseMetadata: ResponseMetadataTypeDef

class DescribeDiscovererResponseTypeDef(TypedDict):
    Description: str
    DiscovererArn: str
    DiscovererId: str
    SourceArn: str
    State: DiscovererStateType
    CrossAccount: bool
    Tags: Dict[str, str]
    ResponseMetadata: ResponseMetadataTypeDef

class DescribeRegistryResponseTypeDef(TypedDict):
    Description: str
    RegistryArn: str
    RegistryName: str
    Tags: Dict[str, str]
    ResponseMetadata: ResponseMetadataTypeDef

DescribeSchemaResponseTypeDef = TypedDict(
    "DescribeSchemaResponseTypeDef",
    {
        "Content": str,
        "Description": str,
        "LastModified": datetime,
        "SchemaArn": str,
        "SchemaName": str,
        "SchemaVersion": str,
        "Tags": Dict[str, str],
        "Type": str,
        "VersionCreatedDate": datetime,
        "ResponseMetadata": ResponseMetadataTypeDef,
    },
)

class EmptyResponseMetadataTypeDef(TypedDict):
    ResponseMetadata: ResponseMetadataTypeDef

ExportSchemaResponseTypeDef = TypedDict(
    "ExportSchemaResponseTypeDef",
    {
        "Content": str,
        "SchemaArn": str,
        "SchemaName": str,
        "SchemaVersion": str,
        "Type": str,
        "ResponseMetadata": ResponseMetadataTypeDef,
    },
)

class GetCodeBindingSourceResponseTypeDef(TypedDict):
    Body: StreamingBody
    ResponseMetadata: ResponseMetadataTypeDef

class GetDiscoveredSchemaResponseTypeDef(TypedDict):
    Content: str
    ResponseMetadata: ResponseMetadataTypeDef

class GetResourcePolicyResponseTypeDef(TypedDict):
    Policy: str
    RevisionId: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListTagsForResourceResponseTypeDef(TypedDict):
    Tags: Dict[str, str]
    ResponseMetadata: ResponseMetadataTypeDef

class PutCodeBindingResponseTypeDef(TypedDict):
    CreationDate: datetime
    LastModified: datetime
    SchemaVersion: str
    Status: CodeGenerationStatusType
    ResponseMetadata: ResponseMetadataTypeDef

class PutResourcePolicyResponseTypeDef(TypedDict):
    Policy: str
    RevisionId: str
    ResponseMetadata: ResponseMetadataTypeDef

class StartDiscovererResponseTypeDef(TypedDict):
    DiscovererId: str
    State: DiscovererStateType
    ResponseMetadata: ResponseMetadataTypeDef

class StopDiscovererResponseTypeDef(TypedDict):
    DiscovererId: str
    State: DiscovererStateType
    ResponseMetadata: ResponseMetadataTypeDef

class UpdateDiscovererResponseTypeDef(TypedDict):
    Description: str
    DiscovererArn: str
    DiscovererId: str
    SourceArn: str
    State: DiscovererStateType
    CrossAccount: bool
    Tags: Dict[str, str]
    ResponseMetadata: ResponseMetadataTypeDef

class UpdateRegistryResponseTypeDef(TypedDict):
    Description: str
    RegistryArn: str
    RegistryName: str
    Tags: Dict[str, str]
    ResponseMetadata: ResponseMetadataTypeDef

UpdateSchemaResponseTypeDef = TypedDict(
    "UpdateSchemaResponseTypeDef",
    {
        "Description": str,
        "LastModified": datetime,
        "SchemaArn": str,
        "SchemaName": str,
        "SchemaVersion": str,
        "Tags": Dict[str, str],
        "Type": str,
        "VersionCreatedDate": datetime,
        "ResponseMetadata": ResponseMetadataTypeDef,
    },
)

class DescribeCodeBindingRequestWaitTypeDef(TypedDict):
    Language: str
    RegistryName: str
    SchemaName: str
    SchemaVersion: NotRequired[str]
    WaiterConfig: NotRequired[WaiterConfigTypeDef]

class ListDiscoverersResponseTypeDef(TypedDict):
    Discoverers: List[DiscovererSummaryTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef
    NextToken: NotRequired[str]

class ListDiscoverersRequestPaginateTypeDef(TypedDict):
    DiscovererIdPrefix: NotRequired[str]
    SourceArnPrefix: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListRegistriesRequestPaginateTypeDef(TypedDict):
    RegistryNamePrefix: NotRequired[str]
    Scope: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListSchemaVersionsRequestPaginateTypeDef(TypedDict):
    RegistryName: str
    SchemaName: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListSchemasRequestPaginateTypeDef(TypedDict):
    RegistryName: str
    SchemaNamePrefix: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class SearchSchemasRequestPaginateTypeDef(TypedDict):
    Keywords: str
    RegistryName: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListRegistriesResponseTypeDef(TypedDict):
    Registries: List[RegistrySummaryTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef
    NextToken: NotRequired[str]

class ListSchemaVersionsResponseTypeDef(TypedDict):
    SchemaVersions: List[SchemaVersionSummaryTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef
    NextToken: NotRequired[str]

class ListSchemasResponseTypeDef(TypedDict):
    Schemas: List[SchemaSummaryTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef
    NextToken: NotRequired[str]

class SearchSchemaSummaryTypeDef(TypedDict):
    RegistryName: NotRequired[str]
    SchemaArn: NotRequired[str]
    SchemaName: NotRequired[str]
    SchemaVersions: NotRequired[List[SearchSchemaVersionSummaryTypeDef]]

class SearchSchemasResponseTypeDef(TypedDict):
    Schemas: List[SearchSchemaSummaryTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef
    NextToken: NotRequired[str]
