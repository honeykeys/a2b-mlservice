"""
Type annotations for lambda service type definitions.

[Documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_lambda/type_defs/)

Usage::

    ```python
    from mypy_boto3_lambda.type_defs import AccountLimitTypeDef

    data: AccountLimitTypeDef = ...
    ```

Copyright 2025 Vlad Emelianov
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import IO, Any, Union

from botocore.eventstream import EventStream
from botocore.response import StreamingBody

from .literals import (
    ApplicationLogLevelType,
    ArchitectureType,
    CodeSigningPolicyType,
    EventSourcePositionType,
    FullDocumentType,
    FunctionUrlAuthTypeType,
    InvocationTypeType,
    InvokeModeType,
    LastUpdateStatusReasonCodeType,
    LastUpdateStatusType,
    LogFormatType,
    LogTypeType,
    PackageTypeType,
    ProvisionedConcurrencyStatusEnumType,
    RecursiveLoopType,
    ResponseStreamingInvocationTypeType,
    RuntimeType,
    SnapStartApplyOnType,
    SnapStartOptimizationStatusType,
    SourceAccessTypeType,
    StateReasonCodeType,
    StateType,
    SystemLogLevelType,
    TracingModeType,
    UpdateRuntimeOnType,
)

if sys.version_info >= (3, 9):
    from builtins import dict as Dict
    from builtins import list as List
    from collections.abc import Mapping, Sequence
else:
    from typing import Dict, List, Mapping, Sequence
if sys.version_info >= (3, 12):
    from typing import Literal, NotRequired, TypedDict
else:
    from typing_extensions import Literal, NotRequired, TypedDict


__all__ = (
    "AccountLimitTypeDef",
    "AccountUsageTypeDef",
    "AddLayerVersionPermissionRequestRequestTypeDef",
    "AddLayerVersionPermissionResponseTypeDef",
    "AddPermissionRequestRequestTypeDef",
    "AddPermissionResponseTypeDef",
    "AliasConfigurationResponseTypeDef",
    "AliasConfigurationTypeDef",
    "AliasRoutingConfigurationOutputTypeDef",
    "AliasRoutingConfigurationTypeDef",
    "AllowedPublishersOutputTypeDef",
    "AllowedPublishersTypeDef",
    "AmazonManagedKafkaEventSourceConfigTypeDef",
    "BlobTypeDef",
    "CodeSigningConfigTypeDef",
    "CodeSigningPoliciesTypeDef",
    "ConcurrencyResponseTypeDef",
    "ConcurrencyTypeDef",
    "CorsOutputTypeDef",
    "CorsTypeDef",
    "CreateAliasRequestRequestTypeDef",
    "CreateCodeSigningConfigRequestRequestTypeDef",
    "CreateCodeSigningConfigResponseTypeDef",
    "CreateEventSourceMappingRequestRequestTypeDef",
    "CreateFunctionRequestRequestTypeDef",
    "CreateFunctionUrlConfigRequestRequestTypeDef",
    "CreateFunctionUrlConfigResponseTypeDef",
    "DeadLetterConfigTypeDef",
    "DeleteAliasRequestRequestTypeDef",
    "DeleteCodeSigningConfigRequestRequestTypeDef",
    "DeleteEventSourceMappingRequestRequestTypeDef",
    "DeleteFunctionCodeSigningConfigRequestRequestTypeDef",
    "DeleteFunctionConcurrencyRequestRequestTypeDef",
    "DeleteFunctionEventInvokeConfigRequestRequestTypeDef",
    "DeleteFunctionRequestRequestTypeDef",
    "DeleteFunctionUrlConfigRequestRequestTypeDef",
    "DeleteLayerVersionRequestRequestTypeDef",
    "DeleteProvisionedConcurrencyConfigRequestRequestTypeDef",
    "DestinationConfigTypeDef",
    "DocumentDBEventSourceConfigTypeDef",
    "EmptyResponseMetadataTypeDef",
    "EnvironmentErrorTypeDef",
    "EnvironmentResponseTypeDef",
    "EnvironmentTypeDef",
    "EphemeralStorageTypeDef",
    "EventSourceMappingConfigurationResponseTypeDef",
    "EventSourceMappingConfigurationTypeDef",
    "EventSourceMappingMetricsConfigOutputTypeDef",
    "EventSourceMappingMetricsConfigTypeDef",
    "FileSystemConfigTypeDef",
    "FilterCriteriaErrorTypeDef",
    "FilterCriteriaOutputTypeDef",
    "FilterCriteriaTypeDef",
    "FilterTypeDef",
    "FunctionCodeLocationTypeDef",
    "FunctionCodeTypeDef",
    "FunctionConfigurationResponseTypeDef",
    "FunctionConfigurationTypeDef",
    "FunctionEventInvokeConfigResponseTypeDef",
    "FunctionEventInvokeConfigTypeDef",
    "FunctionUrlConfigTypeDef",
    "GetAccountSettingsResponseTypeDef",
    "GetAliasRequestRequestTypeDef",
    "GetCodeSigningConfigRequestRequestTypeDef",
    "GetCodeSigningConfigResponseTypeDef",
    "GetEventSourceMappingRequestRequestTypeDef",
    "GetFunctionCodeSigningConfigRequestRequestTypeDef",
    "GetFunctionCodeSigningConfigResponseTypeDef",
    "GetFunctionConcurrencyRequestRequestTypeDef",
    "GetFunctionConcurrencyResponseTypeDef",
    "GetFunctionConfigurationRequestRequestTypeDef",
    "GetFunctionConfigurationRequestWaitTypeDef",
    "GetFunctionEventInvokeConfigRequestRequestTypeDef",
    "GetFunctionRecursionConfigRequestRequestTypeDef",
    "GetFunctionRecursionConfigResponseTypeDef",
    "GetFunctionRequestRequestTypeDef",
    "GetFunctionRequestWaitTypeDef",
    "GetFunctionResponseTypeDef",
    "GetFunctionUrlConfigRequestRequestTypeDef",
    "GetFunctionUrlConfigResponseTypeDef",
    "GetLayerVersionByArnRequestRequestTypeDef",
    "GetLayerVersionPolicyRequestRequestTypeDef",
    "GetLayerVersionPolicyResponseTypeDef",
    "GetLayerVersionRequestRequestTypeDef",
    "GetLayerVersionResponseTypeDef",
    "GetPolicyRequestRequestTypeDef",
    "GetPolicyResponseTypeDef",
    "GetProvisionedConcurrencyConfigRequestRequestTypeDef",
    "GetProvisionedConcurrencyConfigResponseTypeDef",
    "GetRuntimeManagementConfigRequestRequestTypeDef",
    "GetRuntimeManagementConfigResponseTypeDef",
    "ImageConfigErrorTypeDef",
    "ImageConfigOutputTypeDef",
    "ImageConfigResponseTypeDef",
    "ImageConfigTypeDef",
    "InvocationRequestRequestTypeDef",
    "InvocationResponseTypeDef",
    "InvokeAsyncRequestRequestTypeDef",
    "InvokeAsyncResponseTypeDef",
    "InvokeResponseStreamUpdateTypeDef",
    "InvokeWithResponseStreamCompleteEventTypeDef",
    "InvokeWithResponseStreamRequestRequestTypeDef",
    "InvokeWithResponseStreamResponseEventTypeDef",
    "InvokeWithResponseStreamResponseTypeDef",
    "LayerTypeDef",
    "LayerVersionContentInputTypeDef",
    "LayerVersionContentOutputTypeDef",
    "LayerVersionsListItemTypeDef",
    "LayersListItemTypeDef",
    "ListAliasesRequestPaginateTypeDef",
    "ListAliasesRequestRequestTypeDef",
    "ListAliasesResponseTypeDef",
    "ListCodeSigningConfigsRequestPaginateTypeDef",
    "ListCodeSigningConfigsRequestRequestTypeDef",
    "ListCodeSigningConfigsResponseTypeDef",
    "ListEventSourceMappingsRequestPaginateTypeDef",
    "ListEventSourceMappingsRequestRequestTypeDef",
    "ListEventSourceMappingsResponseTypeDef",
    "ListFunctionEventInvokeConfigsRequestPaginateTypeDef",
    "ListFunctionEventInvokeConfigsRequestRequestTypeDef",
    "ListFunctionEventInvokeConfigsResponseTypeDef",
    "ListFunctionUrlConfigsRequestPaginateTypeDef",
    "ListFunctionUrlConfigsRequestRequestTypeDef",
    "ListFunctionUrlConfigsResponseTypeDef",
    "ListFunctionsByCodeSigningConfigRequestPaginateTypeDef",
    "ListFunctionsByCodeSigningConfigRequestRequestTypeDef",
    "ListFunctionsByCodeSigningConfigResponseTypeDef",
    "ListFunctionsRequestPaginateTypeDef",
    "ListFunctionsRequestRequestTypeDef",
    "ListFunctionsResponseTypeDef",
    "ListLayerVersionsRequestPaginateTypeDef",
    "ListLayerVersionsRequestRequestTypeDef",
    "ListLayerVersionsResponseTypeDef",
    "ListLayersRequestPaginateTypeDef",
    "ListLayersRequestRequestTypeDef",
    "ListLayersResponseTypeDef",
    "ListProvisionedConcurrencyConfigsRequestPaginateTypeDef",
    "ListProvisionedConcurrencyConfigsRequestRequestTypeDef",
    "ListProvisionedConcurrencyConfigsResponseTypeDef",
    "ListTagsRequestRequestTypeDef",
    "ListTagsResponseTypeDef",
    "ListVersionsByFunctionRequestPaginateTypeDef",
    "ListVersionsByFunctionRequestRequestTypeDef",
    "ListVersionsByFunctionResponseTypeDef",
    "LoggingConfigTypeDef",
    "OnFailureTypeDef",
    "OnSuccessTypeDef",
    "PaginatorConfigTypeDef",
    "ProvisionedConcurrencyConfigListItemTypeDef",
    "ProvisionedPollerConfigTypeDef",
    "PublishLayerVersionRequestRequestTypeDef",
    "PublishLayerVersionResponseTypeDef",
    "PublishVersionRequestRequestTypeDef",
    "PutFunctionCodeSigningConfigRequestRequestTypeDef",
    "PutFunctionCodeSigningConfigResponseTypeDef",
    "PutFunctionConcurrencyRequestRequestTypeDef",
    "PutFunctionEventInvokeConfigRequestRequestTypeDef",
    "PutFunctionRecursionConfigRequestRequestTypeDef",
    "PutFunctionRecursionConfigResponseTypeDef",
    "PutProvisionedConcurrencyConfigRequestRequestTypeDef",
    "PutProvisionedConcurrencyConfigResponseTypeDef",
    "PutRuntimeManagementConfigRequestRequestTypeDef",
    "PutRuntimeManagementConfigResponseTypeDef",
    "RemoveLayerVersionPermissionRequestRequestTypeDef",
    "RemovePermissionRequestRequestTypeDef",
    "ResponseMetadataTypeDef",
    "RuntimeVersionConfigTypeDef",
    "RuntimeVersionErrorTypeDef",
    "ScalingConfigTypeDef",
    "SelfManagedEventSourceOutputTypeDef",
    "SelfManagedEventSourceTypeDef",
    "SelfManagedKafkaEventSourceConfigTypeDef",
    "SnapStartResponseTypeDef",
    "SnapStartTypeDef",
    "SourceAccessConfigurationTypeDef",
    "TagResourceRequestRequestTypeDef",
    "TagsErrorTypeDef",
    "TimestampTypeDef",
    "TracingConfigResponseTypeDef",
    "TracingConfigTypeDef",
    "UntagResourceRequestRequestTypeDef",
    "UpdateAliasRequestRequestTypeDef",
    "UpdateCodeSigningConfigRequestRequestTypeDef",
    "UpdateCodeSigningConfigResponseTypeDef",
    "UpdateEventSourceMappingRequestRequestTypeDef",
    "UpdateFunctionCodeRequestRequestTypeDef",
    "UpdateFunctionConfigurationRequestRequestTypeDef",
    "UpdateFunctionEventInvokeConfigRequestRequestTypeDef",
    "UpdateFunctionUrlConfigRequestRequestTypeDef",
    "UpdateFunctionUrlConfigResponseTypeDef",
    "VpcConfigResponseTypeDef",
    "VpcConfigTypeDef",
    "WaiterConfigTypeDef",
)


class AccountLimitTypeDef(TypedDict):
    TotalCodeSize: NotRequired[int]
    CodeSizeUnzipped: NotRequired[int]
    CodeSizeZipped: NotRequired[int]
    ConcurrentExecutions: NotRequired[int]
    UnreservedConcurrentExecutions: NotRequired[int]


class AccountUsageTypeDef(TypedDict):
    TotalCodeSize: NotRequired[int]
    FunctionCount: NotRequired[int]


class AddLayerVersionPermissionRequestRequestTypeDef(TypedDict):
    LayerName: str
    VersionNumber: int
    StatementId: str
    Action: str
    Principal: str
    OrganizationId: NotRequired[str]
    RevisionId: NotRequired[str]


class ResponseMetadataTypeDef(TypedDict):
    RequestId: str
    HTTPStatusCode: int
    HTTPHeaders: Dict[str, str]
    RetryAttempts: int
    HostId: NotRequired[str]


class AddPermissionRequestRequestTypeDef(TypedDict):
    FunctionName: str
    StatementId: str
    Action: str
    Principal: str
    SourceArn: NotRequired[str]
    SourceAccount: NotRequired[str]
    EventSourceToken: NotRequired[str]
    Qualifier: NotRequired[str]
    RevisionId: NotRequired[str]
    PrincipalOrgID: NotRequired[str]
    FunctionUrlAuthType: NotRequired[FunctionUrlAuthTypeType]


class AliasRoutingConfigurationOutputTypeDef(TypedDict):
    AdditionalVersionWeights: NotRequired[Dict[str, float]]


class AliasRoutingConfigurationTypeDef(TypedDict):
    AdditionalVersionWeights: NotRequired[Mapping[str, float]]


class AllowedPublishersOutputTypeDef(TypedDict):
    SigningProfileVersionArns: List[str]


class AllowedPublishersTypeDef(TypedDict):
    SigningProfileVersionArns: Sequence[str]


class AmazonManagedKafkaEventSourceConfigTypeDef(TypedDict):
    ConsumerGroupId: NotRequired[str]


BlobTypeDef = Union[str, bytes, IO[Any], StreamingBody]


class CodeSigningPoliciesTypeDef(TypedDict):
    UntrustedArtifactOnDeployment: NotRequired[CodeSigningPolicyType]


class ConcurrencyTypeDef(TypedDict):
    ReservedConcurrentExecutions: NotRequired[int]


class CorsOutputTypeDef(TypedDict):
    AllowCredentials: NotRequired[bool]
    AllowHeaders: NotRequired[List[str]]
    AllowMethods: NotRequired[List[str]]
    AllowOrigins: NotRequired[List[str]]
    ExposeHeaders: NotRequired[List[str]]
    MaxAge: NotRequired[int]


class CorsTypeDef(TypedDict):
    AllowCredentials: NotRequired[bool]
    AllowHeaders: NotRequired[Sequence[str]]
    AllowMethods: NotRequired[Sequence[str]]
    AllowOrigins: NotRequired[Sequence[str]]
    ExposeHeaders: NotRequired[Sequence[str]]
    MaxAge: NotRequired[int]


class DocumentDBEventSourceConfigTypeDef(TypedDict):
    DatabaseName: NotRequired[str]
    CollectionName: NotRequired[str]
    FullDocument: NotRequired[FullDocumentType]


class EventSourceMappingMetricsConfigTypeDef(TypedDict):
    Metrics: NotRequired[Sequence[Literal["EventCount"]]]


class ProvisionedPollerConfigTypeDef(TypedDict):
    MinimumPollers: NotRequired[int]
    MaximumPollers: NotRequired[int]


class ScalingConfigTypeDef(TypedDict):
    MaximumConcurrency: NotRequired[int]


class SelfManagedEventSourceTypeDef(TypedDict):
    Endpoints: NotRequired[Mapping[Literal["KAFKA_BOOTSTRAP_SERVERS"], Sequence[str]]]


class SelfManagedKafkaEventSourceConfigTypeDef(TypedDict):
    ConsumerGroupId: NotRequired[str]


SourceAccessConfigurationTypeDef = TypedDict(
    "SourceAccessConfigurationTypeDef",
    {
        "Type": NotRequired[SourceAccessTypeType],
        "URI": NotRequired[str],
    },
)
TimestampTypeDef = Union[datetime, str]


class DeadLetterConfigTypeDef(TypedDict):
    TargetArn: NotRequired[str]


class EnvironmentTypeDef(TypedDict):
    Variables: NotRequired[Mapping[str, str]]


class EphemeralStorageTypeDef(TypedDict):
    Size: int


class FileSystemConfigTypeDef(TypedDict):
    Arn: str
    LocalMountPath: str


class ImageConfigTypeDef(TypedDict):
    EntryPoint: NotRequired[Sequence[str]]
    Command: NotRequired[Sequence[str]]
    WorkingDirectory: NotRequired[str]


class LoggingConfigTypeDef(TypedDict):
    LogFormat: NotRequired[LogFormatType]
    ApplicationLogLevel: NotRequired[ApplicationLogLevelType]
    SystemLogLevel: NotRequired[SystemLogLevelType]
    LogGroup: NotRequired[str]


class SnapStartTypeDef(TypedDict):
    ApplyOn: NotRequired[SnapStartApplyOnType]


class TracingConfigTypeDef(TypedDict):
    Mode: NotRequired[TracingModeType]


class VpcConfigTypeDef(TypedDict):
    SubnetIds: NotRequired[Sequence[str]]
    SecurityGroupIds: NotRequired[Sequence[str]]
    Ipv6AllowedForDualStack: NotRequired[bool]


class DeleteAliasRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Name: str


class DeleteCodeSigningConfigRequestRequestTypeDef(TypedDict):
    CodeSigningConfigArn: str


class DeleteEventSourceMappingRequestRequestTypeDef(TypedDict):
    UUID: str


class DeleteFunctionCodeSigningConfigRequestRequestTypeDef(TypedDict):
    FunctionName: str


class DeleteFunctionConcurrencyRequestRequestTypeDef(TypedDict):
    FunctionName: str


class DeleteFunctionEventInvokeConfigRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Qualifier: NotRequired[str]


class DeleteFunctionRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Qualifier: NotRequired[str]


class DeleteFunctionUrlConfigRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Qualifier: NotRequired[str]


class DeleteLayerVersionRequestRequestTypeDef(TypedDict):
    LayerName: str
    VersionNumber: int


class DeleteProvisionedConcurrencyConfigRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Qualifier: str


class OnFailureTypeDef(TypedDict):
    Destination: NotRequired[str]


class OnSuccessTypeDef(TypedDict):
    Destination: NotRequired[str]


class EnvironmentErrorTypeDef(TypedDict):
    ErrorCode: NotRequired[str]
    Message: NotRequired[str]


class EventSourceMappingMetricsConfigOutputTypeDef(TypedDict):
    Metrics: NotRequired[List[Literal["EventCount"]]]


class FilterCriteriaErrorTypeDef(TypedDict):
    ErrorCode: NotRequired[str]
    Message: NotRequired[str]


class SelfManagedEventSourceOutputTypeDef(TypedDict):
    Endpoints: NotRequired[Dict[Literal["KAFKA_BOOTSTRAP_SERVERS"], List[str]]]


FilterTypeDef = TypedDict(
    "FilterTypeDef",
    {
        "Pattern": NotRequired[str],
    },
)


class FunctionCodeLocationTypeDef(TypedDict):
    RepositoryType: NotRequired[str]
    Location: NotRequired[str]
    ImageUri: NotRequired[str]
    ResolvedImageUri: NotRequired[str]
    SourceKMSKeyArn: NotRequired[str]


class LayerTypeDef(TypedDict):
    Arn: NotRequired[str]
    CodeSize: NotRequired[int]
    SigningProfileVersionArn: NotRequired[str]
    SigningJobArn: NotRequired[str]


class SnapStartResponseTypeDef(TypedDict):
    ApplyOn: NotRequired[SnapStartApplyOnType]
    OptimizationStatus: NotRequired[SnapStartOptimizationStatusType]


class TracingConfigResponseTypeDef(TypedDict):
    Mode: NotRequired[TracingModeType]


class VpcConfigResponseTypeDef(TypedDict):
    SubnetIds: NotRequired[List[str]]
    SecurityGroupIds: NotRequired[List[str]]
    VpcId: NotRequired[str]
    Ipv6AllowedForDualStack: NotRequired[bool]


class GetAliasRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Name: str


class GetCodeSigningConfigRequestRequestTypeDef(TypedDict):
    CodeSigningConfigArn: str


class GetEventSourceMappingRequestRequestTypeDef(TypedDict):
    UUID: str


class GetFunctionCodeSigningConfigRequestRequestTypeDef(TypedDict):
    FunctionName: str


class GetFunctionConcurrencyRequestRequestTypeDef(TypedDict):
    FunctionName: str


class GetFunctionConfigurationRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Qualifier: NotRequired[str]


class WaiterConfigTypeDef(TypedDict):
    Delay: NotRequired[int]
    MaxAttempts: NotRequired[int]


class GetFunctionEventInvokeConfigRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Qualifier: NotRequired[str]


class GetFunctionRecursionConfigRequestRequestTypeDef(TypedDict):
    FunctionName: str


class GetFunctionRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Qualifier: NotRequired[str]


class TagsErrorTypeDef(TypedDict):
    ErrorCode: str
    Message: str


class GetFunctionUrlConfigRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Qualifier: NotRequired[str]


class GetLayerVersionByArnRequestRequestTypeDef(TypedDict):
    Arn: str


class GetLayerVersionPolicyRequestRequestTypeDef(TypedDict):
    LayerName: str
    VersionNumber: int


class GetLayerVersionRequestRequestTypeDef(TypedDict):
    LayerName: str
    VersionNumber: int


class LayerVersionContentOutputTypeDef(TypedDict):
    Location: NotRequired[str]
    CodeSha256: NotRequired[str]
    CodeSize: NotRequired[int]
    SigningProfileVersionArn: NotRequired[str]
    SigningJobArn: NotRequired[str]


class GetPolicyRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Qualifier: NotRequired[str]


class GetProvisionedConcurrencyConfigRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Qualifier: str


class GetRuntimeManagementConfigRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Qualifier: NotRequired[str]


class ImageConfigErrorTypeDef(TypedDict):
    ErrorCode: NotRequired[str]
    Message: NotRequired[str]


class ImageConfigOutputTypeDef(TypedDict):
    EntryPoint: NotRequired[List[str]]
    Command: NotRequired[List[str]]
    WorkingDirectory: NotRequired[str]


class InvokeResponseStreamUpdateTypeDef(TypedDict):
    Payload: NotRequired[bytes]


class InvokeWithResponseStreamCompleteEventTypeDef(TypedDict):
    ErrorCode: NotRequired[str]
    ErrorDetails: NotRequired[str]
    LogResult: NotRequired[str]


class LayerVersionsListItemTypeDef(TypedDict):
    LayerVersionArn: NotRequired[str]
    Version: NotRequired[int]
    Description: NotRequired[str]
    CreatedDate: NotRequired[str]
    CompatibleRuntimes: NotRequired[List[RuntimeType]]
    LicenseInfo: NotRequired[str]
    CompatibleArchitectures: NotRequired[List[ArchitectureType]]


class PaginatorConfigTypeDef(TypedDict):
    MaxItems: NotRequired[int]
    PageSize: NotRequired[int]
    StartingToken: NotRequired[str]


class ListAliasesRequestRequestTypeDef(TypedDict):
    FunctionName: str
    FunctionVersion: NotRequired[str]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]


class ListCodeSigningConfigsRequestRequestTypeDef(TypedDict):
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]


class ListEventSourceMappingsRequestRequestTypeDef(TypedDict):
    EventSourceArn: NotRequired[str]
    FunctionName: NotRequired[str]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]


class ListFunctionEventInvokeConfigsRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]


class ListFunctionUrlConfigsRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]


class ListFunctionsByCodeSigningConfigRequestRequestTypeDef(TypedDict):
    CodeSigningConfigArn: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]


class ListFunctionsRequestRequestTypeDef(TypedDict):
    MasterRegion: NotRequired[str]
    FunctionVersion: NotRequired[Literal["ALL"]]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]


class ListLayerVersionsRequestRequestTypeDef(TypedDict):
    LayerName: str
    CompatibleRuntime: NotRequired[RuntimeType]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]
    CompatibleArchitecture: NotRequired[ArchitectureType]


class ListLayersRequestRequestTypeDef(TypedDict):
    CompatibleRuntime: NotRequired[RuntimeType]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]
    CompatibleArchitecture: NotRequired[ArchitectureType]


class ListProvisionedConcurrencyConfigsRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]


class ProvisionedConcurrencyConfigListItemTypeDef(TypedDict):
    FunctionArn: NotRequired[str]
    RequestedProvisionedConcurrentExecutions: NotRequired[int]
    AvailableProvisionedConcurrentExecutions: NotRequired[int]
    AllocatedProvisionedConcurrentExecutions: NotRequired[int]
    Status: NotRequired[ProvisionedConcurrencyStatusEnumType]
    StatusReason: NotRequired[str]
    LastModified: NotRequired[str]


class ListTagsRequestRequestTypeDef(TypedDict):
    Resource: str


class ListVersionsByFunctionRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]


class PublishVersionRequestRequestTypeDef(TypedDict):
    FunctionName: str
    CodeSha256: NotRequired[str]
    Description: NotRequired[str]
    RevisionId: NotRequired[str]


class PutFunctionCodeSigningConfigRequestRequestTypeDef(TypedDict):
    CodeSigningConfigArn: str
    FunctionName: str


class PutFunctionConcurrencyRequestRequestTypeDef(TypedDict):
    FunctionName: str
    ReservedConcurrentExecutions: int


class PutFunctionRecursionConfigRequestRequestTypeDef(TypedDict):
    FunctionName: str
    RecursiveLoop: RecursiveLoopType


class PutProvisionedConcurrencyConfigRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Qualifier: str
    ProvisionedConcurrentExecutions: int


class PutRuntimeManagementConfigRequestRequestTypeDef(TypedDict):
    FunctionName: str
    UpdateRuntimeOn: UpdateRuntimeOnType
    Qualifier: NotRequired[str]
    RuntimeVersionArn: NotRequired[str]


class RemoveLayerVersionPermissionRequestRequestTypeDef(TypedDict):
    LayerName: str
    VersionNumber: int
    StatementId: str
    RevisionId: NotRequired[str]


class RemovePermissionRequestRequestTypeDef(TypedDict):
    FunctionName: str
    StatementId: str
    Qualifier: NotRequired[str]
    RevisionId: NotRequired[str]


class RuntimeVersionErrorTypeDef(TypedDict):
    ErrorCode: NotRequired[str]
    Message: NotRequired[str]


class TagResourceRequestRequestTypeDef(TypedDict):
    Resource: str
    Tags: Mapping[str, str]


class UntagResourceRequestRequestTypeDef(TypedDict):
    Resource: str
    TagKeys: Sequence[str]


class AddLayerVersionPermissionResponseTypeDef(TypedDict):
    Statement: str
    RevisionId: str
    ResponseMetadata: ResponseMetadataTypeDef


class AddPermissionResponseTypeDef(TypedDict):
    Statement: str
    ResponseMetadata: ResponseMetadataTypeDef


class ConcurrencyResponseTypeDef(TypedDict):
    ReservedConcurrentExecutions: int
    ResponseMetadata: ResponseMetadataTypeDef


class EmptyResponseMetadataTypeDef(TypedDict):
    ResponseMetadata: ResponseMetadataTypeDef


class GetAccountSettingsResponseTypeDef(TypedDict):
    AccountLimit: AccountLimitTypeDef
    AccountUsage: AccountUsageTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class GetFunctionCodeSigningConfigResponseTypeDef(TypedDict):
    CodeSigningConfigArn: str
    FunctionName: str
    ResponseMetadata: ResponseMetadataTypeDef


class GetFunctionConcurrencyResponseTypeDef(TypedDict):
    ReservedConcurrentExecutions: int
    ResponseMetadata: ResponseMetadataTypeDef


class GetFunctionRecursionConfigResponseTypeDef(TypedDict):
    RecursiveLoop: RecursiveLoopType
    ResponseMetadata: ResponseMetadataTypeDef


class GetLayerVersionPolicyResponseTypeDef(TypedDict):
    Policy: str
    RevisionId: str
    ResponseMetadata: ResponseMetadataTypeDef


class GetPolicyResponseTypeDef(TypedDict):
    Policy: str
    RevisionId: str
    ResponseMetadata: ResponseMetadataTypeDef


class GetProvisionedConcurrencyConfigResponseTypeDef(TypedDict):
    RequestedProvisionedConcurrentExecutions: int
    AvailableProvisionedConcurrentExecutions: int
    AllocatedProvisionedConcurrentExecutions: int
    Status: ProvisionedConcurrencyStatusEnumType
    StatusReason: str
    LastModified: str
    ResponseMetadata: ResponseMetadataTypeDef


class GetRuntimeManagementConfigResponseTypeDef(TypedDict):
    UpdateRuntimeOn: UpdateRuntimeOnType
    RuntimeVersionArn: str
    FunctionArn: str
    ResponseMetadata: ResponseMetadataTypeDef


class InvocationResponseTypeDef(TypedDict):
    StatusCode: int
    FunctionError: str
    LogResult: str
    Payload: StreamingBody
    ExecutedVersion: str
    ResponseMetadata: ResponseMetadataTypeDef


class InvokeAsyncResponseTypeDef(TypedDict):
    Status: int
    ResponseMetadata: ResponseMetadataTypeDef


class ListFunctionsByCodeSigningConfigResponseTypeDef(TypedDict):
    NextMarker: str
    FunctionArns: List[str]
    ResponseMetadata: ResponseMetadataTypeDef


class ListTagsResponseTypeDef(TypedDict):
    Tags: Dict[str, str]
    ResponseMetadata: ResponseMetadataTypeDef


class PutFunctionCodeSigningConfigResponseTypeDef(TypedDict):
    CodeSigningConfigArn: str
    FunctionName: str
    ResponseMetadata: ResponseMetadataTypeDef


class PutFunctionRecursionConfigResponseTypeDef(TypedDict):
    RecursiveLoop: RecursiveLoopType
    ResponseMetadata: ResponseMetadataTypeDef


class PutProvisionedConcurrencyConfigResponseTypeDef(TypedDict):
    RequestedProvisionedConcurrentExecutions: int
    AvailableProvisionedConcurrentExecutions: int
    AllocatedProvisionedConcurrentExecutions: int
    Status: ProvisionedConcurrencyStatusEnumType
    StatusReason: str
    LastModified: str
    ResponseMetadata: ResponseMetadataTypeDef


class PutRuntimeManagementConfigResponseTypeDef(TypedDict):
    UpdateRuntimeOn: UpdateRuntimeOnType
    FunctionArn: str
    RuntimeVersionArn: str
    ResponseMetadata: ResponseMetadataTypeDef


class AliasConfigurationResponseTypeDef(TypedDict):
    AliasArn: str
    Name: str
    FunctionVersion: str
    Description: str
    RoutingConfig: AliasRoutingConfigurationOutputTypeDef
    RevisionId: str
    ResponseMetadata: ResponseMetadataTypeDef


class AliasConfigurationTypeDef(TypedDict):
    AliasArn: NotRequired[str]
    Name: NotRequired[str]
    FunctionVersion: NotRequired[str]
    Description: NotRequired[str]
    RoutingConfig: NotRequired[AliasRoutingConfigurationOutputTypeDef]
    RevisionId: NotRequired[str]


class CreateAliasRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Name: str
    FunctionVersion: str
    Description: NotRequired[str]
    RoutingConfig: NotRequired[AliasRoutingConfigurationTypeDef]


class UpdateAliasRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Name: str
    FunctionVersion: NotRequired[str]
    Description: NotRequired[str]
    RoutingConfig: NotRequired[AliasRoutingConfigurationTypeDef]
    RevisionId: NotRequired[str]


class FunctionCodeTypeDef(TypedDict):
    ZipFile: NotRequired[BlobTypeDef]
    S3Bucket: NotRequired[str]
    S3Key: NotRequired[str]
    S3ObjectVersion: NotRequired[str]
    ImageUri: NotRequired[str]
    SourceKMSKeyArn: NotRequired[str]


class InvocationRequestRequestTypeDef(TypedDict):
    FunctionName: str
    InvocationType: NotRequired[InvocationTypeType]
    LogType: NotRequired[LogTypeType]
    ClientContext: NotRequired[str]
    Payload: NotRequired[BlobTypeDef]
    Qualifier: NotRequired[str]


class InvokeAsyncRequestRequestTypeDef(TypedDict):
    FunctionName: str
    InvokeArgs: BlobTypeDef


class InvokeWithResponseStreamRequestRequestTypeDef(TypedDict):
    FunctionName: str
    InvocationType: NotRequired[ResponseStreamingInvocationTypeType]
    LogType: NotRequired[LogTypeType]
    ClientContext: NotRequired[str]
    Qualifier: NotRequired[str]
    Payload: NotRequired[BlobTypeDef]


class LayerVersionContentInputTypeDef(TypedDict):
    S3Bucket: NotRequired[str]
    S3Key: NotRequired[str]
    S3ObjectVersion: NotRequired[str]
    ZipFile: NotRequired[BlobTypeDef]


class UpdateFunctionCodeRequestRequestTypeDef(TypedDict):
    FunctionName: str
    ZipFile: NotRequired[BlobTypeDef]
    S3Bucket: NotRequired[str]
    S3Key: NotRequired[str]
    S3ObjectVersion: NotRequired[str]
    ImageUri: NotRequired[str]
    Publish: NotRequired[bool]
    DryRun: NotRequired[bool]
    RevisionId: NotRequired[str]
    Architectures: NotRequired[Sequence[ArchitectureType]]
    SourceKMSKeyArn: NotRequired[str]


class CodeSigningConfigTypeDef(TypedDict):
    CodeSigningConfigId: str
    CodeSigningConfigArn: str
    AllowedPublishers: AllowedPublishersOutputTypeDef
    CodeSigningPolicies: CodeSigningPoliciesTypeDef
    LastModified: str
    Description: NotRequired[str]


class CreateCodeSigningConfigRequestRequestTypeDef(TypedDict):
    AllowedPublishers: AllowedPublishersTypeDef
    Description: NotRequired[str]
    CodeSigningPolicies: NotRequired[CodeSigningPoliciesTypeDef]
    Tags: NotRequired[Mapping[str, str]]


class UpdateCodeSigningConfigRequestRequestTypeDef(TypedDict):
    CodeSigningConfigArn: str
    Description: NotRequired[str]
    AllowedPublishers: NotRequired[AllowedPublishersTypeDef]
    CodeSigningPolicies: NotRequired[CodeSigningPoliciesTypeDef]


class CreateFunctionUrlConfigResponseTypeDef(TypedDict):
    FunctionUrl: str
    FunctionArn: str
    AuthType: FunctionUrlAuthTypeType
    Cors: CorsOutputTypeDef
    CreationTime: str
    InvokeMode: InvokeModeType
    ResponseMetadata: ResponseMetadataTypeDef


class FunctionUrlConfigTypeDef(TypedDict):
    FunctionUrl: str
    FunctionArn: str
    CreationTime: str
    LastModifiedTime: str
    AuthType: FunctionUrlAuthTypeType
    Cors: NotRequired[CorsOutputTypeDef]
    InvokeMode: NotRequired[InvokeModeType]


class GetFunctionUrlConfigResponseTypeDef(TypedDict):
    FunctionUrl: str
    FunctionArn: str
    AuthType: FunctionUrlAuthTypeType
    Cors: CorsOutputTypeDef
    CreationTime: str
    LastModifiedTime: str
    InvokeMode: InvokeModeType
    ResponseMetadata: ResponseMetadataTypeDef


class UpdateFunctionUrlConfigResponseTypeDef(TypedDict):
    FunctionUrl: str
    FunctionArn: str
    AuthType: FunctionUrlAuthTypeType
    Cors: CorsOutputTypeDef
    CreationTime: str
    LastModifiedTime: str
    InvokeMode: InvokeModeType
    ResponseMetadata: ResponseMetadataTypeDef


class CreateFunctionUrlConfigRequestRequestTypeDef(TypedDict):
    FunctionName: str
    AuthType: FunctionUrlAuthTypeType
    Qualifier: NotRequired[str]
    Cors: NotRequired[CorsTypeDef]
    InvokeMode: NotRequired[InvokeModeType]


class UpdateFunctionUrlConfigRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Qualifier: NotRequired[str]
    AuthType: NotRequired[FunctionUrlAuthTypeType]
    Cors: NotRequired[CorsTypeDef]
    InvokeMode: NotRequired[InvokeModeType]


class UpdateFunctionConfigurationRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Role: NotRequired[str]
    Handler: NotRequired[str]
    Description: NotRequired[str]
    Timeout: NotRequired[int]
    MemorySize: NotRequired[int]
    VpcConfig: NotRequired[VpcConfigTypeDef]
    Environment: NotRequired[EnvironmentTypeDef]
    Runtime: NotRequired[RuntimeType]
    DeadLetterConfig: NotRequired[DeadLetterConfigTypeDef]
    KMSKeyArn: NotRequired[str]
    TracingConfig: NotRequired[TracingConfigTypeDef]
    RevisionId: NotRequired[str]
    Layers: NotRequired[Sequence[str]]
    FileSystemConfigs: NotRequired[Sequence[FileSystemConfigTypeDef]]
    ImageConfig: NotRequired[ImageConfigTypeDef]
    EphemeralStorage: NotRequired[EphemeralStorageTypeDef]
    SnapStart: NotRequired[SnapStartTypeDef]
    LoggingConfig: NotRequired[LoggingConfigTypeDef]


class DestinationConfigTypeDef(TypedDict):
    OnSuccess: NotRequired[OnSuccessTypeDef]
    OnFailure: NotRequired[OnFailureTypeDef]


class EnvironmentResponseTypeDef(TypedDict):
    Variables: NotRequired[Dict[str, str]]
    Error: NotRequired[EnvironmentErrorTypeDef]


class FilterCriteriaOutputTypeDef(TypedDict):
    Filters: NotRequired[List[FilterTypeDef]]


class FilterCriteriaTypeDef(TypedDict):
    Filters: NotRequired[Sequence[FilterTypeDef]]


class GetFunctionConfigurationRequestWaitTypeDef(TypedDict):
    FunctionName: str
    Qualifier: NotRequired[str]
    WaiterConfig: NotRequired[WaiterConfigTypeDef]


class GetFunctionRequestWaitTypeDef(TypedDict):
    FunctionName: str
    Qualifier: NotRequired[str]
    WaiterConfig: NotRequired[WaiterConfigTypeDef]


class GetLayerVersionResponseTypeDef(TypedDict):
    Content: LayerVersionContentOutputTypeDef
    LayerArn: str
    LayerVersionArn: str
    Description: str
    CreatedDate: str
    Version: int
    CompatibleRuntimes: List[RuntimeType]
    LicenseInfo: str
    CompatibleArchitectures: List[ArchitectureType]
    ResponseMetadata: ResponseMetadataTypeDef


class PublishLayerVersionResponseTypeDef(TypedDict):
    Content: LayerVersionContentOutputTypeDef
    LayerArn: str
    LayerVersionArn: str
    Description: str
    CreatedDate: str
    Version: int
    CompatibleRuntimes: List[RuntimeType]
    LicenseInfo: str
    CompatibleArchitectures: List[ArchitectureType]
    ResponseMetadata: ResponseMetadataTypeDef


class ImageConfigResponseTypeDef(TypedDict):
    ImageConfig: NotRequired[ImageConfigOutputTypeDef]
    Error: NotRequired[ImageConfigErrorTypeDef]


class InvokeWithResponseStreamResponseEventTypeDef(TypedDict):
    PayloadChunk: NotRequired[InvokeResponseStreamUpdateTypeDef]
    InvokeComplete: NotRequired[InvokeWithResponseStreamCompleteEventTypeDef]


class LayersListItemTypeDef(TypedDict):
    LayerName: NotRequired[str]
    LayerArn: NotRequired[str]
    LatestMatchingVersion: NotRequired[LayerVersionsListItemTypeDef]


class ListLayerVersionsResponseTypeDef(TypedDict):
    NextMarker: str
    LayerVersions: List[LayerVersionsListItemTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class ListAliasesRequestPaginateTypeDef(TypedDict):
    FunctionName: str
    FunctionVersion: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class ListCodeSigningConfigsRequestPaginateTypeDef(TypedDict):
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class ListEventSourceMappingsRequestPaginateTypeDef(TypedDict):
    EventSourceArn: NotRequired[str]
    FunctionName: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class ListFunctionEventInvokeConfigsRequestPaginateTypeDef(TypedDict):
    FunctionName: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class ListFunctionUrlConfigsRequestPaginateTypeDef(TypedDict):
    FunctionName: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class ListFunctionsByCodeSigningConfigRequestPaginateTypeDef(TypedDict):
    CodeSigningConfigArn: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class ListFunctionsRequestPaginateTypeDef(TypedDict):
    MasterRegion: NotRequired[str]
    FunctionVersion: NotRequired[Literal["ALL"]]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class ListLayerVersionsRequestPaginateTypeDef(TypedDict):
    LayerName: str
    CompatibleRuntime: NotRequired[RuntimeType]
    CompatibleArchitecture: NotRequired[ArchitectureType]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class ListLayersRequestPaginateTypeDef(TypedDict):
    CompatibleRuntime: NotRequired[RuntimeType]
    CompatibleArchitecture: NotRequired[ArchitectureType]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class ListProvisionedConcurrencyConfigsRequestPaginateTypeDef(TypedDict):
    FunctionName: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class ListVersionsByFunctionRequestPaginateTypeDef(TypedDict):
    FunctionName: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class ListProvisionedConcurrencyConfigsResponseTypeDef(TypedDict):
    ProvisionedConcurrencyConfigs: List[ProvisionedConcurrencyConfigListItemTypeDef]
    NextMarker: str
    ResponseMetadata: ResponseMetadataTypeDef


class RuntimeVersionConfigTypeDef(TypedDict):
    RuntimeVersionArn: NotRequired[str]
    Error: NotRequired[RuntimeVersionErrorTypeDef]


class ListAliasesResponseTypeDef(TypedDict):
    NextMarker: str
    Aliases: List[AliasConfigurationTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class CreateFunctionRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Role: str
    Code: FunctionCodeTypeDef
    Runtime: NotRequired[RuntimeType]
    Handler: NotRequired[str]
    Description: NotRequired[str]
    Timeout: NotRequired[int]
    MemorySize: NotRequired[int]
    Publish: NotRequired[bool]
    VpcConfig: NotRequired[VpcConfigTypeDef]
    PackageType: NotRequired[PackageTypeType]
    DeadLetterConfig: NotRequired[DeadLetterConfigTypeDef]
    Environment: NotRequired[EnvironmentTypeDef]
    KMSKeyArn: NotRequired[str]
    TracingConfig: NotRequired[TracingConfigTypeDef]
    Tags: NotRequired[Mapping[str, str]]
    Layers: NotRequired[Sequence[str]]
    FileSystemConfigs: NotRequired[Sequence[FileSystemConfigTypeDef]]
    ImageConfig: NotRequired[ImageConfigTypeDef]
    CodeSigningConfigArn: NotRequired[str]
    Architectures: NotRequired[Sequence[ArchitectureType]]
    EphemeralStorage: NotRequired[EphemeralStorageTypeDef]
    SnapStart: NotRequired[SnapStartTypeDef]
    LoggingConfig: NotRequired[LoggingConfigTypeDef]


class PublishLayerVersionRequestRequestTypeDef(TypedDict):
    LayerName: str
    Content: LayerVersionContentInputTypeDef
    Description: NotRequired[str]
    CompatibleRuntimes: NotRequired[Sequence[RuntimeType]]
    LicenseInfo: NotRequired[str]
    CompatibleArchitectures: NotRequired[Sequence[ArchitectureType]]


class CreateCodeSigningConfigResponseTypeDef(TypedDict):
    CodeSigningConfig: CodeSigningConfigTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class GetCodeSigningConfigResponseTypeDef(TypedDict):
    CodeSigningConfig: CodeSigningConfigTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class ListCodeSigningConfigsResponseTypeDef(TypedDict):
    NextMarker: str
    CodeSigningConfigs: List[CodeSigningConfigTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class UpdateCodeSigningConfigResponseTypeDef(TypedDict):
    CodeSigningConfig: CodeSigningConfigTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class ListFunctionUrlConfigsResponseTypeDef(TypedDict):
    FunctionUrlConfigs: List[FunctionUrlConfigTypeDef]
    NextMarker: str
    ResponseMetadata: ResponseMetadataTypeDef


class FunctionEventInvokeConfigResponseTypeDef(TypedDict):
    LastModified: datetime
    FunctionArn: str
    MaximumRetryAttempts: int
    MaximumEventAgeInSeconds: int
    DestinationConfig: DestinationConfigTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class FunctionEventInvokeConfigTypeDef(TypedDict):
    LastModified: NotRequired[datetime]
    FunctionArn: NotRequired[str]
    MaximumRetryAttempts: NotRequired[int]
    MaximumEventAgeInSeconds: NotRequired[int]
    DestinationConfig: NotRequired[DestinationConfigTypeDef]


class PutFunctionEventInvokeConfigRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Qualifier: NotRequired[str]
    MaximumRetryAttempts: NotRequired[int]
    MaximumEventAgeInSeconds: NotRequired[int]
    DestinationConfig: NotRequired[DestinationConfigTypeDef]


class UpdateFunctionEventInvokeConfigRequestRequestTypeDef(TypedDict):
    FunctionName: str
    Qualifier: NotRequired[str]
    MaximumRetryAttempts: NotRequired[int]
    MaximumEventAgeInSeconds: NotRequired[int]
    DestinationConfig: NotRequired[DestinationConfigTypeDef]


class EventSourceMappingConfigurationResponseTypeDef(TypedDict):
    UUID: str
    StartingPosition: EventSourcePositionType
    StartingPositionTimestamp: datetime
    BatchSize: int
    MaximumBatchingWindowInSeconds: int
    ParallelizationFactor: int
    EventSourceArn: str
    FilterCriteria: FilterCriteriaOutputTypeDef
    FunctionArn: str
    LastModified: datetime
    LastProcessingResult: str
    State: str
    StateTransitionReason: str
    DestinationConfig: DestinationConfigTypeDef
    Topics: List[str]
    Queues: List[str]
    SourceAccessConfigurations: List[SourceAccessConfigurationTypeDef]
    SelfManagedEventSource: SelfManagedEventSourceOutputTypeDef
    MaximumRecordAgeInSeconds: int
    BisectBatchOnFunctionError: bool
    MaximumRetryAttempts: int
    TumblingWindowInSeconds: int
    FunctionResponseTypes: List[Literal["ReportBatchItemFailures"]]
    AmazonManagedKafkaEventSourceConfig: AmazonManagedKafkaEventSourceConfigTypeDef
    SelfManagedKafkaEventSourceConfig: SelfManagedKafkaEventSourceConfigTypeDef
    ScalingConfig: ScalingConfigTypeDef
    DocumentDBEventSourceConfig: DocumentDBEventSourceConfigTypeDef
    KMSKeyArn: str
    FilterCriteriaError: FilterCriteriaErrorTypeDef
    EventSourceMappingArn: str
    MetricsConfig: EventSourceMappingMetricsConfigOutputTypeDef
    ProvisionedPollerConfig: ProvisionedPollerConfigTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class EventSourceMappingConfigurationTypeDef(TypedDict):
    UUID: NotRequired[str]
    StartingPosition: NotRequired[EventSourcePositionType]
    StartingPositionTimestamp: NotRequired[datetime]
    BatchSize: NotRequired[int]
    MaximumBatchingWindowInSeconds: NotRequired[int]
    ParallelizationFactor: NotRequired[int]
    EventSourceArn: NotRequired[str]
    FilterCriteria: NotRequired[FilterCriteriaOutputTypeDef]
    FunctionArn: NotRequired[str]
    LastModified: NotRequired[datetime]
    LastProcessingResult: NotRequired[str]
    State: NotRequired[str]
    StateTransitionReason: NotRequired[str]
    DestinationConfig: NotRequired[DestinationConfigTypeDef]
    Topics: NotRequired[List[str]]
    Queues: NotRequired[List[str]]
    SourceAccessConfigurations: NotRequired[List[SourceAccessConfigurationTypeDef]]
    SelfManagedEventSource: NotRequired[SelfManagedEventSourceOutputTypeDef]
    MaximumRecordAgeInSeconds: NotRequired[int]
    BisectBatchOnFunctionError: NotRequired[bool]
    MaximumRetryAttempts: NotRequired[int]
    TumblingWindowInSeconds: NotRequired[int]
    FunctionResponseTypes: NotRequired[List[Literal["ReportBatchItemFailures"]]]
    AmazonManagedKafkaEventSourceConfig: NotRequired[AmazonManagedKafkaEventSourceConfigTypeDef]
    SelfManagedKafkaEventSourceConfig: NotRequired[SelfManagedKafkaEventSourceConfigTypeDef]
    ScalingConfig: NotRequired[ScalingConfigTypeDef]
    DocumentDBEventSourceConfig: NotRequired[DocumentDBEventSourceConfigTypeDef]
    KMSKeyArn: NotRequired[str]
    FilterCriteriaError: NotRequired[FilterCriteriaErrorTypeDef]
    EventSourceMappingArn: NotRequired[str]
    MetricsConfig: NotRequired[EventSourceMappingMetricsConfigOutputTypeDef]
    ProvisionedPollerConfig: NotRequired[ProvisionedPollerConfigTypeDef]


class CreateEventSourceMappingRequestRequestTypeDef(TypedDict):
    FunctionName: str
    EventSourceArn: NotRequired[str]
    Enabled: NotRequired[bool]
    BatchSize: NotRequired[int]
    FilterCriteria: NotRequired[FilterCriteriaTypeDef]
    MaximumBatchingWindowInSeconds: NotRequired[int]
    ParallelizationFactor: NotRequired[int]
    StartingPosition: NotRequired[EventSourcePositionType]
    StartingPositionTimestamp: NotRequired[TimestampTypeDef]
    DestinationConfig: NotRequired[DestinationConfigTypeDef]
    MaximumRecordAgeInSeconds: NotRequired[int]
    BisectBatchOnFunctionError: NotRequired[bool]
    MaximumRetryAttempts: NotRequired[int]
    Tags: NotRequired[Mapping[str, str]]
    TumblingWindowInSeconds: NotRequired[int]
    Topics: NotRequired[Sequence[str]]
    Queues: NotRequired[Sequence[str]]
    SourceAccessConfigurations: NotRequired[Sequence[SourceAccessConfigurationTypeDef]]
    SelfManagedEventSource: NotRequired[SelfManagedEventSourceTypeDef]
    FunctionResponseTypes: NotRequired[Sequence[Literal["ReportBatchItemFailures"]]]
    AmazonManagedKafkaEventSourceConfig: NotRequired[AmazonManagedKafkaEventSourceConfigTypeDef]
    SelfManagedKafkaEventSourceConfig: NotRequired[SelfManagedKafkaEventSourceConfigTypeDef]
    ScalingConfig: NotRequired[ScalingConfigTypeDef]
    DocumentDBEventSourceConfig: NotRequired[DocumentDBEventSourceConfigTypeDef]
    KMSKeyArn: NotRequired[str]
    MetricsConfig: NotRequired[EventSourceMappingMetricsConfigTypeDef]
    ProvisionedPollerConfig: NotRequired[ProvisionedPollerConfigTypeDef]


class UpdateEventSourceMappingRequestRequestTypeDef(TypedDict):
    UUID: str
    FunctionName: NotRequired[str]
    Enabled: NotRequired[bool]
    BatchSize: NotRequired[int]
    FilterCriteria: NotRequired[FilterCriteriaTypeDef]
    MaximumBatchingWindowInSeconds: NotRequired[int]
    DestinationConfig: NotRequired[DestinationConfigTypeDef]
    MaximumRecordAgeInSeconds: NotRequired[int]
    BisectBatchOnFunctionError: NotRequired[bool]
    MaximumRetryAttempts: NotRequired[int]
    ParallelizationFactor: NotRequired[int]
    SourceAccessConfigurations: NotRequired[Sequence[SourceAccessConfigurationTypeDef]]
    TumblingWindowInSeconds: NotRequired[int]
    FunctionResponseTypes: NotRequired[Sequence[Literal["ReportBatchItemFailures"]]]
    ScalingConfig: NotRequired[ScalingConfigTypeDef]
    DocumentDBEventSourceConfig: NotRequired[DocumentDBEventSourceConfigTypeDef]
    KMSKeyArn: NotRequired[str]
    MetricsConfig: NotRequired[EventSourceMappingMetricsConfigTypeDef]
    ProvisionedPollerConfig: NotRequired[ProvisionedPollerConfigTypeDef]


class InvokeWithResponseStreamResponseTypeDef(TypedDict):
    StatusCode: int
    ExecutedVersion: str
    EventStream: EventStream[InvokeWithResponseStreamResponseEventTypeDef]
    ResponseStreamContentType: str
    ResponseMetadata: ResponseMetadataTypeDef


class ListLayersResponseTypeDef(TypedDict):
    NextMarker: str
    Layers: List[LayersListItemTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class FunctionConfigurationResponseTypeDef(TypedDict):
    FunctionName: str
    FunctionArn: str
    Runtime: RuntimeType
    Role: str
    Handler: str
    CodeSize: int
    Description: str
    Timeout: int
    MemorySize: int
    LastModified: str
    CodeSha256: str
    Version: str
    VpcConfig: VpcConfigResponseTypeDef
    DeadLetterConfig: DeadLetterConfigTypeDef
    Environment: EnvironmentResponseTypeDef
    KMSKeyArn: str
    TracingConfig: TracingConfigResponseTypeDef
    MasterArn: str
    RevisionId: str
    Layers: List[LayerTypeDef]
    State: StateType
    StateReason: str
    StateReasonCode: StateReasonCodeType
    LastUpdateStatus: LastUpdateStatusType
    LastUpdateStatusReason: str
    LastUpdateStatusReasonCode: LastUpdateStatusReasonCodeType
    FileSystemConfigs: List[FileSystemConfigTypeDef]
    PackageType: PackageTypeType
    ImageConfigResponse: ImageConfigResponseTypeDef
    SigningProfileVersionArn: str
    SigningJobArn: str
    Architectures: List[ArchitectureType]
    EphemeralStorage: EphemeralStorageTypeDef
    SnapStart: SnapStartResponseTypeDef
    RuntimeVersionConfig: RuntimeVersionConfigTypeDef
    LoggingConfig: LoggingConfigTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class FunctionConfigurationTypeDef(TypedDict):
    FunctionName: NotRequired[str]
    FunctionArn: NotRequired[str]
    Runtime: NotRequired[RuntimeType]
    Role: NotRequired[str]
    Handler: NotRequired[str]
    CodeSize: NotRequired[int]
    Description: NotRequired[str]
    Timeout: NotRequired[int]
    MemorySize: NotRequired[int]
    LastModified: NotRequired[str]
    CodeSha256: NotRequired[str]
    Version: NotRequired[str]
    VpcConfig: NotRequired[VpcConfigResponseTypeDef]
    DeadLetterConfig: NotRequired[DeadLetterConfigTypeDef]
    Environment: NotRequired[EnvironmentResponseTypeDef]
    KMSKeyArn: NotRequired[str]
    TracingConfig: NotRequired[TracingConfigResponseTypeDef]
    MasterArn: NotRequired[str]
    RevisionId: NotRequired[str]
    Layers: NotRequired[List[LayerTypeDef]]
    State: NotRequired[StateType]
    StateReason: NotRequired[str]
    StateReasonCode: NotRequired[StateReasonCodeType]
    LastUpdateStatus: NotRequired[LastUpdateStatusType]
    LastUpdateStatusReason: NotRequired[str]
    LastUpdateStatusReasonCode: NotRequired[LastUpdateStatusReasonCodeType]
    FileSystemConfigs: NotRequired[List[FileSystemConfigTypeDef]]
    PackageType: NotRequired[PackageTypeType]
    ImageConfigResponse: NotRequired[ImageConfigResponseTypeDef]
    SigningProfileVersionArn: NotRequired[str]
    SigningJobArn: NotRequired[str]
    Architectures: NotRequired[List[ArchitectureType]]
    EphemeralStorage: NotRequired[EphemeralStorageTypeDef]
    SnapStart: NotRequired[SnapStartResponseTypeDef]
    RuntimeVersionConfig: NotRequired[RuntimeVersionConfigTypeDef]
    LoggingConfig: NotRequired[LoggingConfigTypeDef]


class ListFunctionEventInvokeConfigsResponseTypeDef(TypedDict):
    FunctionEventInvokeConfigs: List[FunctionEventInvokeConfigTypeDef]
    NextMarker: str
    ResponseMetadata: ResponseMetadataTypeDef


class ListEventSourceMappingsResponseTypeDef(TypedDict):
    NextMarker: str
    EventSourceMappings: List[EventSourceMappingConfigurationTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class GetFunctionResponseTypeDef(TypedDict):
    Configuration: FunctionConfigurationTypeDef
    Code: FunctionCodeLocationTypeDef
    Tags: Dict[str, str]
    TagsError: TagsErrorTypeDef
    Concurrency: ConcurrencyTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class ListFunctionsResponseTypeDef(TypedDict):
    NextMarker: str
    Functions: List[FunctionConfigurationTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class ListVersionsByFunctionResponseTypeDef(TypedDict):
    NextMarker: str
    Versions: List[FunctionConfigurationTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef
