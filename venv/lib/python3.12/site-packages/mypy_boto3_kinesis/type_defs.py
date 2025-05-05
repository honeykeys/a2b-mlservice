"""
Type annotations for kinesis service type definitions.

[Documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_kinesis/type_defs/)

Usage::

    ```python
    from mypy_boto3_kinesis.type_defs import AddTagsToStreamInputRequestTypeDef

    data: AddTagsToStreamInputRequestTypeDef = ...
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
    ConsumerStatusType,
    EncryptionTypeType,
    MetricsNameType,
    ShardFilterTypeType,
    ShardIteratorTypeType,
    StreamModeType,
    StreamStatusType,
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
    "AddTagsToStreamInputRequestTypeDef",
    "BlobTypeDef",
    "ChildShardTypeDef",
    "ConsumerDescriptionTypeDef",
    "ConsumerTypeDef",
    "CreateStreamInputRequestTypeDef",
    "DecreaseStreamRetentionPeriodInputRequestTypeDef",
    "DeleteResourcePolicyInputRequestTypeDef",
    "DeleteStreamInputRequestTypeDef",
    "DeregisterStreamConsumerInputRequestTypeDef",
    "DescribeLimitsOutputTypeDef",
    "DescribeStreamConsumerInputRequestTypeDef",
    "DescribeStreamConsumerOutputTypeDef",
    "DescribeStreamInputPaginateTypeDef",
    "DescribeStreamInputRequestTypeDef",
    "DescribeStreamInputWaitTypeDef",
    "DescribeStreamOutputTypeDef",
    "DescribeStreamSummaryInputRequestTypeDef",
    "DescribeStreamSummaryOutputTypeDef",
    "DisableEnhancedMonitoringInputRequestTypeDef",
    "EmptyResponseMetadataTypeDef",
    "EnableEnhancedMonitoringInputRequestTypeDef",
    "EnhancedMetricsTypeDef",
    "EnhancedMonitoringOutputTypeDef",
    "GetRecordsInputRequestTypeDef",
    "GetRecordsOutputTypeDef",
    "GetResourcePolicyInputRequestTypeDef",
    "GetResourcePolicyOutputTypeDef",
    "GetShardIteratorInputRequestTypeDef",
    "GetShardIteratorOutputTypeDef",
    "HashKeyRangeTypeDef",
    "IncreaseStreamRetentionPeriodInputRequestTypeDef",
    "InternalFailureExceptionTypeDef",
    "KMSAccessDeniedExceptionTypeDef",
    "KMSDisabledExceptionTypeDef",
    "KMSInvalidStateExceptionTypeDef",
    "KMSNotFoundExceptionTypeDef",
    "KMSOptInRequiredTypeDef",
    "KMSThrottlingExceptionTypeDef",
    "ListShardsInputPaginateTypeDef",
    "ListShardsInputRequestTypeDef",
    "ListShardsOutputTypeDef",
    "ListStreamConsumersInputPaginateTypeDef",
    "ListStreamConsumersInputRequestTypeDef",
    "ListStreamConsumersOutputTypeDef",
    "ListStreamsInputPaginateTypeDef",
    "ListStreamsInputRequestTypeDef",
    "ListStreamsOutputTypeDef",
    "ListTagsForStreamInputRequestTypeDef",
    "ListTagsForStreamOutputTypeDef",
    "MergeShardsInputRequestTypeDef",
    "PaginatorConfigTypeDef",
    "PutRecordInputRequestTypeDef",
    "PutRecordOutputTypeDef",
    "PutRecordsInputRequestTypeDef",
    "PutRecordsOutputTypeDef",
    "PutRecordsRequestEntryTypeDef",
    "PutRecordsResultEntryTypeDef",
    "PutResourcePolicyInputRequestTypeDef",
    "RecordTypeDef",
    "RegisterStreamConsumerInputRequestTypeDef",
    "RegisterStreamConsumerOutputTypeDef",
    "RemoveTagsFromStreamInputRequestTypeDef",
    "ResourceInUseExceptionTypeDef",
    "ResourceNotFoundExceptionTypeDef",
    "ResponseMetadataTypeDef",
    "SequenceNumberRangeTypeDef",
    "ShardFilterTypeDef",
    "ShardTypeDef",
    "SplitShardInputRequestTypeDef",
    "StartStreamEncryptionInputRequestTypeDef",
    "StartingPositionTypeDef",
    "StopStreamEncryptionInputRequestTypeDef",
    "StreamDescriptionSummaryTypeDef",
    "StreamDescriptionTypeDef",
    "StreamModeDetailsTypeDef",
    "StreamSummaryTypeDef",
    "SubscribeToShardEventStreamTypeDef",
    "SubscribeToShardEventTypeDef",
    "SubscribeToShardInputRequestTypeDef",
    "SubscribeToShardOutputTypeDef",
    "TagTypeDef",
    "TimestampTypeDef",
    "UpdateShardCountInputRequestTypeDef",
    "UpdateShardCountOutputTypeDef",
    "UpdateStreamModeInputRequestTypeDef",
    "WaiterConfigTypeDef",
)


class AddTagsToStreamInputRequestTypeDef(TypedDict):
    Tags: Mapping[str, str]
    StreamName: NotRequired[str]
    StreamARN: NotRequired[str]


BlobTypeDef = Union[str, bytes, IO[Any], StreamingBody]


class HashKeyRangeTypeDef(TypedDict):
    StartingHashKey: str
    EndingHashKey: str


class ConsumerDescriptionTypeDef(TypedDict):
    ConsumerName: str
    ConsumerARN: str
    ConsumerStatus: ConsumerStatusType
    ConsumerCreationTimestamp: datetime
    StreamARN: str


class ConsumerTypeDef(TypedDict):
    ConsumerName: str
    ConsumerARN: str
    ConsumerStatus: ConsumerStatusType
    ConsumerCreationTimestamp: datetime


class StreamModeDetailsTypeDef(TypedDict):
    StreamMode: StreamModeType


class DecreaseStreamRetentionPeriodInputRequestTypeDef(TypedDict):
    RetentionPeriodHours: int
    StreamName: NotRequired[str]
    StreamARN: NotRequired[str]


class DeleteResourcePolicyInputRequestTypeDef(TypedDict):
    ResourceARN: str


class DeleteStreamInputRequestTypeDef(TypedDict):
    StreamName: NotRequired[str]
    EnforceConsumerDeletion: NotRequired[bool]
    StreamARN: NotRequired[str]


class DeregisterStreamConsumerInputRequestTypeDef(TypedDict):
    StreamARN: NotRequired[str]
    ConsumerName: NotRequired[str]
    ConsumerARN: NotRequired[str]


class ResponseMetadataTypeDef(TypedDict):
    RequestId: str
    HTTPStatusCode: int
    HTTPHeaders: Dict[str, str]
    RetryAttempts: int
    HostId: NotRequired[str]


class DescribeStreamConsumerInputRequestTypeDef(TypedDict):
    StreamARN: NotRequired[str]
    ConsumerName: NotRequired[str]
    ConsumerARN: NotRequired[str]


class PaginatorConfigTypeDef(TypedDict):
    MaxItems: NotRequired[int]
    PageSize: NotRequired[int]
    StartingToken: NotRequired[str]


class DescribeStreamInputRequestTypeDef(TypedDict):
    StreamName: NotRequired[str]
    Limit: NotRequired[int]
    ExclusiveStartShardId: NotRequired[str]
    StreamARN: NotRequired[str]


class WaiterConfigTypeDef(TypedDict):
    Delay: NotRequired[int]
    MaxAttempts: NotRequired[int]


class DescribeStreamSummaryInputRequestTypeDef(TypedDict):
    StreamName: NotRequired[str]
    StreamARN: NotRequired[str]


class DisableEnhancedMonitoringInputRequestTypeDef(TypedDict):
    ShardLevelMetrics: Sequence[MetricsNameType]
    StreamName: NotRequired[str]
    StreamARN: NotRequired[str]


class EnableEnhancedMonitoringInputRequestTypeDef(TypedDict):
    ShardLevelMetrics: Sequence[MetricsNameType]
    StreamName: NotRequired[str]
    StreamARN: NotRequired[str]


class EnhancedMetricsTypeDef(TypedDict):
    ShardLevelMetrics: NotRequired[List[MetricsNameType]]


class GetRecordsInputRequestTypeDef(TypedDict):
    ShardIterator: str
    Limit: NotRequired[int]
    StreamARN: NotRequired[str]


class RecordTypeDef(TypedDict):
    SequenceNumber: str
    Data: bytes
    PartitionKey: str
    ApproximateArrivalTimestamp: NotRequired[datetime]
    EncryptionType: NotRequired[EncryptionTypeType]


class GetResourcePolicyInputRequestTypeDef(TypedDict):
    ResourceARN: str


TimestampTypeDef = Union[datetime, str]


class IncreaseStreamRetentionPeriodInputRequestTypeDef(TypedDict):
    RetentionPeriodHours: int
    StreamName: NotRequired[str]
    StreamARN: NotRequired[str]


class InternalFailureExceptionTypeDef(TypedDict):
    message: NotRequired[str]


class KMSAccessDeniedExceptionTypeDef(TypedDict):
    message: NotRequired[str]


class KMSDisabledExceptionTypeDef(TypedDict):
    message: NotRequired[str]


class KMSInvalidStateExceptionTypeDef(TypedDict):
    message: NotRequired[str]


class KMSNotFoundExceptionTypeDef(TypedDict):
    message: NotRequired[str]


class KMSOptInRequiredTypeDef(TypedDict):
    message: NotRequired[str]


class KMSThrottlingExceptionTypeDef(TypedDict):
    message: NotRequired[str]


class ListStreamsInputRequestTypeDef(TypedDict):
    Limit: NotRequired[int]
    ExclusiveStartStreamName: NotRequired[str]
    NextToken: NotRequired[str]


class ListTagsForStreamInputRequestTypeDef(TypedDict):
    StreamName: NotRequired[str]
    ExclusiveStartTagKey: NotRequired[str]
    Limit: NotRequired[int]
    StreamARN: NotRequired[str]


class TagTypeDef(TypedDict):
    Key: str
    Value: NotRequired[str]


class MergeShardsInputRequestTypeDef(TypedDict):
    ShardToMerge: str
    AdjacentShardToMerge: str
    StreamName: NotRequired[str]
    StreamARN: NotRequired[str]


class PutRecordsResultEntryTypeDef(TypedDict):
    SequenceNumber: NotRequired[str]
    ShardId: NotRequired[str]
    ErrorCode: NotRequired[str]
    ErrorMessage: NotRequired[str]


class PutResourcePolicyInputRequestTypeDef(TypedDict):
    ResourceARN: str
    Policy: str


class RegisterStreamConsumerInputRequestTypeDef(TypedDict):
    StreamARN: str
    ConsumerName: str


class RemoveTagsFromStreamInputRequestTypeDef(TypedDict):
    TagKeys: Sequence[str]
    StreamName: NotRequired[str]
    StreamARN: NotRequired[str]


class ResourceInUseExceptionTypeDef(TypedDict):
    message: NotRequired[str]


class ResourceNotFoundExceptionTypeDef(TypedDict):
    message: NotRequired[str]


class SequenceNumberRangeTypeDef(TypedDict):
    StartingSequenceNumber: str
    EndingSequenceNumber: NotRequired[str]


class SplitShardInputRequestTypeDef(TypedDict):
    ShardToSplit: str
    NewStartingHashKey: str
    StreamName: NotRequired[str]
    StreamARN: NotRequired[str]


class StartStreamEncryptionInputRequestTypeDef(TypedDict):
    EncryptionType: EncryptionTypeType
    KeyId: str
    StreamName: NotRequired[str]
    StreamARN: NotRequired[str]


class StopStreamEncryptionInputRequestTypeDef(TypedDict):
    EncryptionType: EncryptionTypeType
    KeyId: str
    StreamName: NotRequired[str]
    StreamARN: NotRequired[str]


class UpdateShardCountInputRequestTypeDef(TypedDict):
    TargetShardCount: int
    ScalingType: Literal["UNIFORM_SCALING"]
    StreamName: NotRequired[str]
    StreamARN: NotRequired[str]


class PutRecordInputRequestTypeDef(TypedDict):
    Data: BlobTypeDef
    PartitionKey: str
    StreamName: NotRequired[str]
    ExplicitHashKey: NotRequired[str]
    SequenceNumberForOrdering: NotRequired[str]
    StreamARN: NotRequired[str]


class PutRecordsRequestEntryTypeDef(TypedDict):
    Data: BlobTypeDef
    PartitionKey: str
    ExplicitHashKey: NotRequired[str]


class ChildShardTypeDef(TypedDict):
    ShardId: str
    ParentShards: List[str]
    HashKeyRange: HashKeyRangeTypeDef


class CreateStreamInputRequestTypeDef(TypedDict):
    StreamName: str
    ShardCount: NotRequired[int]
    StreamModeDetails: NotRequired[StreamModeDetailsTypeDef]
    Tags: NotRequired[Mapping[str, str]]


class StreamSummaryTypeDef(TypedDict):
    StreamName: str
    StreamARN: str
    StreamStatus: StreamStatusType
    StreamModeDetails: NotRequired[StreamModeDetailsTypeDef]
    StreamCreationTimestamp: NotRequired[datetime]


class UpdateStreamModeInputRequestTypeDef(TypedDict):
    StreamARN: str
    StreamModeDetails: StreamModeDetailsTypeDef


class DescribeLimitsOutputTypeDef(TypedDict):
    ShardLimit: int
    OpenShardCount: int
    OnDemandStreamCount: int
    OnDemandStreamCountLimit: int
    ResponseMetadata: ResponseMetadataTypeDef


class DescribeStreamConsumerOutputTypeDef(TypedDict):
    ConsumerDescription: ConsumerDescriptionTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class EmptyResponseMetadataTypeDef(TypedDict):
    ResponseMetadata: ResponseMetadataTypeDef


class EnhancedMonitoringOutputTypeDef(TypedDict):
    StreamName: str
    CurrentShardLevelMetrics: List[MetricsNameType]
    DesiredShardLevelMetrics: List[MetricsNameType]
    StreamARN: str
    ResponseMetadata: ResponseMetadataTypeDef


class GetResourcePolicyOutputTypeDef(TypedDict):
    Policy: str
    ResponseMetadata: ResponseMetadataTypeDef


class GetShardIteratorOutputTypeDef(TypedDict):
    ShardIterator: str
    ResponseMetadata: ResponseMetadataTypeDef


class ListStreamConsumersOutputTypeDef(TypedDict):
    Consumers: List[ConsumerTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef
    NextToken: NotRequired[str]


class PutRecordOutputTypeDef(TypedDict):
    ShardId: str
    SequenceNumber: str
    EncryptionType: EncryptionTypeType
    ResponseMetadata: ResponseMetadataTypeDef


class RegisterStreamConsumerOutputTypeDef(TypedDict):
    Consumer: ConsumerTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class UpdateShardCountOutputTypeDef(TypedDict):
    StreamName: str
    CurrentShardCount: int
    TargetShardCount: int
    StreamARN: str
    ResponseMetadata: ResponseMetadataTypeDef


class DescribeStreamInputPaginateTypeDef(TypedDict):
    StreamName: NotRequired[str]
    StreamARN: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class ListStreamsInputPaginateTypeDef(TypedDict):
    ExclusiveStartStreamName: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class DescribeStreamInputWaitTypeDef(TypedDict):
    StreamName: NotRequired[str]
    Limit: NotRequired[int]
    ExclusiveStartShardId: NotRequired[str]
    StreamARN: NotRequired[str]
    WaiterConfig: NotRequired[WaiterConfigTypeDef]


class StreamDescriptionSummaryTypeDef(TypedDict):
    StreamName: str
    StreamARN: str
    StreamStatus: StreamStatusType
    RetentionPeriodHours: int
    StreamCreationTimestamp: datetime
    EnhancedMonitoring: List[EnhancedMetricsTypeDef]
    OpenShardCount: int
    StreamModeDetails: NotRequired[StreamModeDetailsTypeDef]
    EncryptionType: NotRequired[EncryptionTypeType]
    KeyId: NotRequired[str]
    ConsumerCount: NotRequired[int]


class GetShardIteratorInputRequestTypeDef(TypedDict):
    ShardId: str
    ShardIteratorType: ShardIteratorTypeType
    StreamName: NotRequired[str]
    StartingSequenceNumber: NotRequired[str]
    Timestamp: NotRequired[TimestampTypeDef]
    StreamARN: NotRequired[str]


class ListStreamConsumersInputPaginateTypeDef(TypedDict):
    StreamARN: str
    StreamCreationTimestamp: NotRequired[TimestampTypeDef]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class ListStreamConsumersInputRequestTypeDef(TypedDict):
    StreamARN: str
    NextToken: NotRequired[str]
    MaxResults: NotRequired[int]
    StreamCreationTimestamp: NotRequired[TimestampTypeDef]


ShardFilterTypeDef = TypedDict(
    "ShardFilterTypeDef",
    {
        "Type": ShardFilterTypeType,
        "ShardId": NotRequired[str],
        "Timestamp": NotRequired[TimestampTypeDef],
    },
)
StartingPositionTypeDef = TypedDict(
    "StartingPositionTypeDef",
    {
        "Type": ShardIteratorTypeType,
        "SequenceNumber": NotRequired[str],
        "Timestamp": NotRequired[TimestampTypeDef],
    },
)


class ListTagsForStreamOutputTypeDef(TypedDict):
    Tags: List[TagTypeDef]
    HasMoreTags: bool
    ResponseMetadata: ResponseMetadataTypeDef


class PutRecordsOutputTypeDef(TypedDict):
    FailedRecordCount: int
    Records: List[PutRecordsResultEntryTypeDef]
    EncryptionType: EncryptionTypeType
    ResponseMetadata: ResponseMetadataTypeDef


class ShardTypeDef(TypedDict):
    ShardId: str
    HashKeyRange: HashKeyRangeTypeDef
    SequenceNumberRange: SequenceNumberRangeTypeDef
    ParentShardId: NotRequired[str]
    AdjacentParentShardId: NotRequired[str]


class PutRecordsInputRequestTypeDef(TypedDict):
    Records: Sequence[PutRecordsRequestEntryTypeDef]
    StreamName: NotRequired[str]
    StreamARN: NotRequired[str]


class GetRecordsOutputTypeDef(TypedDict):
    Records: List[RecordTypeDef]
    NextShardIterator: str
    MillisBehindLatest: int
    ChildShards: List[ChildShardTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class SubscribeToShardEventTypeDef(TypedDict):
    Records: List[RecordTypeDef]
    ContinuationSequenceNumber: str
    MillisBehindLatest: int
    ChildShards: NotRequired[List[ChildShardTypeDef]]


class ListStreamsOutputTypeDef(TypedDict):
    StreamNames: List[str]
    HasMoreStreams: bool
    StreamSummaries: List[StreamSummaryTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef
    NextToken: NotRequired[str]


class DescribeStreamSummaryOutputTypeDef(TypedDict):
    StreamDescriptionSummary: StreamDescriptionSummaryTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class ListShardsInputPaginateTypeDef(TypedDict):
    StreamName: NotRequired[str]
    ExclusiveStartShardId: NotRequired[str]
    StreamCreationTimestamp: NotRequired[TimestampTypeDef]
    ShardFilter: NotRequired[ShardFilterTypeDef]
    StreamARN: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class ListShardsInputRequestTypeDef(TypedDict):
    StreamName: NotRequired[str]
    NextToken: NotRequired[str]
    ExclusiveStartShardId: NotRequired[str]
    MaxResults: NotRequired[int]
    StreamCreationTimestamp: NotRequired[TimestampTypeDef]
    ShardFilter: NotRequired[ShardFilterTypeDef]
    StreamARN: NotRequired[str]


class SubscribeToShardInputRequestTypeDef(TypedDict):
    ConsumerARN: str
    ShardId: str
    StartingPosition: StartingPositionTypeDef


class ListShardsOutputTypeDef(TypedDict):
    Shards: List[ShardTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef
    NextToken: NotRequired[str]


class StreamDescriptionTypeDef(TypedDict):
    StreamName: str
    StreamARN: str
    StreamStatus: StreamStatusType
    Shards: List[ShardTypeDef]
    HasMoreShards: bool
    RetentionPeriodHours: int
    StreamCreationTimestamp: datetime
    EnhancedMonitoring: List[EnhancedMetricsTypeDef]
    StreamModeDetails: NotRequired[StreamModeDetailsTypeDef]
    EncryptionType: NotRequired[EncryptionTypeType]
    KeyId: NotRequired[str]


class SubscribeToShardEventStreamTypeDef(TypedDict):
    SubscribeToShardEvent: SubscribeToShardEventTypeDef
    ResourceNotFoundException: NotRequired[ResourceNotFoundExceptionTypeDef]
    ResourceInUseException: NotRequired[ResourceInUseExceptionTypeDef]
    KMSDisabledException: NotRequired[KMSDisabledExceptionTypeDef]
    KMSInvalidStateException: NotRequired[KMSInvalidStateExceptionTypeDef]
    KMSAccessDeniedException: NotRequired[KMSAccessDeniedExceptionTypeDef]
    KMSNotFoundException: NotRequired[KMSNotFoundExceptionTypeDef]
    KMSOptInRequired: NotRequired[KMSOptInRequiredTypeDef]
    KMSThrottlingException: NotRequired[KMSThrottlingExceptionTypeDef]
    InternalFailureException: NotRequired[InternalFailureExceptionTypeDef]


class DescribeStreamOutputTypeDef(TypedDict):
    StreamDescription: StreamDescriptionTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class SubscribeToShardOutputTypeDef(TypedDict):
    EventStream: EventStream[SubscribeToShardEventStreamTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef
