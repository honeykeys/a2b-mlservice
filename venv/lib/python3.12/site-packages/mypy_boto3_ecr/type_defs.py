"""
Type annotations for ecr service type definitions.

[Documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/type_defs/)

Usage::

    ```python
    from mypy_boto3_ecr.type_defs import AttributeTypeDef

    data: AttributeTypeDef = ...
    ```

Copyright 2025 Vlad Emelianov
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import IO, Any, Union

from botocore.response import StreamingBody

from .literals import (
    EncryptionTypeType,
    FindingSeverityType,
    ImageFailureCodeType,
    ImageTagMutabilityType,
    LayerAvailabilityType,
    LayerFailureCodeType,
    LifecyclePolicyPreviewStatusType,
    RCTAppliedForType,
    ReplicationStatusType,
    ScanFrequencyType,
    ScanStatusType,
    ScanTypeType,
    TagStatusType,
    UpstreamRegistryType,
)

if sys.version_info >= (3, 9):
    from builtins import dict as Dict
    from builtins import list as List
    from collections.abc import Sequence
else:
    from typing import Dict, List, Sequence
if sys.version_info >= (3, 12):
    from typing import Literal, NotRequired, TypedDict
else:
    from typing_extensions import Literal, NotRequired, TypedDict


__all__ = (
    "AttributeTypeDef",
    "AuthorizationDataTypeDef",
    "AwsEcrContainerImageDetailsTypeDef",
    "BatchCheckLayerAvailabilityRequestRequestTypeDef",
    "BatchCheckLayerAvailabilityResponseTypeDef",
    "BatchDeleteImageRequestRequestTypeDef",
    "BatchDeleteImageResponseTypeDef",
    "BatchGetImageRequestRequestTypeDef",
    "BatchGetImageResponseTypeDef",
    "BatchGetRepositoryScanningConfigurationRequestRequestTypeDef",
    "BatchGetRepositoryScanningConfigurationResponseTypeDef",
    "BlobTypeDef",
    "CompleteLayerUploadRequestRequestTypeDef",
    "CompleteLayerUploadResponseTypeDef",
    "CreatePullThroughCacheRuleRequestRequestTypeDef",
    "CreatePullThroughCacheRuleResponseTypeDef",
    "CreateRepositoryCreationTemplateRequestRequestTypeDef",
    "CreateRepositoryCreationTemplateResponseTypeDef",
    "CreateRepositoryRequestRequestTypeDef",
    "CreateRepositoryResponseTypeDef",
    "CvssScoreAdjustmentTypeDef",
    "CvssScoreDetailsTypeDef",
    "CvssScoreTypeDef",
    "DeleteLifecyclePolicyRequestRequestTypeDef",
    "DeleteLifecyclePolicyResponseTypeDef",
    "DeletePullThroughCacheRuleRequestRequestTypeDef",
    "DeletePullThroughCacheRuleResponseTypeDef",
    "DeleteRegistryPolicyResponseTypeDef",
    "DeleteRepositoryCreationTemplateRequestRequestTypeDef",
    "DeleteRepositoryCreationTemplateResponseTypeDef",
    "DeleteRepositoryPolicyRequestRequestTypeDef",
    "DeleteRepositoryPolicyResponseTypeDef",
    "DeleteRepositoryRequestRequestTypeDef",
    "DeleteRepositoryResponseTypeDef",
    "DescribeImageReplicationStatusRequestRequestTypeDef",
    "DescribeImageReplicationStatusResponseTypeDef",
    "DescribeImageScanFindingsRequestPaginateTypeDef",
    "DescribeImageScanFindingsRequestRequestTypeDef",
    "DescribeImageScanFindingsRequestWaitTypeDef",
    "DescribeImageScanFindingsResponseTypeDef",
    "DescribeImagesFilterTypeDef",
    "DescribeImagesRequestPaginateTypeDef",
    "DescribeImagesRequestRequestTypeDef",
    "DescribeImagesResponseTypeDef",
    "DescribePullThroughCacheRulesRequestPaginateTypeDef",
    "DescribePullThroughCacheRulesRequestRequestTypeDef",
    "DescribePullThroughCacheRulesResponseTypeDef",
    "DescribeRegistryResponseTypeDef",
    "DescribeRepositoriesRequestPaginateTypeDef",
    "DescribeRepositoriesRequestRequestTypeDef",
    "DescribeRepositoriesResponseTypeDef",
    "DescribeRepositoryCreationTemplatesRequestPaginateTypeDef",
    "DescribeRepositoryCreationTemplatesRequestRequestTypeDef",
    "DescribeRepositoryCreationTemplatesResponseTypeDef",
    "EncryptionConfigurationForRepositoryCreationTemplateTypeDef",
    "EncryptionConfigurationTypeDef",
    "EnhancedImageScanFindingTypeDef",
    "GetAccountSettingRequestRequestTypeDef",
    "GetAccountSettingResponseTypeDef",
    "GetAuthorizationTokenRequestRequestTypeDef",
    "GetAuthorizationTokenResponseTypeDef",
    "GetDownloadUrlForLayerRequestRequestTypeDef",
    "GetDownloadUrlForLayerResponseTypeDef",
    "GetLifecyclePolicyPreviewRequestPaginateTypeDef",
    "GetLifecyclePolicyPreviewRequestRequestTypeDef",
    "GetLifecyclePolicyPreviewRequestWaitTypeDef",
    "GetLifecyclePolicyPreviewResponseTypeDef",
    "GetLifecyclePolicyRequestRequestTypeDef",
    "GetLifecyclePolicyResponseTypeDef",
    "GetRegistryPolicyResponseTypeDef",
    "GetRegistryScanningConfigurationResponseTypeDef",
    "GetRepositoryPolicyRequestRequestTypeDef",
    "GetRepositoryPolicyResponseTypeDef",
    "ImageDetailTypeDef",
    "ImageFailureTypeDef",
    "ImageIdentifierTypeDef",
    "ImageReplicationStatusTypeDef",
    "ImageScanFindingTypeDef",
    "ImageScanFindingsSummaryTypeDef",
    "ImageScanFindingsTypeDef",
    "ImageScanStatusTypeDef",
    "ImageScanningConfigurationTypeDef",
    "ImageTypeDef",
    "InitiateLayerUploadRequestRequestTypeDef",
    "InitiateLayerUploadResponseTypeDef",
    "LayerFailureTypeDef",
    "LayerTypeDef",
    "LifecyclePolicyPreviewFilterTypeDef",
    "LifecyclePolicyPreviewResultTypeDef",
    "LifecyclePolicyPreviewSummaryTypeDef",
    "LifecyclePolicyRuleActionTypeDef",
    "ListImagesFilterTypeDef",
    "ListImagesRequestPaginateTypeDef",
    "ListImagesRequestRequestTypeDef",
    "ListImagesResponseTypeDef",
    "ListTagsForResourceRequestRequestTypeDef",
    "ListTagsForResourceResponseTypeDef",
    "PackageVulnerabilityDetailsTypeDef",
    "PaginatorConfigTypeDef",
    "PullThroughCacheRuleTypeDef",
    "PutAccountSettingRequestRequestTypeDef",
    "PutAccountSettingResponseTypeDef",
    "PutImageRequestRequestTypeDef",
    "PutImageResponseTypeDef",
    "PutImageScanningConfigurationRequestRequestTypeDef",
    "PutImageScanningConfigurationResponseTypeDef",
    "PutImageTagMutabilityRequestRequestTypeDef",
    "PutImageTagMutabilityResponseTypeDef",
    "PutLifecyclePolicyRequestRequestTypeDef",
    "PutLifecyclePolicyResponseTypeDef",
    "PutRegistryPolicyRequestRequestTypeDef",
    "PutRegistryPolicyResponseTypeDef",
    "PutRegistryScanningConfigurationRequestRequestTypeDef",
    "PutRegistryScanningConfigurationResponseTypeDef",
    "PutReplicationConfigurationRequestRequestTypeDef",
    "PutReplicationConfigurationResponseTypeDef",
    "RecommendationTypeDef",
    "RegistryScanningConfigurationTypeDef",
    "RegistryScanningRuleOutputTypeDef",
    "RegistryScanningRuleTypeDef",
    "RegistryScanningRuleUnionTypeDef",
    "RemediationTypeDef",
    "ReplicationConfigurationOutputTypeDef",
    "ReplicationConfigurationTypeDef",
    "ReplicationDestinationTypeDef",
    "ReplicationRuleOutputTypeDef",
    "ReplicationRuleTypeDef",
    "ReplicationRuleUnionTypeDef",
    "RepositoryCreationTemplateTypeDef",
    "RepositoryFilterTypeDef",
    "RepositoryScanningConfigurationFailureTypeDef",
    "RepositoryScanningConfigurationTypeDef",
    "RepositoryTypeDef",
    "ResourceDetailsTypeDef",
    "ResourceTypeDef",
    "ResponseMetadataTypeDef",
    "ScanningRepositoryFilterTypeDef",
    "ScoreDetailsTypeDef",
    "SetRepositoryPolicyRequestRequestTypeDef",
    "SetRepositoryPolicyResponseTypeDef",
    "StartImageScanRequestRequestTypeDef",
    "StartImageScanResponseTypeDef",
    "StartLifecyclePolicyPreviewRequestRequestTypeDef",
    "StartLifecyclePolicyPreviewResponseTypeDef",
    "TagResourceRequestRequestTypeDef",
    "TagTypeDef",
    "UntagResourceRequestRequestTypeDef",
    "UpdatePullThroughCacheRuleRequestRequestTypeDef",
    "UpdatePullThroughCacheRuleResponseTypeDef",
    "UpdateRepositoryCreationTemplateRequestRequestTypeDef",
    "UpdateRepositoryCreationTemplateResponseTypeDef",
    "UploadLayerPartRequestRequestTypeDef",
    "UploadLayerPartResponseTypeDef",
    "ValidatePullThroughCacheRuleRequestRequestTypeDef",
    "ValidatePullThroughCacheRuleResponseTypeDef",
    "VulnerablePackageTypeDef",
    "WaiterConfigTypeDef",
)


class AttributeTypeDef(TypedDict):
    key: str
    value: NotRequired[str]


class AuthorizationDataTypeDef(TypedDict):
    authorizationToken: NotRequired[str]
    expiresAt: NotRequired[datetime]
    proxyEndpoint: NotRequired[str]


class AwsEcrContainerImageDetailsTypeDef(TypedDict):
    architecture: NotRequired[str]
    author: NotRequired[str]
    imageHash: NotRequired[str]
    imageTags: NotRequired[List[str]]
    platform: NotRequired[str]
    pushedAt: NotRequired[datetime]
    registry: NotRequired[str]
    repositoryName: NotRequired[str]


class BatchCheckLayerAvailabilityRequestRequestTypeDef(TypedDict):
    repositoryName: str
    layerDigests: Sequence[str]
    registryId: NotRequired[str]


class LayerFailureTypeDef(TypedDict):
    layerDigest: NotRequired[str]
    failureCode: NotRequired[LayerFailureCodeType]
    failureReason: NotRequired[str]


class LayerTypeDef(TypedDict):
    layerDigest: NotRequired[str]
    layerAvailability: NotRequired[LayerAvailabilityType]
    layerSize: NotRequired[int]
    mediaType: NotRequired[str]


class ResponseMetadataTypeDef(TypedDict):
    RequestId: str
    HTTPStatusCode: int
    HTTPHeaders: Dict[str, str]
    RetryAttempts: int
    HostId: NotRequired[str]


class ImageIdentifierTypeDef(TypedDict):
    imageDigest: NotRequired[str]
    imageTag: NotRequired[str]


class BatchGetRepositoryScanningConfigurationRequestRequestTypeDef(TypedDict):
    repositoryNames: Sequence[str]


class RepositoryScanningConfigurationFailureTypeDef(TypedDict):
    repositoryName: NotRequired[str]
    failureCode: NotRequired[Literal["REPOSITORY_NOT_FOUND"]]
    failureReason: NotRequired[str]


BlobTypeDef = Union[str, bytes, IO[Any], StreamingBody]


class CompleteLayerUploadRequestRequestTypeDef(TypedDict):
    repositoryName: str
    uploadId: str
    layerDigests: Sequence[str]
    registryId: NotRequired[str]


class CreatePullThroughCacheRuleRequestRequestTypeDef(TypedDict):
    ecrRepositoryPrefix: str
    upstreamRegistryUrl: str
    registryId: NotRequired[str]
    upstreamRegistry: NotRequired[UpstreamRegistryType]
    credentialArn: NotRequired[str]


class EncryptionConfigurationForRepositoryCreationTemplateTypeDef(TypedDict):
    encryptionType: EncryptionTypeType
    kmsKey: NotRequired[str]


class TagTypeDef(TypedDict):
    Key: str
    Value: str


class EncryptionConfigurationTypeDef(TypedDict):
    encryptionType: EncryptionTypeType
    kmsKey: NotRequired[str]


class ImageScanningConfigurationTypeDef(TypedDict):
    scanOnPush: NotRequired[bool]


class CvssScoreAdjustmentTypeDef(TypedDict):
    metric: NotRequired[str]
    reason: NotRequired[str]


class CvssScoreTypeDef(TypedDict):
    baseScore: NotRequired[float]
    scoringVector: NotRequired[str]
    source: NotRequired[str]
    version: NotRequired[str]


class DeleteLifecyclePolicyRequestRequestTypeDef(TypedDict):
    repositoryName: str
    registryId: NotRequired[str]


class DeletePullThroughCacheRuleRequestRequestTypeDef(TypedDict):
    ecrRepositoryPrefix: str
    registryId: NotRequired[str]


class DeleteRepositoryCreationTemplateRequestRequestTypeDef(TypedDict):
    prefix: str


class DeleteRepositoryPolicyRequestRequestTypeDef(TypedDict):
    repositoryName: str
    registryId: NotRequired[str]


class DeleteRepositoryRequestRequestTypeDef(TypedDict):
    repositoryName: str
    registryId: NotRequired[str]
    force: NotRequired[bool]


class ImageReplicationStatusTypeDef(TypedDict):
    region: NotRequired[str]
    registryId: NotRequired[str]
    status: NotRequired[ReplicationStatusType]
    failureCode: NotRequired[str]


class PaginatorConfigTypeDef(TypedDict):
    MaxItems: NotRequired[int]
    PageSize: NotRequired[int]
    StartingToken: NotRequired[str]


class WaiterConfigTypeDef(TypedDict):
    Delay: NotRequired[int]
    MaxAttempts: NotRequired[int]


class ImageScanStatusTypeDef(TypedDict):
    status: NotRequired[ScanStatusType]
    description: NotRequired[str]


class DescribeImagesFilterTypeDef(TypedDict):
    tagStatus: NotRequired[TagStatusType]


class DescribePullThroughCacheRulesRequestRequestTypeDef(TypedDict):
    registryId: NotRequired[str]
    ecrRepositoryPrefixes: NotRequired[Sequence[str]]
    nextToken: NotRequired[str]
    maxResults: NotRequired[int]


class PullThroughCacheRuleTypeDef(TypedDict):
    ecrRepositoryPrefix: NotRequired[str]
    upstreamRegistryUrl: NotRequired[str]
    createdAt: NotRequired[datetime]
    registryId: NotRequired[str]
    credentialArn: NotRequired[str]
    upstreamRegistry: NotRequired[UpstreamRegistryType]
    updatedAt: NotRequired[datetime]


class DescribeRepositoriesRequestRequestTypeDef(TypedDict):
    registryId: NotRequired[str]
    repositoryNames: NotRequired[Sequence[str]]
    nextToken: NotRequired[str]
    maxResults: NotRequired[int]


class DescribeRepositoryCreationTemplatesRequestRequestTypeDef(TypedDict):
    prefixes: NotRequired[Sequence[str]]
    nextToken: NotRequired[str]
    maxResults: NotRequired[int]


class GetAccountSettingRequestRequestTypeDef(TypedDict):
    name: str


class GetAuthorizationTokenRequestRequestTypeDef(TypedDict):
    registryIds: NotRequired[Sequence[str]]


class GetDownloadUrlForLayerRequestRequestTypeDef(TypedDict):
    repositoryName: str
    layerDigest: str
    registryId: NotRequired[str]


class LifecyclePolicyPreviewFilterTypeDef(TypedDict):
    tagStatus: NotRequired[TagStatusType]


class LifecyclePolicyPreviewSummaryTypeDef(TypedDict):
    expiringImageTotalCount: NotRequired[int]


class GetLifecyclePolicyRequestRequestTypeDef(TypedDict):
    repositoryName: str
    registryId: NotRequired[str]


class GetRepositoryPolicyRequestRequestTypeDef(TypedDict):
    repositoryName: str
    registryId: NotRequired[str]


class ImageScanFindingsSummaryTypeDef(TypedDict):
    imageScanCompletedAt: NotRequired[datetime]
    vulnerabilitySourceUpdatedAt: NotRequired[datetime]
    findingSeverityCounts: NotRequired[Dict[FindingSeverityType, int]]


class InitiateLayerUploadRequestRequestTypeDef(TypedDict):
    repositoryName: str
    registryId: NotRequired[str]


LifecyclePolicyRuleActionTypeDef = TypedDict(
    "LifecyclePolicyRuleActionTypeDef",
    {
        "type": NotRequired[Literal["EXPIRE"]],
    },
)


class ListImagesFilterTypeDef(TypedDict):
    tagStatus: NotRequired[TagStatusType]


class ListTagsForResourceRequestRequestTypeDef(TypedDict):
    resourceArn: str


class VulnerablePackageTypeDef(TypedDict):
    arch: NotRequired[str]
    epoch: NotRequired[int]
    filePath: NotRequired[str]
    name: NotRequired[str]
    packageManager: NotRequired[str]
    release: NotRequired[str]
    sourceLayerHash: NotRequired[str]
    version: NotRequired[str]
    fixedInVersion: NotRequired[str]


class PutAccountSettingRequestRequestTypeDef(TypedDict):
    name: str
    value: str


class PutImageRequestRequestTypeDef(TypedDict):
    repositoryName: str
    imageManifest: str
    registryId: NotRequired[str]
    imageManifestMediaType: NotRequired[str]
    imageTag: NotRequired[str]
    imageDigest: NotRequired[str]


class PutImageTagMutabilityRequestRequestTypeDef(TypedDict):
    repositoryName: str
    imageTagMutability: ImageTagMutabilityType
    registryId: NotRequired[str]


class PutLifecyclePolicyRequestRequestTypeDef(TypedDict):
    repositoryName: str
    lifecyclePolicyText: str
    registryId: NotRequired[str]


class PutRegistryPolicyRequestRequestTypeDef(TypedDict):
    policyText: str


class RecommendationTypeDef(TypedDict):
    url: NotRequired[str]
    text: NotRequired[str]


ScanningRepositoryFilterTypeDef = TypedDict(
    "ScanningRepositoryFilterTypeDef",
    {
        "filter": str,
        "filterType": Literal["WILDCARD"],
    },
)


class ReplicationDestinationTypeDef(TypedDict):
    region: str
    registryId: str


RepositoryFilterTypeDef = TypedDict(
    "RepositoryFilterTypeDef",
    {
        "filter": str,
        "filterType": Literal["PREFIX_MATCH"],
    },
)


class SetRepositoryPolicyRequestRequestTypeDef(TypedDict):
    repositoryName: str
    policyText: str
    registryId: NotRequired[str]
    force: NotRequired[bool]


class StartLifecyclePolicyPreviewRequestRequestTypeDef(TypedDict):
    repositoryName: str
    registryId: NotRequired[str]
    lifecyclePolicyText: NotRequired[str]


class UntagResourceRequestRequestTypeDef(TypedDict):
    resourceArn: str
    tagKeys: Sequence[str]


class UpdatePullThroughCacheRuleRequestRequestTypeDef(TypedDict):
    ecrRepositoryPrefix: str
    credentialArn: str
    registryId: NotRequired[str]


class ValidatePullThroughCacheRuleRequestRequestTypeDef(TypedDict):
    ecrRepositoryPrefix: str
    registryId: NotRequired[str]


class ImageScanFindingTypeDef(TypedDict):
    name: NotRequired[str]
    description: NotRequired[str]
    uri: NotRequired[str]
    severity: NotRequired[FindingSeverityType]
    attributes: NotRequired[List[AttributeTypeDef]]


class ResourceDetailsTypeDef(TypedDict):
    awsEcrContainerImage: NotRequired[AwsEcrContainerImageDetailsTypeDef]


class BatchCheckLayerAvailabilityResponseTypeDef(TypedDict):
    layers: List[LayerTypeDef]
    failures: List[LayerFailureTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class CompleteLayerUploadResponseTypeDef(TypedDict):
    registryId: str
    repositoryName: str
    uploadId: str
    layerDigest: str
    ResponseMetadata: ResponseMetadataTypeDef


class CreatePullThroughCacheRuleResponseTypeDef(TypedDict):
    ecrRepositoryPrefix: str
    upstreamRegistryUrl: str
    createdAt: datetime
    registryId: str
    upstreamRegistry: UpstreamRegistryType
    credentialArn: str
    ResponseMetadata: ResponseMetadataTypeDef


class DeleteLifecyclePolicyResponseTypeDef(TypedDict):
    registryId: str
    repositoryName: str
    lifecyclePolicyText: str
    lastEvaluatedAt: datetime
    ResponseMetadata: ResponseMetadataTypeDef


class DeletePullThroughCacheRuleResponseTypeDef(TypedDict):
    ecrRepositoryPrefix: str
    upstreamRegistryUrl: str
    createdAt: datetime
    registryId: str
    credentialArn: str
    ResponseMetadata: ResponseMetadataTypeDef


class DeleteRegistryPolicyResponseTypeDef(TypedDict):
    registryId: str
    policyText: str
    ResponseMetadata: ResponseMetadataTypeDef


class DeleteRepositoryPolicyResponseTypeDef(TypedDict):
    registryId: str
    repositoryName: str
    policyText: str
    ResponseMetadata: ResponseMetadataTypeDef


class GetAccountSettingResponseTypeDef(TypedDict):
    name: str
    value: str
    ResponseMetadata: ResponseMetadataTypeDef


class GetAuthorizationTokenResponseTypeDef(TypedDict):
    authorizationData: List[AuthorizationDataTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class GetDownloadUrlForLayerResponseTypeDef(TypedDict):
    downloadUrl: str
    layerDigest: str
    ResponseMetadata: ResponseMetadataTypeDef


class GetLifecyclePolicyResponseTypeDef(TypedDict):
    registryId: str
    repositoryName: str
    lifecyclePolicyText: str
    lastEvaluatedAt: datetime
    ResponseMetadata: ResponseMetadataTypeDef


class GetRegistryPolicyResponseTypeDef(TypedDict):
    registryId: str
    policyText: str
    ResponseMetadata: ResponseMetadataTypeDef


class GetRepositoryPolicyResponseTypeDef(TypedDict):
    registryId: str
    repositoryName: str
    policyText: str
    ResponseMetadata: ResponseMetadataTypeDef


class InitiateLayerUploadResponseTypeDef(TypedDict):
    uploadId: str
    partSize: int
    ResponseMetadata: ResponseMetadataTypeDef


class PutAccountSettingResponseTypeDef(TypedDict):
    name: str
    value: str
    ResponseMetadata: ResponseMetadataTypeDef


class PutImageTagMutabilityResponseTypeDef(TypedDict):
    registryId: str
    repositoryName: str
    imageTagMutability: ImageTagMutabilityType
    ResponseMetadata: ResponseMetadataTypeDef


class PutLifecyclePolicyResponseTypeDef(TypedDict):
    registryId: str
    repositoryName: str
    lifecyclePolicyText: str
    ResponseMetadata: ResponseMetadataTypeDef


class PutRegistryPolicyResponseTypeDef(TypedDict):
    registryId: str
    policyText: str
    ResponseMetadata: ResponseMetadataTypeDef


class SetRepositoryPolicyResponseTypeDef(TypedDict):
    registryId: str
    repositoryName: str
    policyText: str
    ResponseMetadata: ResponseMetadataTypeDef


class StartLifecyclePolicyPreviewResponseTypeDef(TypedDict):
    registryId: str
    repositoryName: str
    lifecyclePolicyText: str
    status: LifecyclePolicyPreviewStatusType
    ResponseMetadata: ResponseMetadataTypeDef


class UpdatePullThroughCacheRuleResponseTypeDef(TypedDict):
    ecrRepositoryPrefix: str
    registryId: str
    updatedAt: datetime
    credentialArn: str
    ResponseMetadata: ResponseMetadataTypeDef


class UploadLayerPartResponseTypeDef(TypedDict):
    registryId: str
    repositoryName: str
    uploadId: str
    lastByteReceived: int
    ResponseMetadata: ResponseMetadataTypeDef


class ValidatePullThroughCacheRuleResponseTypeDef(TypedDict):
    ecrRepositoryPrefix: str
    registryId: str
    upstreamRegistryUrl: str
    credentialArn: str
    isValid: bool
    failure: str
    ResponseMetadata: ResponseMetadataTypeDef


class BatchDeleteImageRequestRequestTypeDef(TypedDict):
    repositoryName: str
    imageIds: Sequence[ImageIdentifierTypeDef]
    registryId: NotRequired[str]


class BatchGetImageRequestRequestTypeDef(TypedDict):
    repositoryName: str
    imageIds: Sequence[ImageIdentifierTypeDef]
    registryId: NotRequired[str]
    acceptedMediaTypes: NotRequired[Sequence[str]]


class DescribeImageReplicationStatusRequestRequestTypeDef(TypedDict):
    repositoryName: str
    imageId: ImageIdentifierTypeDef
    registryId: NotRequired[str]


class DescribeImageScanFindingsRequestRequestTypeDef(TypedDict):
    repositoryName: str
    imageId: ImageIdentifierTypeDef
    registryId: NotRequired[str]
    nextToken: NotRequired[str]
    maxResults: NotRequired[int]


class ImageFailureTypeDef(TypedDict):
    imageId: NotRequired[ImageIdentifierTypeDef]
    failureCode: NotRequired[ImageFailureCodeType]
    failureReason: NotRequired[str]


class ImageTypeDef(TypedDict):
    registryId: NotRequired[str]
    repositoryName: NotRequired[str]
    imageId: NotRequired[ImageIdentifierTypeDef]
    imageManifest: NotRequired[str]
    imageManifestMediaType: NotRequired[str]


class ListImagesResponseTypeDef(TypedDict):
    imageIds: List[ImageIdentifierTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef
    nextToken: NotRequired[str]


class StartImageScanRequestRequestTypeDef(TypedDict):
    repositoryName: str
    imageId: ImageIdentifierTypeDef
    registryId: NotRequired[str]


class UploadLayerPartRequestRequestTypeDef(TypedDict):
    repositoryName: str
    uploadId: str
    partFirstByte: int
    partLastByte: int
    layerPartBlob: BlobTypeDef
    registryId: NotRequired[str]


class CreateRepositoryCreationTemplateRequestRequestTypeDef(TypedDict):
    prefix: str
    appliedFor: Sequence[RCTAppliedForType]
    description: NotRequired[str]
    encryptionConfiguration: NotRequired[
        EncryptionConfigurationForRepositoryCreationTemplateTypeDef
    ]
    resourceTags: NotRequired[Sequence[TagTypeDef]]
    imageTagMutability: NotRequired[ImageTagMutabilityType]
    repositoryPolicy: NotRequired[str]
    lifecyclePolicy: NotRequired[str]
    customRoleArn: NotRequired[str]


class ListTagsForResourceResponseTypeDef(TypedDict):
    tags: List[TagTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class RepositoryCreationTemplateTypeDef(TypedDict):
    prefix: NotRequired[str]
    description: NotRequired[str]
    encryptionConfiguration: NotRequired[
        EncryptionConfigurationForRepositoryCreationTemplateTypeDef
    ]
    resourceTags: NotRequired[List[TagTypeDef]]
    imageTagMutability: NotRequired[ImageTagMutabilityType]
    repositoryPolicy: NotRequired[str]
    lifecyclePolicy: NotRequired[str]
    appliedFor: NotRequired[List[RCTAppliedForType]]
    customRoleArn: NotRequired[str]
    createdAt: NotRequired[datetime]
    updatedAt: NotRequired[datetime]


class TagResourceRequestRequestTypeDef(TypedDict):
    resourceArn: str
    tags: Sequence[TagTypeDef]


class UpdateRepositoryCreationTemplateRequestRequestTypeDef(TypedDict):
    prefix: str
    description: NotRequired[str]
    encryptionConfiguration: NotRequired[
        EncryptionConfigurationForRepositoryCreationTemplateTypeDef
    ]
    resourceTags: NotRequired[Sequence[TagTypeDef]]
    imageTagMutability: NotRequired[ImageTagMutabilityType]
    repositoryPolicy: NotRequired[str]
    lifecyclePolicy: NotRequired[str]
    appliedFor: NotRequired[Sequence[RCTAppliedForType]]
    customRoleArn: NotRequired[str]


class CreateRepositoryRequestRequestTypeDef(TypedDict):
    repositoryName: str
    registryId: NotRequired[str]
    tags: NotRequired[Sequence[TagTypeDef]]
    imageTagMutability: NotRequired[ImageTagMutabilityType]
    imageScanningConfiguration: NotRequired[ImageScanningConfigurationTypeDef]
    encryptionConfiguration: NotRequired[EncryptionConfigurationTypeDef]


class PutImageScanningConfigurationRequestRequestTypeDef(TypedDict):
    repositoryName: str
    imageScanningConfiguration: ImageScanningConfigurationTypeDef
    registryId: NotRequired[str]


class PutImageScanningConfigurationResponseTypeDef(TypedDict):
    registryId: str
    repositoryName: str
    imageScanningConfiguration: ImageScanningConfigurationTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class RepositoryTypeDef(TypedDict):
    repositoryArn: NotRequired[str]
    registryId: NotRequired[str]
    repositoryName: NotRequired[str]
    repositoryUri: NotRequired[str]
    createdAt: NotRequired[datetime]
    imageTagMutability: NotRequired[ImageTagMutabilityType]
    imageScanningConfiguration: NotRequired[ImageScanningConfigurationTypeDef]
    encryptionConfiguration: NotRequired[EncryptionConfigurationTypeDef]


class CvssScoreDetailsTypeDef(TypedDict):
    adjustments: NotRequired[List[CvssScoreAdjustmentTypeDef]]
    score: NotRequired[float]
    scoreSource: NotRequired[str]
    scoringVector: NotRequired[str]
    version: NotRequired[str]


class DescribeImageReplicationStatusResponseTypeDef(TypedDict):
    repositoryName: str
    imageId: ImageIdentifierTypeDef
    replicationStatuses: List[ImageReplicationStatusTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class DescribeImageScanFindingsRequestPaginateTypeDef(TypedDict):
    repositoryName: str
    imageId: ImageIdentifierTypeDef
    registryId: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class DescribePullThroughCacheRulesRequestPaginateTypeDef(TypedDict):
    registryId: NotRequired[str]
    ecrRepositoryPrefixes: NotRequired[Sequence[str]]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class DescribeRepositoriesRequestPaginateTypeDef(TypedDict):
    registryId: NotRequired[str]
    repositoryNames: NotRequired[Sequence[str]]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class DescribeRepositoryCreationTemplatesRequestPaginateTypeDef(TypedDict):
    prefixes: NotRequired[Sequence[str]]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class DescribeImageScanFindingsRequestWaitTypeDef(TypedDict):
    repositoryName: str
    imageId: ImageIdentifierTypeDef
    registryId: NotRequired[str]
    nextToken: NotRequired[str]
    maxResults: NotRequired[int]
    WaiterConfig: NotRequired[WaiterConfigTypeDef]


class StartImageScanResponseTypeDef(TypedDict):
    registryId: str
    repositoryName: str
    imageId: ImageIdentifierTypeDef
    imageScanStatus: ImageScanStatusTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


DescribeImagesRequestPaginateTypeDef = TypedDict(
    "DescribeImagesRequestPaginateTypeDef",
    {
        "repositoryName": str,
        "registryId": NotRequired[str],
        "imageIds": NotRequired[Sequence[ImageIdentifierTypeDef]],
        "filter": NotRequired[DescribeImagesFilterTypeDef],
        "PaginationConfig": NotRequired[PaginatorConfigTypeDef],
    },
)
DescribeImagesRequestRequestTypeDef = TypedDict(
    "DescribeImagesRequestRequestTypeDef",
    {
        "repositoryName": str,
        "registryId": NotRequired[str],
        "imageIds": NotRequired[Sequence[ImageIdentifierTypeDef]],
        "nextToken": NotRequired[str],
        "maxResults": NotRequired[int],
        "filter": NotRequired[DescribeImagesFilterTypeDef],
    },
)


class DescribePullThroughCacheRulesResponseTypeDef(TypedDict):
    pullThroughCacheRules: List[PullThroughCacheRuleTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef
    nextToken: NotRequired[str]


GetLifecyclePolicyPreviewRequestPaginateTypeDef = TypedDict(
    "GetLifecyclePolicyPreviewRequestPaginateTypeDef",
    {
        "repositoryName": str,
        "registryId": NotRequired[str],
        "imageIds": NotRequired[Sequence[ImageIdentifierTypeDef]],
        "filter": NotRequired[LifecyclePolicyPreviewFilterTypeDef],
        "PaginationConfig": NotRequired[PaginatorConfigTypeDef],
    },
)
GetLifecyclePolicyPreviewRequestRequestTypeDef = TypedDict(
    "GetLifecyclePolicyPreviewRequestRequestTypeDef",
    {
        "repositoryName": str,
        "registryId": NotRequired[str],
        "imageIds": NotRequired[Sequence[ImageIdentifierTypeDef]],
        "nextToken": NotRequired[str],
        "maxResults": NotRequired[int],
        "filter": NotRequired[LifecyclePolicyPreviewFilterTypeDef],
    },
)
GetLifecyclePolicyPreviewRequestWaitTypeDef = TypedDict(
    "GetLifecyclePolicyPreviewRequestWaitTypeDef",
    {
        "repositoryName": str,
        "registryId": NotRequired[str],
        "imageIds": NotRequired[Sequence[ImageIdentifierTypeDef]],
        "nextToken": NotRequired[str],
        "maxResults": NotRequired[int],
        "filter": NotRequired[LifecyclePolicyPreviewFilterTypeDef],
        "WaiterConfig": NotRequired[WaiterConfigTypeDef],
    },
)


class ImageDetailTypeDef(TypedDict):
    registryId: NotRequired[str]
    repositoryName: NotRequired[str]
    imageDigest: NotRequired[str]
    imageTags: NotRequired[List[str]]
    imageSizeInBytes: NotRequired[int]
    imagePushedAt: NotRequired[datetime]
    imageScanStatus: NotRequired[ImageScanStatusTypeDef]
    imageScanFindingsSummary: NotRequired[ImageScanFindingsSummaryTypeDef]
    imageManifestMediaType: NotRequired[str]
    artifactMediaType: NotRequired[str]
    lastRecordedPullTime: NotRequired[datetime]


class LifecyclePolicyPreviewResultTypeDef(TypedDict):
    imageTags: NotRequired[List[str]]
    imageDigest: NotRequired[str]
    imagePushedAt: NotRequired[datetime]
    action: NotRequired[LifecyclePolicyRuleActionTypeDef]
    appliedRulePriority: NotRequired[int]


ListImagesRequestPaginateTypeDef = TypedDict(
    "ListImagesRequestPaginateTypeDef",
    {
        "repositoryName": str,
        "registryId": NotRequired[str],
        "filter": NotRequired[ListImagesFilterTypeDef],
        "PaginationConfig": NotRequired[PaginatorConfigTypeDef],
    },
)
ListImagesRequestRequestTypeDef = TypedDict(
    "ListImagesRequestRequestTypeDef",
    {
        "repositoryName": str,
        "registryId": NotRequired[str],
        "nextToken": NotRequired[str],
        "maxResults": NotRequired[int],
        "filter": NotRequired[ListImagesFilterTypeDef],
    },
)


class PackageVulnerabilityDetailsTypeDef(TypedDict):
    cvss: NotRequired[List[CvssScoreTypeDef]]
    referenceUrls: NotRequired[List[str]]
    relatedVulnerabilities: NotRequired[List[str]]
    source: NotRequired[str]
    sourceUrl: NotRequired[str]
    vendorCreatedAt: NotRequired[datetime]
    vendorSeverity: NotRequired[str]
    vendorUpdatedAt: NotRequired[datetime]
    vulnerabilityId: NotRequired[str]
    vulnerablePackages: NotRequired[List[VulnerablePackageTypeDef]]


class RemediationTypeDef(TypedDict):
    recommendation: NotRequired[RecommendationTypeDef]


class RegistryScanningRuleOutputTypeDef(TypedDict):
    scanFrequency: ScanFrequencyType
    repositoryFilters: List[ScanningRepositoryFilterTypeDef]


class RegistryScanningRuleTypeDef(TypedDict):
    scanFrequency: ScanFrequencyType
    repositoryFilters: Sequence[ScanningRepositoryFilterTypeDef]


class RepositoryScanningConfigurationTypeDef(TypedDict):
    repositoryArn: NotRequired[str]
    repositoryName: NotRequired[str]
    scanOnPush: NotRequired[bool]
    scanFrequency: NotRequired[ScanFrequencyType]
    appliedScanFilters: NotRequired[List[ScanningRepositoryFilterTypeDef]]


class ReplicationRuleOutputTypeDef(TypedDict):
    destinations: List[ReplicationDestinationTypeDef]
    repositoryFilters: NotRequired[List[RepositoryFilterTypeDef]]


class ReplicationRuleTypeDef(TypedDict):
    destinations: Sequence[ReplicationDestinationTypeDef]
    repositoryFilters: NotRequired[Sequence[RepositoryFilterTypeDef]]


ResourceTypeDef = TypedDict(
    "ResourceTypeDef",
    {
        "details": NotRequired[ResourceDetailsTypeDef],
        "id": NotRequired[str],
        "tags": NotRequired[Dict[str, str]],
        "type": NotRequired[str],
    },
)


class BatchDeleteImageResponseTypeDef(TypedDict):
    imageIds: List[ImageIdentifierTypeDef]
    failures: List[ImageFailureTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class BatchGetImageResponseTypeDef(TypedDict):
    images: List[ImageTypeDef]
    failures: List[ImageFailureTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class PutImageResponseTypeDef(TypedDict):
    image: ImageTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class CreateRepositoryCreationTemplateResponseTypeDef(TypedDict):
    registryId: str
    repositoryCreationTemplate: RepositoryCreationTemplateTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class DeleteRepositoryCreationTemplateResponseTypeDef(TypedDict):
    registryId: str
    repositoryCreationTemplate: RepositoryCreationTemplateTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class DescribeRepositoryCreationTemplatesResponseTypeDef(TypedDict):
    registryId: str
    repositoryCreationTemplates: List[RepositoryCreationTemplateTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef
    nextToken: NotRequired[str]


class UpdateRepositoryCreationTemplateResponseTypeDef(TypedDict):
    registryId: str
    repositoryCreationTemplate: RepositoryCreationTemplateTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class CreateRepositoryResponseTypeDef(TypedDict):
    repository: RepositoryTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class DeleteRepositoryResponseTypeDef(TypedDict):
    repository: RepositoryTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class DescribeRepositoriesResponseTypeDef(TypedDict):
    repositories: List[RepositoryTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef
    nextToken: NotRequired[str]


class ScoreDetailsTypeDef(TypedDict):
    cvss: NotRequired[CvssScoreDetailsTypeDef]


class DescribeImagesResponseTypeDef(TypedDict):
    imageDetails: List[ImageDetailTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef
    nextToken: NotRequired[str]


class GetLifecyclePolicyPreviewResponseTypeDef(TypedDict):
    registryId: str
    repositoryName: str
    lifecyclePolicyText: str
    status: LifecyclePolicyPreviewStatusType
    previewResults: List[LifecyclePolicyPreviewResultTypeDef]
    summary: LifecyclePolicyPreviewSummaryTypeDef
    ResponseMetadata: ResponseMetadataTypeDef
    nextToken: NotRequired[str]


class RegistryScanningConfigurationTypeDef(TypedDict):
    scanType: NotRequired[ScanTypeType]
    rules: NotRequired[List[RegistryScanningRuleOutputTypeDef]]


RegistryScanningRuleUnionTypeDef = Union[
    RegistryScanningRuleTypeDef, RegistryScanningRuleOutputTypeDef
]


class BatchGetRepositoryScanningConfigurationResponseTypeDef(TypedDict):
    scanningConfigurations: List[RepositoryScanningConfigurationTypeDef]
    failures: List[RepositoryScanningConfigurationFailureTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class ReplicationConfigurationOutputTypeDef(TypedDict):
    rules: List[ReplicationRuleOutputTypeDef]


ReplicationRuleUnionTypeDef = Union[ReplicationRuleTypeDef, ReplicationRuleOutputTypeDef]
EnhancedImageScanFindingTypeDef = TypedDict(
    "EnhancedImageScanFindingTypeDef",
    {
        "awsAccountId": NotRequired[str],
        "description": NotRequired[str],
        "findingArn": NotRequired[str],
        "firstObservedAt": NotRequired[datetime],
        "lastObservedAt": NotRequired[datetime],
        "packageVulnerabilityDetails": NotRequired[PackageVulnerabilityDetailsTypeDef],
        "remediation": NotRequired[RemediationTypeDef],
        "resources": NotRequired[List[ResourceTypeDef]],
        "score": NotRequired[float],
        "scoreDetails": NotRequired[ScoreDetailsTypeDef],
        "severity": NotRequired[str],
        "status": NotRequired[str],
        "title": NotRequired[str],
        "type": NotRequired[str],
        "updatedAt": NotRequired[datetime],
        "fixAvailable": NotRequired[str],
        "exploitAvailable": NotRequired[str],
    },
)


class GetRegistryScanningConfigurationResponseTypeDef(TypedDict):
    registryId: str
    scanningConfiguration: RegistryScanningConfigurationTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class PutRegistryScanningConfigurationResponseTypeDef(TypedDict):
    registryScanningConfiguration: RegistryScanningConfigurationTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class PutRegistryScanningConfigurationRequestRequestTypeDef(TypedDict):
    scanType: NotRequired[ScanTypeType]
    rules: NotRequired[Sequence[RegistryScanningRuleUnionTypeDef]]


class DescribeRegistryResponseTypeDef(TypedDict):
    registryId: str
    replicationConfiguration: ReplicationConfigurationOutputTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class PutReplicationConfigurationResponseTypeDef(TypedDict):
    replicationConfiguration: ReplicationConfigurationOutputTypeDef
    ResponseMetadata: ResponseMetadataTypeDef


class ReplicationConfigurationTypeDef(TypedDict):
    rules: Sequence[ReplicationRuleUnionTypeDef]


class ImageScanFindingsTypeDef(TypedDict):
    imageScanCompletedAt: NotRequired[datetime]
    vulnerabilitySourceUpdatedAt: NotRequired[datetime]
    findingSeverityCounts: NotRequired[Dict[FindingSeverityType, int]]
    findings: NotRequired[List[ImageScanFindingTypeDef]]
    enhancedFindings: NotRequired[List[EnhancedImageScanFindingTypeDef]]


class PutReplicationConfigurationRequestRequestTypeDef(TypedDict):
    replicationConfiguration: ReplicationConfigurationTypeDef


class DescribeImageScanFindingsResponseTypeDef(TypedDict):
    registryId: str
    repositoryName: str
    imageId: ImageIdentifierTypeDef
    imageScanStatus: ImageScanStatusTypeDef
    imageScanFindings: ImageScanFindingsTypeDef
    ResponseMetadata: ResponseMetadataTypeDef
    nextToken: NotRequired[str]
