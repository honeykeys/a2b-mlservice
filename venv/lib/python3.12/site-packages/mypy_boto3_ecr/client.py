"""
Type annotations for ecr service Client.

[Documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/)

Usage::

    ```python
    from boto3.session import Session
    from mypy_boto3_ecr.client import ECRClient

    session = Session()
    client: ECRClient = session.client("ecr")
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
    DescribeImageScanFindingsPaginator,
    DescribeImagesPaginator,
    DescribePullThroughCacheRulesPaginator,
    DescribeRepositoriesPaginator,
    DescribeRepositoryCreationTemplatesPaginator,
    GetLifecyclePolicyPreviewPaginator,
    ListImagesPaginator,
)
from .type_defs import (
    BatchCheckLayerAvailabilityRequestRequestTypeDef,
    BatchCheckLayerAvailabilityResponseTypeDef,
    BatchDeleteImageRequestRequestTypeDef,
    BatchDeleteImageResponseTypeDef,
    BatchGetImageRequestRequestTypeDef,
    BatchGetImageResponseTypeDef,
    BatchGetRepositoryScanningConfigurationRequestRequestTypeDef,
    BatchGetRepositoryScanningConfigurationResponseTypeDef,
    CompleteLayerUploadRequestRequestTypeDef,
    CompleteLayerUploadResponseTypeDef,
    CreatePullThroughCacheRuleRequestRequestTypeDef,
    CreatePullThroughCacheRuleResponseTypeDef,
    CreateRepositoryCreationTemplateRequestRequestTypeDef,
    CreateRepositoryCreationTemplateResponseTypeDef,
    CreateRepositoryRequestRequestTypeDef,
    CreateRepositoryResponseTypeDef,
    DeleteLifecyclePolicyRequestRequestTypeDef,
    DeleteLifecyclePolicyResponseTypeDef,
    DeletePullThroughCacheRuleRequestRequestTypeDef,
    DeletePullThroughCacheRuleResponseTypeDef,
    DeleteRegistryPolicyResponseTypeDef,
    DeleteRepositoryCreationTemplateRequestRequestTypeDef,
    DeleteRepositoryCreationTemplateResponseTypeDef,
    DeleteRepositoryPolicyRequestRequestTypeDef,
    DeleteRepositoryPolicyResponseTypeDef,
    DeleteRepositoryRequestRequestTypeDef,
    DeleteRepositoryResponseTypeDef,
    DescribeImageReplicationStatusRequestRequestTypeDef,
    DescribeImageReplicationStatusResponseTypeDef,
    DescribeImageScanFindingsRequestRequestTypeDef,
    DescribeImageScanFindingsResponseTypeDef,
    DescribeImagesRequestRequestTypeDef,
    DescribeImagesResponseTypeDef,
    DescribePullThroughCacheRulesRequestRequestTypeDef,
    DescribePullThroughCacheRulesResponseTypeDef,
    DescribeRegistryResponseTypeDef,
    DescribeRepositoriesRequestRequestTypeDef,
    DescribeRepositoriesResponseTypeDef,
    DescribeRepositoryCreationTemplatesRequestRequestTypeDef,
    DescribeRepositoryCreationTemplatesResponseTypeDef,
    GetAccountSettingRequestRequestTypeDef,
    GetAccountSettingResponseTypeDef,
    GetAuthorizationTokenRequestRequestTypeDef,
    GetAuthorizationTokenResponseTypeDef,
    GetDownloadUrlForLayerRequestRequestTypeDef,
    GetDownloadUrlForLayerResponseTypeDef,
    GetLifecyclePolicyPreviewRequestRequestTypeDef,
    GetLifecyclePolicyPreviewResponseTypeDef,
    GetLifecyclePolicyRequestRequestTypeDef,
    GetLifecyclePolicyResponseTypeDef,
    GetRegistryPolicyResponseTypeDef,
    GetRegistryScanningConfigurationResponseTypeDef,
    GetRepositoryPolicyRequestRequestTypeDef,
    GetRepositoryPolicyResponseTypeDef,
    InitiateLayerUploadRequestRequestTypeDef,
    InitiateLayerUploadResponseTypeDef,
    ListImagesRequestRequestTypeDef,
    ListImagesResponseTypeDef,
    ListTagsForResourceRequestRequestTypeDef,
    ListTagsForResourceResponseTypeDef,
    PutAccountSettingRequestRequestTypeDef,
    PutAccountSettingResponseTypeDef,
    PutImageRequestRequestTypeDef,
    PutImageResponseTypeDef,
    PutImageScanningConfigurationRequestRequestTypeDef,
    PutImageScanningConfigurationResponseTypeDef,
    PutImageTagMutabilityRequestRequestTypeDef,
    PutImageTagMutabilityResponseTypeDef,
    PutLifecyclePolicyRequestRequestTypeDef,
    PutLifecyclePolicyResponseTypeDef,
    PutRegistryPolicyRequestRequestTypeDef,
    PutRegistryPolicyResponseTypeDef,
    PutRegistryScanningConfigurationRequestRequestTypeDef,
    PutRegistryScanningConfigurationResponseTypeDef,
    PutReplicationConfigurationRequestRequestTypeDef,
    PutReplicationConfigurationResponseTypeDef,
    SetRepositoryPolicyRequestRequestTypeDef,
    SetRepositoryPolicyResponseTypeDef,
    StartImageScanRequestRequestTypeDef,
    StartImageScanResponseTypeDef,
    StartLifecyclePolicyPreviewRequestRequestTypeDef,
    StartLifecyclePolicyPreviewResponseTypeDef,
    TagResourceRequestRequestTypeDef,
    UntagResourceRequestRequestTypeDef,
    UpdatePullThroughCacheRuleRequestRequestTypeDef,
    UpdatePullThroughCacheRuleResponseTypeDef,
    UpdateRepositoryCreationTemplateRequestRequestTypeDef,
    UpdateRepositoryCreationTemplateResponseTypeDef,
    UploadLayerPartRequestRequestTypeDef,
    UploadLayerPartResponseTypeDef,
    ValidatePullThroughCacheRuleRequestRequestTypeDef,
    ValidatePullThroughCacheRuleResponseTypeDef,
)
from .waiter import ImageScanCompleteWaiter, LifecyclePolicyPreviewCompleteWaiter

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


__all__ = ("ECRClient",)


class Exceptions(BaseClientExceptions):
    ClientError: Type[BotocoreClientError]
    EmptyUploadException: Type[BotocoreClientError]
    ImageAlreadyExistsException: Type[BotocoreClientError]
    ImageDigestDoesNotMatchException: Type[BotocoreClientError]
    ImageNotFoundException: Type[BotocoreClientError]
    ImageTagAlreadyExistsException: Type[BotocoreClientError]
    InvalidLayerException: Type[BotocoreClientError]
    InvalidLayerPartException: Type[BotocoreClientError]
    InvalidParameterException: Type[BotocoreClientError]
    InvalidTagParameterException: Type[BotocoreClientError]
    KmsException: Type[BotocoreClientError]
    LayerAlreadyExistsException: Type[BotocoreClientError]
    LayerInaccessibleException: Type[BotocoreClientError]
    LayerPartTooSmallException: Type[BotocoreClientError]
    LayersNotFoundException: Type[BotocoreClientError]
    LifecyclePolicyNotFoundException: Type[BotocoreClientError]
    LifecyclePolicyPreviewInProgressException: Type[BotocoreClientError]
    LifecyclePolicyPreviewNotFoundException: Type[BotocoreClientError]
    LimitExceededException: Type[BotocoreClientError]
    PullThroughCacheRuleAlreadyExistsException: Type[BotocoreClientError]
    PullThroughCacheRuleNotFoundException: Type[BotocoreClientError]
    ReferencedImagesNotFoundException: Type[BotocoreClientError]
    RegistryPolicyNotFoundException: Type[BotocoreClientError]
    RepositoryAlreadyExistsException: Type[BotocoreClientError]
    RepositoryNotEmptyException: Type[BotocoreClientError]
    RepositoryNotFoundException: Type[BotocoreClientError]
    RepositoryPolicyNotFoundException: Type[BotocoreClientError]
    ScanNotFoundException: Type[BotocoreClientError]
    SecretNotFoundException: Type[BotocoreClientError]
    ServerException: Type[BotocoreClientError]
    TemplateAlreadyExistsException: Type[BotocoreClientError]
    TemplateNotFoundException: Type[BotocoreClientError]
    TooManyTagsException: Type[BotocoreClientError]
    UnableToAccessSecretException: Type[BotocoreClientError]
    UnableToDecryptSecretValueException: Type[BotocoreClientError]
    UnableToGetUpstreamImageException: Type[BotocoreClientError]
    UnableToGetUpstreamLayerException: Type[BotocoreClientError]
    UnsupportedImageTypeException: Type[BotocoreClientError]
    UnsupportedUpstreamRegistryException: Type[BotocoreClientError]
    UploadNotFoundException: Type[BotocoreClientError]
    ValidationException: Type[BotocoreClientError]


class ECRClient(BaseClient):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr.html#ECR.Client)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/)
    """

    meta: ClientMeta

    @property
    def exceptions(self) -> Exceptions:
        """
        ECRClient exceptions.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr.html#ECR.Client)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#exceptions)
        """

    def can_paginate(self, operation_name: str) -> bool:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/can_paginate.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#can_paginate)
        """

    def generate_presigned_url(
        self,
        ClientMethod: str,
        Params: Mapping[str, Any] = ...,
        ExpiresIn: int = 3600,
        HttpMethod: str = ...,
    ) -> str:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/generate_presigned_url.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#generate_presigned_url)
        """

    def batch_check_layer_availability(
        self, **kwargs: Unpack[BatchCheckLayerAvailabilityRequestRequestTypeDef]
    ) -> BatchCheckLayerAvailabilityResponseTypeDef:
        """
        Checks the availability of one or more image layers in a repository.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/batch_check_layer_availability.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#batch_check_layer_availability)
        """

    def batch_delete_image(
        self, **kwargs: Unpack[BatchDeleteImageRequestRequestTypeDef]
    ) -> BatchDeleteImageResponseTypeDef:
        """
        Deletes a list of specified images within a repository.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/batch_delete_image.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#batch_delete_image)
        """

    def batch_get_image(
        self, **kwargs: Unpack[BatchGetImageRequestRequestTypeDef]
    ) -> BatchGetImageResponseTypeDef:
        """
        Gets detailed information for an image.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/batch_get_image.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#batch_get_image)
        """

    def batch_get_repository_scanning_configuration(
        self, **kwargs: Unpack[BatchGetRepositoryScanningConfigurationRequestRequestTypeDef]
    ) -> BatchGetRepositoryScanningConfigurationResponseTypeDef:
        """
        Gets the scanning configuration for one or more repositories.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/batch_get_repository_scanning_configuration.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#batch_get_repository_scanning_configuration)
        """

    def complete_layer_upload(
        self, **kwargs: Unpack[CompleteLayerUploadRequestRequestTypeDef]
    ) -> CompleteLayerUploadResponseTypeDef:
        """
        Informs Amazon ECR that the image layer upload has completed for a specified
        registry, repository name, and upload ID.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/complete_layer_upload.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#complete_layer_upload)
        """

    def create_pull_through_cache_rule(
        self, **kwargs: Unpack[CreatePullThroughCacheRuleRequestRequestTypeDef]
    ) -> CreatePullThroughCacheRuleResponseTypeDef:
        """
        Creates a pull through cache rule.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/create_pull_through_cache_rule.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#create_pull_through_cache_rule)
        """

    def create_repository(
        self, **kwargs: Unpack[CreateRepositoryRequestRequestTypeDef]
    ) -> CreateRepositoryResponseTypeDef:
        """
        Creates a repository.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/create_repository.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#create_repository)
        """

    def create_repository_creation_template(
        self, **kwargs: Unpack[CreateRepositoryCreationTemplateRequestRequestTypeDef]
    ) -> CreateRepositoryCreationTemplateResponseTypeDef:
        """
        Creates a repository creation template.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/create_repository_creation_template.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#create_repository_creation_template)
        """

    def delete_lifecycle_policy(
        self, **kwargs: Unpack[DeleteLifecyclePolicyRequestRequestTypeDef]
    ) -> DeleteLifecyclePolicyResponseTypeDef:
        """
        Deletes the lifecycle policy associated with the specified repository.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/delete_lifecycle_policy.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#delete_lifecycle_policy)
        """

    def delete_pull_through_cache_rule(
        self, **kwargs: Unpack[DeletePullThroughCacheRuleRequestRequestTypeDef]
    ) -> DeletePullThroughCacheRuleResponseTypeDef:
        """
        Deletes a pull through cache rule.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/delete_pull_through_cache_rule.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#delete_pull_through_cache_rule)
        """

    def delete_registry_policy(self) -> DeleteRegistryPolicyResponseTypeDef:
        """
        Deletes the registry permissions policy.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/delete_registry_policy.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#delete_registry_policy)
        """

    def delete_repository(
        self, **kwargs: Unpack[DeleteRepositoryRequestRequestTypeDef]
    ) -> DeleteRepositoryResponseTypeDef:
        """
        Deletes a repository.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/delete_repository.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#delete_repository)
        """

    def delete_repository_creation_template(
        self, **kwargs: Unpack[DeleteRepositoryCreationTemplateRequestRequestTypeDef]
    ) -> DeleteRepositoryCreationTemplateResponseTypeDef:
        """
        Deletes a repository creation template.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/delete_repository_creation_template.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#delete_repository_creation_template)
        """

    def delete_repository_policy(
        self, **kwargs: Unpack[DeleteRepositoryPolicyRequestRequestTypeDef]
    ) -> DeleteRepositoryPolicyResponseTypeDef:
        """
        Deletes the repository policy associated with the specified repository.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/delete_repository_policy.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#delete_repository_policy)
        """

    def describe_image_replication_status(
        self, **kwargs: Unpack[DescribeImageReplicationStatusRequestRequestTypeDef]
    ) -> DescribeImageReplicationStatusResponseTypeDef:
        """
        Returns the replication status for a specified image.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/describe_image_replication_status.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#describe_image_replication_status)
        """

    def describe_image_scan_findings(
        self, **kwargs: Unpack[DescribeImageScanFindingsRequestRequestTypeDef]
    ) -> DescribeImageScanFindingsResponseTypeDef:
        """
        Returns the scan findings for the specified image.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/describe_image_scan_findings.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#describe_image_scan_findings)
        """

    def describe_images(
        self, **kwargs: Unpack[DescribeImagesRequestRequestTypeDef]
    ) -> DescribeImagesResponseTypeDef:
        """
        Returns metadata about the images in a repository.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/describe_images.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#describe_images)
        """

    def describe_pull_through_cache_rules(
        self, **kwargs: Unpack[DescribePullThroughCacheRulesRequestRequestTypeDef]
    ) -> DescribePullThroughCacheRulesResponseTypeDef:
        """
        Returns the pull through cache rules for a registry.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/describe_pull_through_cache_rules.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#describe_pull_through_cache_rules)
        """

    def describe_registry(self) -> DescribeRegistryResponseTypeDef:
        """
        Describes the settings for a registry.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/describe_registry.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#describe_registry)
        """

    def describe_repositories(
        self, **kwargs: Unpack[DescribeRepositoriesRequestRequestTypeDef]
    ) -> DescribeRepositoriesResponseTypeDef:
        """
        Describes image repositories in a registry.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/describe_repositories.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#describe_repositories)
        """

    def describe_repository_creation_templates(
        self, **kwargs: Unpack[DescribeRepositoryCreationTemplatesRequestRequestTypeDef]
    ) -> DescribeRepositoryCreationTemplatesResponseTypeDef:
        """
        Returns details about the repository creation templates in a registry.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/describe_repository_creation_templates.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#describe_repository_creation_templates)
        """

    def get_account_setting(
        self, **kwargs: Unpack[GetAccountSettingRequestRequestTypeDef]
    ) -> GetAccountSettingResponseTypeDef:
        """
        Retrieves the account setting value for the specified setting name.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/get_account_setting.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#get_account_setting)
        """

    def get_authorization_token(
        self, **kwargs: Unpack[GetAuthorizationTokenRequestRequestTypeDef]
    ) -> GetAuthorizationTokenResponseTypeDef:
        """
        Retrieves an authorization token.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/get_authorization_token.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#get_authorization_token)
        """

    def get_download_url_for_layer(
        self, **kwargs: Unpack[GetDownloadUrlForLayerRequestRequestTypeDef]
    ) -> GetDownloadUrlForLayerResponseTypeDef:
        """
        Retrieves the pre-signed Amazon S3 download URL corresponding to an image layer.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/get_download_url_for_layer.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#get_download_url_for_layer)
        """

    def get_lifecycle_policy(
        self, **kwargs: Unpack[GetLifecyclePolicyRequestRequestTypeDef]
    ) -> GetLifecyclePolicyResponseTypeDef:
        """
        Retrieves the lifecycle policy for the specified repository.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/get_lifecycle_policy.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#get_lifecycle_policy)
        """

    def get_lifecycle_policy_preview(
        self, **kwargs: Unpack[GetLifecyclePolicyPreviewRequestRequestTypeDef]
    ) -> GetLifecyclePolicyPreviewResponseTypeDef:
        """
        Retrieves the results of the lifecycle policy preview request for the specified
        repository.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/get_lifecycle_policy_preview.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#get_lifecycle_policy_preview)
        """

    def get_registry_policy(self) -> GetRegistryPolicyResponseTypeDef:
        """
        Retrieves the permissions policy for a registry.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/get_registry_policy.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#get_registry_policy)
        """

    def get_registry_scanning_configuration(
        self,
    ) -> GetRegistryScanningConfigurationResponseTypeDef:
        """
        Retrieves the scanning configuration for a registry.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/get_registry_scanning_configuration.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#get_registry_scanning_configuration)
        """

    def get_repository_policy(
        self, **kwargs: Unpack[GetRepositoryPolicyRequestRequestTypeDef]
    ) -> GetRepositoryPolicyResponseTypeDef:
        """
        Retrieves the repository policy for the specified repository.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/get_repository_policy.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#get_repository_policy)
        """

    def initiate_layer_upload(
        self, **kwargs: Unpack[InitiateLayerUploadRequestRequestTypeDef]
    ) -> InitiateLayerUploadResponseTypeDef:
        """
        Notifies Amazon ECR that you intend to upload an image layer.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/initiate_layer_upload.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#initiate_layer_upload)
        """

    def list_images(
        self, **kwargs: Unpack[ListImagesRequestRequestTypeDef]
    ) -> ListImagesResponseTypeDef:
        """
        Lists all the image IDs for the specified repository.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/list_images.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#list_images)
        """

    def list_tags_for_resource(
        self, **kwargs: Unpack[ListTagsForResourceRequestRequestTypeDef]
    ) -> ListTagsForResourceResponseTypeDef:
        """
        List the tags for an Amazon ECR resource.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/list_tags_for_resource.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#list_tags_for_resource)
        """

    def put_account_setting(
        self, **kwargs: Unpack[PutAccountSettingRequestRequestTypeDef]
    ) -> PutAccountSettingResponseTypeDef:
        """
        Allows you to change the basic scan type version or registry policy scope.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/put_account_setting.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#put_account_setting)
        """

    def put_image(self, **kwargs: Unpack[PutImageRequestRequestTypeDef]) -> PutImageResponseTypeDef:
        """
        Creates or updates the image manifest and tags associated with an image.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/put_image.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#put_image)
        """

    def put_image_scanning_configuration(
        self, **kwargs: Unpack[PutImageScanningConfigurationRequestRequestTypeDef]
    ) -> PutImageScanningConfigurationResponseTypeDef:
        """
        The <code>PutImageScanningConfiguration</code> API is being deprecated, in
        favor of specifying the image scanning configuration at the registry level.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/put_image_scanning_configuration.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#put_image_scanning_configuration)
        """

    def put_image_tag_mutability(
        self, **kwargs: Unpack[PutImageTagMutabilityRequestRequestTypeDef]
    ) -> PutImageTagMutabilityResponseTypeDef:
        """
        Updates the image tag mutability settings for the specified repository.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/put_image_tag_mutability.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#put_image_tag_mutability)
        """

    def put_lifecycle_policy(
        self, **kwargs: Unpack[PutLifecyclePolicyRequestRequestTypeDef]
    ) -> PutLifecyclePolicyResponseTypeDef:
        """
        Creates or updates the lifecycle policy for the specified repository.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/put_lifecycle_policy.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#put_lifecycle_policy)
        """

    def put_registry_policy(
        self, **kwargs: Unpack[PutRegistryPolicyRequestRequestTypeDef]
    ) -> PutRegistryPolicyResponseTypeDef:
        """
        Creates or updates the permissions policy for your registry.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/put_registry_policy.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#put_registry_policy)
        """

    def put_registry_scanning_configuration(
        self, **kwargs: Unpack[PutRegistryScanningConfigurationRequestRequestTypeDef]
    ) -> PutRegistryScanningConfigurationResponseTypeDef:
        """
        Creates or updates the scanning configuration for your private registry.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/put_registry_scanning_configuration.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#put_registry_scanning_configuration)
        """

    def put_replication_configuration(
        self, **kwargs: Unpack[PutReplicationConfigurationRequestRequestTypeDef]
    ) -> PutReplicationConfigurationResponseTypeDef:
        """
        Creates or updates the replication configuration for a registry.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/put_replication_configuration.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#put_replication_configuration)
        """

    def set_repository_policy(
        self, **kwargs: Unpack[SetRepositoryPolicyRequestRequestTypeDef]
    ) -> SetRepositoryPolicyResponseTypeDef:
        """
        Applies a repository policy to the specified repository to control access
        permissions.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/set_repository_policy.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#set_repository_policy)
        """

    def start_image_scan(
        self, **kwargs: Unpack[StartImageScanRequestRequestTypeDef]
    ) -> StartImageScanResponseTypeDef:
        """
        Starts an image vulnerability scan.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/start_image_scan.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#start_image_scan)
        """

    def start_lifecycle_policy_preview(
        self, **kwargs: Unpack[StartLifecyclePolicyPreviewRequestRequestTypeDef]
    ) -> StartLifecyclePolicyPreviewResponseTypeDef:
        """
        Starts a preview of a lifecycle policy for the specified repository.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/start_lifecycle_policy_preview.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#start_lifecycle_policy_preview)
        """

    def tag_resource(self, **kwargs: Unpack[TagResourceRequestRequestTypeDef]) -> Dict[str, Any]:
        """
        Adds specified tags to a resource with the specified ARN.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/tag_resource.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#tag_resource)
        """

    def untag_resource(
        self, **kwargs: Unpack[UntagResourceRequestRequestTypeDef]
    ) -> Dict[str, Any]:
        """
        Deletes specified tags from a resource.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/untag_resource.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#untag_resource)
        """

    def update_pull_through_cache_rule(
        self, **kwargs: Unpack[UpdatePullThroughCacheRuleRequestRequestTypeDef]
    ) -> UpdatePullThroughCacheRuleResponseTypeDef:
        """
        Updates an existing pull through cache rule.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/update_pull_through_cache_rule.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#update_pull_through_cache_rule)
        """

    def update_repository_creation_template(
        self, **kwargs: Unpack[UpdateRepositoryCreationTemplateRequestRequestTypeDef]
    ) -> UpdateRepositoryCreationTemplateResponseTypeDef:
        """
        Updates an existing repository creation template.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/update_repository_creation_template.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#update_repository_creation_template)
        """

    def upload_layer_part(
        self, **kwargs: Unpack[UploadLayerPartRequestRequestTypeDef]
    ) -> UploadLayerPartResponseTypeDef:
        """
        Uploads an image layer part to Amazon ECR.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/upload_layer_part.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#upload_layer_part)
        """

    def validate_pull_through_cache_rule(
        self, **kwargs: Unpack[ValidatePullThroughCacheRuleRequestRequestTypeDef]
    ) -> ValidatePullThroughCacheRuleResponseTypeDef:
        """
        Validates an existing pull through cache rule for an upstream registry that
        requires authentication.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/validate_pull_through_cache_rule.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#validate_pull_through_cache_rule)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["describe_image_scan_findings"]
    ) -> DescribeImageScanFindingsPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["describe_images"]
    ) -> DescribeImagesPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["describe_pull_through_cache_rules"]
    ) -> DescribePullThroughCacheRulesPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["describe_repositories"]
    ) -> DescribeRepositoriesPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["describe_repository_creation_templates"]
    ) -> DescribeRepositoryCreationTemplatesPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["get_lifecycle_policy_preview"]
    ) -> GetLifecyclePolicyPreviewPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_paginator(  # type: ignore[override]
        self, operation_name: Literal["list_images"]
    ) -> ListImagesPaginator:
        """
        Create a paginator for an operation.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/get_paginator.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#get_paginator)
        """

    @overload  # type: ignore[override]
    def get_waiter(  # type: ignore[override]
        self, waiter_name: Literal["image_scan_complete"]
    ) -> ImageScanCompleteWaiter:
        """
        Returns an object that can wait for some condition.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/get_waiter.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#get_waiter)
        """

    @overload  # type: ignore[override]
    def get_waiter(  # type: ignore[override]
        self, waiter_name: Literal["lifecycle_policy_preview_complete"]
    ) -> LifecyclePolicyPreviewCompleteWaiter:
        """
        Returns an object that can wait for some condition.

        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/client/get_waiter.html)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/client/#get_waiter)
        """
