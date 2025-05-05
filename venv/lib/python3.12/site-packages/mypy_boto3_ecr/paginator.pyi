"""
Type annotations for ecr service client paginators.

[Documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/paginators/)

Usage::

    ```python
    from boto3.session import Session

    from mypy_boto3_ecr.client import ECRClient
    from mypy_boto3_ecr.paginator import (
        DescribeImageScanFindingsPaginator,
        DescribeImagesPaginator,
        DescribePullThroughCacheRulesPaginator,
        DescribeRepositoriesPaginator,
        DescribeRepositoryCreationTemplatesPaginator,
        GetLifecyclePolicyPreviewPaginator,
        ListImagesPaginator,
    )

    session = Session()
    client: ECRClient = session.client("ecr")

    describe_image_scan_findings_paginator: DescribeImageScanFindingsPaginator = client.get_paginator("describe_image_scan_findings")
    describe_images_paginator: DescribeImagesPaginator = client.get_paginator("describe_images")
    describe_pull_through_cache_rules_paginator: DescribePullThroughCacheRulesPaginator = client.get_paginator("describe_pull_through_cache_rules")
    describe_repositories_paginator: DescribeRepositoriesPaginator = client.get_paginator("describe_repositories")
    describe_repository_creation_templates_paginator: DescribeRepositoryCreationTemplatesPaginator = client.get_paginator("describe_repository_creation_templates")
    get_lifecycle_policy_preview_paginator: GetLifecyclePolicyPreviewPaginator = client.get_paginator("get_lifecycle_policy_preview")
    list_images_paginator: ListImagesPaginator = client.get_paginator("list_images")
    ```

Copyright 2025 Vlad Emelianov
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from botocore.paginate import PageIterator, Paginator

from .type_defs import (
    DescribeImageScanFindingsRequestPaginateTypeDef,
    DescribeImageScanFindingsResponseTypeDef,
    DescribeImagesRequestPaginateTypeDef,
    DescribeImagesResponseTypeDef,
    DescribePullThroughCacheRulesRequestPaginateTypeDef,
    DescribePullThroughCacheRulesResponseTypeDef,
    DescribeRepositoriesRequestPaginateTypeDef,
    DescribeRepositoriesResponseTypeDef,
    DescribeRepositoryCreationTemplatesRequestPaginateTypeDef,
    DescribeRepositoryCreationTemplatesResponseTypeDef,
    GetLifecyclePolicyPreviewRequestPaginateTypeDef,
    GetLifecyclePolicyPreviewResponseTypeDef,
    ListImagesRequestPaginateTypeDef,
    ListImagesResponseTypeDef,
)

if sys.version_info >= (3, 12):
    from typing import Unpack
else:
    from typing_extensions import Unpack

__all__ = (
    "DescribeImageScanFindingsPaginator",
    "DescribeImagesPaginator",
    "DescribePullThroughCacheRulesPaginator",
    "DescribeRepositoriesPaginator",
    "DescribeRepositoryCreationTemplatesPaginator",
    "GetLifecyclePolicyPreviewPaginator",
    "ListImagesPaginator",
)

if TYPE_CHECKING:
    _DescribeImageScanFindingsPaginatorBase = Paginator[DescribeImageScanFindingsResponseTypeDef]
else:
    _DescribeImageScanFindingsPaginatorBase = Paginator  # type: ignore[assignment]

class DescribeImageScanFindingsPaginator(_DescribeImageScanFindingsPaginatorBase):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/paginator/DescribeImageScanFindings.html#ECR.Paginator.DescribeImageScanFindings)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/paginators/#describeimagescanfindingspaginator)
    """
    def paginate(  # type: ignore[override]
        self, **kwargs: Unpack[DescribeImageScanFindingsRequestPaginateTypeDef]
    ) -> PageIterator[DescribeImageScanFindingsResponseTypeDef]:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/paginator/DescribeImageScanFindings.html#ECR.Paginator.DescribeImageScanFindings.paginate)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/paginators/#describeimagescanfindingspaginator)
        """

if TYPE_CHECKING:
    _DescribeImagesPaginatorBase = Paginator[DescribeImagesResponseTypeDef]
else:
    _DescribeImagesPaginatorBase = Paginator  # type: ignore[assignment]

class DescribeImagesPaginator(_DescribeImagesPaginatorBase):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/paginator/DescribeImages.html#ECR.Paginator.DescribeImages)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/paginators/#describeimagespaginator)
    """
    def paginate(  # type: ignore[override]
        self, **kwargs: Unpack[DescribeImagesRequestPaginateTypeDef]
    ) -> PageIterator[DescribeImagesResponseTypeDef]:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/paginator/DescribeImages.html#ECR.Paginator.DescribeImages.paginate)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/paginators/#describeimagespaginator)
        """

if TYPE_CHECKING:
    _DescribePullThroughCacheRulesPaginatorBase = Paginator[
        DescribePullThroughCacheRulesResponseTypeDef
    ]
else:
    _DescribePullThroughCacheRulesPaginatorBase = Paginator  # type: ignore[assignment]

class DescribePullThroughCacheRulesPaginator(_DescribePullThroughCacheRulesPaginatorBase):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/paginator/DescribePullThroughCacheRules.html#ECR.Paginator.DescribePullThroughCacheRules)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/paginators/#describepullthroughcacherulespaginator)
    """
    def paginate(  # type: ignore[override]
        self, **kwargs: Unpack[DescribePullThroughCacheRulesRequestPaginateTypeDef]
    ) -> PageIterator[DescribePullThroughCacheRulesResponseTypeDef]:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/paginator/DescribePullThroughCacheRules.html#ECR.Paginator.DescribePullThroughCacheRules.paginate)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/paginators/#describepullthroughcacherulespaginator)
        """

if TYPE_CHECKING:
    _DescribeRepositoriesPaginatorBase = Paginator[DescribeRepositoriesResponseTypeDef]
else:
    _DescribeRepositoriesPaginatorBase = Paginator  # type: ignore[assignment]

class DescribeRepositoriesPaginator(_DescribeRepositoriesPaginatorBase):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/paginator/DescribeRepositories.html#ECR.Paginator.DescribeRepositories)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/paginators/#describerepositoriespaginator)
    """
    def paginate(  # type: ignore[override]
        self, **kwargs: Unpack[DescribeRepositoriesRequestPaginateTypeDef]
    ) -> PageIterator[DescribeRepositoriesResponseTypeDef]:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/paginator/DescribeRepositories.html#ECR.Paginator.DescribeRepositories.paginate)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/paginators/#describerepositoriespaginator)
        """

if TYPE_CHECKING:
    _DescribeRepositoryCreationTemplatesPaginatorBase = Paginator[
        DescribeRepositoryCreationTemplatesResponseTypeDef
    ]
else:
    _DescribeRepositoryCreationTemplatesPaginatorBase = Paginator  # type: ignore[assignment]

class DescribeRepositoryCreationTemplatesPaginator(
    _DescribeRepositoryCreationTemplatesPaginatorBase
):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/paginator/DescribeRepositoryCreationTemplates.html#ECR.Paginator.DescribeRepositoryCreationTemplates)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/paginators/#describerepositorycreationtemplatespaginator)
    """
    def paginate(  # type: ignore[override]
        self, **kwargs: Unpack[DescribeRepositoryCreationTemplatesRequestPaginateTypeDef]
    ) -> PageIterator[DescribeRepositoryCreationTemplatesResponseTypeDef]:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/paginator/DescribeRepositoryCreationTemplates.html#ECR.Paginator.DescribeRepositoryCreationTemplates.paginate)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/paginators/#describerepositorycreationtemplatespaginator)
        """

if TYPE_CHECKING:
    _GetLifecyclePolicyPreviewPaginatorBase = Paginator[GetLifecyclePolicyPreviewResponseTypeDef]
else:
    _GetLifecyclePolicyPreviewPaginatorBase = Paginator  # type: ignore[assignment]

class GetLifecyclePolicyPreviewPaginator(_GetLifecyclePolicyPreviewPaginatorBase):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/paginator/GetLifecyclePolicyPreview.html#ECR.Paginator.GetLifecyclePolicyPreview)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/paginators/#getlifecyclepolicypreviewpaginator)
    """
    def paginate(  # type: ignore[override]
        self, **kwargs: Unpack[GetLifecyclePolicyPreviewRequestPaginateTypeDef]
    ) -> PageIterator[GetLifecyclePolicyPreviewResponseTypeDef]:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/paginator/GetLifecyclePolicyPreview.html#ECR.Paginator.GetLifecyclePolicyPreview.paginate)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/paginators/#getlifecyclepolicypreviewpaginator)
        """

if TYPE_CHECKING:
    _ListImagesPaginatorBase = Paginator[ListImagesResponseTypeDef]
else:
    _ListImagesPaginatorBase = Paginator  # type: ignore[assignment]

class ListImagesPaginator(_ListImagesPaginatorBase):
    """
    [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/paginator/ListImages.html#ECR.Paginator.ListImages)
    [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/paginators/#listimagespaginator)
    """
    def paginate(  # type: ignore[override]
        self, **kwargs: Unpack[ListImagesRequestPaginateTypeDef]
    ) -> PageIterator[ListImagesResponseTypeDef]:
        """
        [Show boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecr/paginator/ListImages.html#ECR.Paginator.ListImages.paginate)
        [Show boto3-stubs documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_ecr/paginators/#listimagespaginator)
        """
