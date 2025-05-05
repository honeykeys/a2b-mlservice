"""
Type annotations for apigateway service type definitions.

[Documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_apigateway/type_defs/)

Usage::

    ```python
    from mypy_boto3_apigateway.type_defs import AccessLogSettingsTypeDef

    data: AccessLogSettingsTypeDef = ...
    ```

Copyright 2025 Vlad Emelianov
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import IO, Any, Union

from botocore.response import StreamingBody

from .literals import (
    ApiKeySourceTypeType,
    AuthorizerTypeType,
    CacheClusterSizeType,
    CacheClusterStatusType,
    ConnectionTypeType,
    ContentHandlingStrategyType,
    DocumentationPartTypeType,
    DomainNameStatusType,
    EndpointTypeType,
    GatewayResponseTypeType,
    IntegrationTypeType,
    LocationStatusTypeType,
    OpType,
    PutModeType,
    QuotaPeriodTypeType,
    ResourceOwnerType,
    SecurityPolicyType,
    UnauthorizedCacheControlHeaderStrategyType,
    VpcLinkStatusType,
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
    "AccessLogSettingsTypeDef",
    "AccountTypeDef",
    "ApiKeyIdsTypeDef",
    "ApiKeyResponseTypeDef",
    "ApiKeyTypeDef",
    "ApiKeysTypeDef",
    "ApiStageOutputTypeDef",
    "ApiStageTypeDef",
    "ApiStageUnionTypeDef",
    "AuthorizerResponseTypeDef",
    "AuthorizerTypeDef",
    "AuthorizersTypeDef",
    "BasePathMappingResponseTypeDef",
    "BasePathMappingTypeDef",
    "BasePathMappingsTypeDef",
    "BlobTypeDef",
    "CanarySettingsOutputTypeDef",
    "CanarySettingsTypeDef",
    "ClientCertificateResponseTypeDef",
    "ClientCertificateTypeDef",
    "ClientCertificatesTypeDef",
    "CreateApiKeyRequestRequestTypeDef",
    "CreateAuthorizerRequestRequestTypeDef",
    "CreateBasePathMappingRequestRequestTypeDef",
    "CreateDeploymentRequestRequestTypeDef",
    "CreateDocumentationPartRequestRequestTypeDef",
    "CreateDocumentationVersionRequestRequestTypeDef",
    "CreateDomainNameAccessAssociationRequestRequestTypeDef",
    "CreateDomainNameRequestRequestTypeDef",
    "CreateModelRequestRequestTypeDef",
    "CreateRequestValidatorRequestRequestTypeDef",
    "CreateResourceRequestRequestTypeDef",
    "CreateRestApiRequestRequestTypeDef",
    "CreateStageRequestRequestTypeDef",
    "CreateUsagePlanKeyRequestRequestTypeDef",
    "CreateUsagePlanRequestRequestTypeDef",
    "CreateVpcLinkRequestRequestTypeDef",
    "DeleteApiKeyRequestRequestTypeDef",
    "DeleteAuthorizerRequestRequestTypeDef",
    "DeleteBasePathMappingRequestRequestTypeDef",
    "DeleteClientCertificateRequestRequestTypeDef",
    "DeleteDeploymentRequestRequestTypeDef",
    "DeleteDocumentationPartRequestRequestTypeDef",
    "DeleteDocumentationVersionRequestRequestTypeDef",
    "DeleteDomainNameAccessAssociationRequestRequestTypeDef",
    "DeleteDomainNameRequestRequestTypeDef",
    "DeleteGatewayResponseRequestRequestTypeDef",
    "DeleteIntegrationRequestRequestTypeDef",
    "DeleteIntegrationResponseRequestRequestTypeDef",
    "DeleteMethodRequestRequestTypeDef",
    "DeleteMethodResponseRequestRequestTypeDef",
    "DeleteModelRequestRequestTypeDef",
    "DeleteRequestValidatorRequestRequestTypeDef",
    "DeleteResourceRequestRequestTypeDef",
    "DeleteRestApiRequestRequestTypeDef",
    "DeleteStageRequestRequestTypeDef",
    "DeleteUsagePlanKeyRequestRequestTypeDef",
    "DeleteUsagePlanRequestRequestTypeDef",
    "DeleteVpcLinkRequestRequestTypeDef",
    "DeploymentCanarySettingsTypeDef",
    "DeploymentResponseTypeDef",
    "DeploymentTypeDef",
    "DeploymentsTypeDef",
    "DocumentationPartIdsTypeDef",
    "DocumentationPartLocationTypeDef",
    "DocumentationPartResponseTypeDef",
    "DocumentationPartTypeDef",
    "DocumentationPartsTypeDef",
    "DocumentationVersionResponseTypeDef",
    "DocumentationVersionTypeDef",
    "DocumentationVersionsTypeDef",
    "DomainNameAccessAssociationResponseTypeDef",
    "DomainNameAccessAssociationTypeDef",
    "DomainNameAccessAssociationsTypeDef",
    "DomainNameResponseTypeDef",
    "DomainNameTypeDef",
    "DomainNamesTypeDef",
    "EmptyResponseMetadataTypeDef",
    "EndpointConfigurationOutputTypeDef",
    "EndpointConfigurationTypeDef",
    "ExportResponseTypeDef",
    "FlushStageAuthorizersCacheRequestRequestTypeDef",
    "FlushStageCacheRequestRequestTypeDef",
    "GatewayResponseResponseTypeDef",
    "GatewayResponseTypeDef",
    "GatewayResponsesTypeDef",
    "GenerateClientCertificateRequestRequestTypeDef",
    "GetApiKeyRequestRequestTypeDef",
    "GetApiKeysRequestPaginateTypeDef",
    "GetApiKeysRequestRequestTypeDef",
    "GetAuthorizerRequestRequestTypeDef",
    "GetAuthorizersRequestPaginateTypeDef",
    "GetAuthorizersRequestRequestTypeDef",
    "GetBasePathMappingRequestRequestTypeDef",
    "GetBasePathMappingsRequestPaginateTypeDef",
    "GetBasePathMappingsRequestRequestTypeDef",
    "GetClientCertificateRequestRequestTypeDef",
    "GetClientCertificatesRequestPaginateTypeDef",
    "GetClientCertificatesRequestRequestTypeDef",
    "GetDeploymentRequestRequestTypeDef",
    "GetDeploymentsRequestPaginateTypeDef",
    "GetDeploymentsRequestRequestTypeDef",
    "GetDocumentationPartRequestRequestTypeDef",
    "GetDocumentationPartsRequestPaginateTypeDef",
    "GetDocumentationPartsRequestRequestTypeDef",
    "GetDocumentationVersionRequestRequestTypeDef",
    "GetDocumentationVersionsRequestPaginateTypeDef",
    "GetDocumentationVersionsRequestRequestTypeDef",
    "GetDomainNameAccessAssociationsRequestRequestTypeDef",
    "GetDomainNameRequestRequestTypeDef",
    "GetDomainNamesRequestPaginateTypeDef",
    "GetDomainNamesRequestRequestTypeDef",
    "GetExportRequestRequestTypeDef",
    "GetGatewayResponseRequestRequestTypeDef",
    "GetGatewayResponsesRequestPaginateTypeDef",
    "GetGatewayResponsesRequestRequestTypeDef",
    "GetIntegrationRequestRequestTypeDef",
    "GetIntegrationResponseRequestRequestTypeDef",
    "GetMethodRequestRequestTypeDef",
    "GetMethodResponseRequestRequestTypeDef",
    "GetModelRequestRequestTypeDef",
    "GetModelTemplateRequestRequestTypeDef",
    "GetModelsRequestPaginateTypeDef",
    "GetModelsRequestRequestTypeDef",
    "GetRequestValidatorRequestRequestTypeDef",
    "GetRequestValidatorsRequestPaginateTypeDef",
    "GetRequestValidatorsRequestRequestTypeDef",
    "GetResourceRequestRequestTypeDef",
    "GetResourcesRequestPaginateTypeDef",
    "GetResourcesRequestRequestTypeDef",
    "GetRestApiRequestRequestTypeDef",
    "GetRestApisRequestPaginateTypeDef",
    "GetRestApisRequestRequestTypeDef",
    "GetSdkRequestRequestTypeDef",
    "GetSdkTypeRequestRequestTypeDef",
    "GetSdkTypesRequestPaginateTypeDef",
    "GetSdkTypesRequestRequestTypeDef",
    "GetStageRequestRequestTypeDef",
    "GetStagesRequestRequestTypeDef",
    "GetTagsRequestRequestTypeDef",
    "GetUsagePlanKeyRequestRequestTypeDef",
    "GetUsagePlanKeysRequestPaginateTypeDef",
    "GetUsagePlanKeysRequestRequestTypeDef",
    "GetUsagePlanRequestRequestTypeDef",
    "GetUsagePlansRequestPaginateTypeDef",
    "GetUsagePlansRequestRequestTypeDef",
    "GetUsageRequestPaginateTypeDef",
    "GetUsageRequestRequestTypeDef",
    "GetVpcLinkRequestRequestTypeDef",
    "GetVpcLinksRequestPaginateTypeDef",
    "GetVpcLinksRequestRequestTypeDef",
    "ImportApiKeysRequestRequestTypeDef",
    "ImportDocumentationPartsRequestRequestTypeDef",
    "ImportRestApiRequestRequestTypeDef",
    "IntegrationExtraResponseTypeDef",
    "IntegrationResponseResponseTypeDef",
    "IntegrationResponseTypeDef",
    "IntegrationTypeDef",
    "MethodExtraResponseTypeDef",
    "MethodResponseResponseTypeDef",
    "MethodResponseTypeDef",
    "MethodSettingTypeDef",
    "MethodSnapshotTypeDef",
    "MethodTypeDef",
    "ModelResponseTypeDef",
    "ModelTypeDef",
    "ModelsTypeDef",
    "MutualTlsAuthenticationInputTypeDef",
    "MutualTlsAuthenticationTypeDef",
    "PaginatorConfigTypeDef",
    "PatchOperationTypeDef",
    "PutGatewayResponseRequestRequestTypeDef",
    "PutIntegrationRequestRequestTypeDef",
    "PutIntegrationResponseRequestRequestTypeDef",
    "PutMethodRequestRequestTypeDef",
    "PutMethodResponseRequestRequestTypeDef",
    "PutRestApiRequestRequestTypeDef",
    "QuotaSettingsTypeDef",
    "RejectDomainNameAccessAssociationRequestRequestTypeDef",
    "RequestValidatorResponseTypeDef",
    "RequestValidatorTypeDef",
    "RequestValidatorsTypeDef",
    "ResourceResponseTypeDef",
    "ResourceTypeDef",
    "ResourcesTypeDef",
    "ResponseMetadataTypeDef",
    "RestApiResponseTypeDef",
    "RestApiTypeDef",
    "RestApisTypeDef",
    "SdkConfigurationPropertyTypeDef",
    "SdkResponseTypeDef",
    "SdkTypeResponseTypeDef",
    "SdkTypeTypeDef",
    "SdkTypesTypeDef",
    "StageKeyTypeDef",
    "StageResponseTypeDef",
    "StageTypeDef",
    "StagesTypeDef",
    "TagResourceRequestRequestTypeDef",
    "TagsTypeDef",
    "TemplateTypeDef",
    "TestInvokeAuthorizerRequestRequestTypeDef",
    "TestInvokeAuthorizerResponseTypeDef",
    "TestInvokeMethodRequestRequestTypeDef",
    "TestInvokeMethodResponseTypeDef",
    "ThrottleSettingsTypeDef",
    "TlsConfigTypeDef",
    "UntagResourceRequestRequestTypeDef",
    "UpdateAccountRequestRequestTypeDef",
    "UpdateApiKeyRequestRequestTypeDef",
    "UpdateAuthorizerRequestRequestTypeDef",
    "UpdateBasePathMappingRequestRequestTypeDef",
    "UpdateClientCertificateRequestRequestTypeDef",
    "UpdateDeploymentRequestRequestTypeDef",
    "UpdateDocumentationPartRequestRequestTypeDef",
    "UpdateDocumentationVersionRequestRequestTypeDef",
    "UpdateDomainNameRequestRequestTypeDef",
    "UpdateGatewayResponseRequestRequestTypeDef",
    "UpdateIntegrationRequestRequestTypeDef",
    "UpdateIntegrationResponseRequestRequestTypeDef",
    "UpdateMethodRequestRequestTypeDef",
    "UpdateMethodResponseRequestRequestTypeDef",
    "UpdateModelRequestRequestTypeDef",
    "UpdateRequestValidatorRequestRequestTypeDef",
    "UpdateResourceRequestRequestTypeDef",
    "UpdateRestApiRequestRequestTypeDef",
    "UpdateStageRequestRequestTypeDef",
    "UpdateUsagePlanRequestRequestTypeDef",
    "UpdateUsageRequestRequestTypeDef",
    "UpdateVpcLinkRequestRequestTypeDef",
    "UsagePlanKeyResponseTypeDef",
    "UsagePlanKeyTypeDef",
    "UsagePlanKeysTypeDef",
    "UsagePlanResponseTypeDef",
    "UsagePlanTypeDef",
    "UsagePlansTypeDef",
    "UsageTypeDef",
    "VpcLinkResponseTypeDef",
    "VpcLinkTypeDef",
    "VpcLinksTypeDef",
)

AccessLogSettingsTypeDef = TypedDict(
    "AccessLogSettingsTypeDef",
    {
        "format": NotRequired[str],
        "destinationArn": NotRequired[str],
    },
)


class ResponseMetadataTypeDef(TypedDict):
    RequestId: str
    HTTPStatusCode: int
    HTTPHeaders: Dict[str, str]
    RetryAttempts: int
    HostId: NotRequired[str]


class ThrottleSettingsTypeDef(TypedDict):
    burstLimit: NotRequired[int]
    rateLimit: NotRequired[float]


ApiKeyTypeDef = TypedDict(
    "ApiKeyTypeDef",
    {
        "id": NotRequired[str],
        "value": NotRequired[str],
        "name": NotRequired[str],
        "customerId": NotRequired[str],
        "description": NotRequired[str],
        "enabled": NotRequired[bool],
        "createdDate": NotRequired[datetime],
        "lastUpdatedDate": NotRequired[datetime],
        "stageKeys": NotRequired[List[str]],
        "tags": NotRequired[Dict[str, str]],
    },
)
AuthorizerTypeDef = TypedDict(
    "AuthorizerTypeDef",
    {
        "id": NotRequired[str],
        "name": NotRequired[str],
        "type": NotRequired[AuthorizerTypeType],
        "providerARNs": NotRequired[List[str]],
        "authType": NotRequired[str],
        "authorizerUri": NotRequired[str],
        "authorizerCredentials": NotRequired[str],
        "identitySource": NotRequired[str],
        "identityValidationExpression": NotRequired[str],
        "authorizerResultTtlInSeconds": NotRequired[int],
    },
)


class BasePathMappingTypeDef(TypedDict):
    basePath: NotRequired[str]
    restApiId: NotRequired[str]
    stage: NotRequired[str]


BlobTypeDef = Union[str, bytes, IO[Any], StreamingBody]


class CanarySettingsOutputTypeDef(TypedDict):
    percentTraffic: NotRequired[float]
    deploymentId: NotRequired[str]
    stageVariableOverrides: NotRequired[Dict[str, str]]
    useStageCache: NotRequired[bool]


class CanarySettingsTypeDef(TypedDict):
    percentTraffic: NotRequired[float]
    deploymentId: NotRequired[str]
    stageVariableOverrides: NotRequired[Mapping[str, str]]
    useStageCache: NotRequired[bool]


class ClientCertificateTypeDef(TypedDict):
    clientCertificateId: NotRequired[str]
    description: NotRequired[str]
    pemEncodedCertificate: NotRequired[str]
    createdDate: NotRequired[datetime]
    expirationDate: NotRequired[datetime]
    tags: NotRequired[Dict[str, str]]


class StageKeyTypeDef(TypedDict):
    restApiId: NotRequired[str]
    stageName: NotRequired[str]


CreateAuthorizerRequestRequestTypeDef = TypedDict(
    "CreateAuthorizerRequestRequestTypeDef",
    {
        "restApiId": str,
        "name": str,
        "type": AuthorizerTypeType,
        "providerARNs": NotRequired[Sequence[str]],
        "authType": NotRequired[str],
        "authorizerUri": NotRequired[str],
        "authorizerCredentials": NotRequired[str],
        "identitySource": NotRequired[str],
        "identityValidationExpression": NotRequired[str],
        "authorizerResultTtlInSeconds": NotRequired[int],
    },
)


class CreateBasePathMappingRequestRequestTypeDef(TypedDict):
    domainName: str
    restApiId: str
    domainNameId: NotRequired[str]
    basePath: NotRequired[str]
    stage: NotRequired[str]


class DeploymentCanarySettingsTypeDef(TypedDict):
    percentTraffic: NotRequired[float]
    stageVariableOverrides: NotRequired[Mapping[str, str]]
    useStageCache: NotRequired[bool]


DocumentationPartLocationTypeDef = TypedDict(
    "DocumentationPartLocationTypeDef",
    {
        "type": DocumentationPartTypeType,
        "path": NotRequired[str],
        "method": NotRequired[str],
        "statusCode": NotRequired[str],
        "name": NotRequired[str],
    },
)


class CreateDocumentationVersionRequestRequestTypeDef(TypedDict):
    restApiId: str
    documentationVersion: str
    stageName: NotRequired[str]
    description: NotRequired[str]


class CreateDomainNameAccessAssociationRequestRequestTypeDef(TypedDict):
    domainNameArn: str
    accessAssociationSourceType: Literal["VPCE"]
    accessAssociationSource: str
    tags: NotRequired[Mapping[str, str]]


EndpointConfigurationTypeDef = TypedDict(
    "EndpointConfigurationTypeDef",
    {
        "types": NotRequired[Sequence[EndpointTypeType]],
        "vpcEndpointIds": NotRequired[Sequence[str]],
    },
)


class MutualTlsAuthenticationInputTypeDef(TypedDict):
    truststoreUri: NotRequired[str]
    truststoreVersion: NotRequired[str]


class CreateModelRequestRequestTypeDef(TypedDict):
    restApiId: str
    name: str
    contentType: str
    description: NotRequired[str]
    schema: NotRequired[str]


class CreateRequestValidatorRequestRequestTypeDef(TypedDict):
    restApiId: str
    name: NotRequired[str]
    validateRequestBody: NotRequired[bool]
    validateRequestParameters: NotRequired[bool]


class CreateResourceRequestRequestTypeDef(TypedDict):
    restApiId: str
    parentId: str
    pathPart: str


class CreateUsagePlanKeyRequestRequestTypeDef(TypedDict):
    usagePlanId: str
    keyId: str
    keyType: str


class QuotaSettingsTypeDef(TypedDict):
    limit: NotRequired[int]
    offset: NotRequired[int]
    period: NotRequired[QuotaPeriodTypeType]


class CreateVpcLinkRequestRequestTypeDef(TypedDict):
    name: str
    targetArns: Sequence[str]
    description: NotRequired[str]
    tags: NotRequired[Mapping[str, str]]


class DeleteApiKeyRequestRequestTypeDef(TypedDict):
    apiKey: str


class DeleteAuthorizerRequestRequestTypeDef(TypedDict):
    restApiId: str
    authorizerId: str


class DeleteBasePathMappingRequestRequestTypeDef(TypedDict):
    domainName: str
    basePath: str
    domainNameId: NotRequired[str]


class DeleteClientCertificateRequestRequestTypeDef(TypedDict):
    clientCertificateId: str


class DeleteDeploymentRequestRequestTypeDef(TypedDict):
    restApiId: str
    deploymentId: str


class DeleteDocumentationPartRequestRequestTypeDef(TypedDict):
    restApiId: str
    documentationPartId: str


class DeleteDocumentationVersionRequestRequestTypeDef(TypedDict):
    restApiId: str
    documentationVersion: str


class DeleteDomainNameAccessAssociationRequestRequestTypeDef(TypedDict):
    domainNameAccessAssociationArn: str


class DeleteDomainNameRequestRequestTypeDef(TypedDict):
    domainName: str
    domainNameId: NotRequired[str]


class DeleteGatewayResponseRequestRequestTypeDef(TypedDict):
    restApiId: str
    responseType: GatewayResponseTypeType


class DeleteIntegrationRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    httpMethod: str


class DeleteIntegrationResponseRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    httpMethod: str
    statusCode: str


class DeleteMethodRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    httpMethod: str


class DeleteMethodResponseRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    httpMethod: str
    statusCode: str


class DeleteModelRequestRequestTypeDef(TypedDict):
    restApiId: str
    modelName: str


class DeleteRequestValidatorRequestRequestTypeDef(TypedDict):
    restApiId: str
    requestValidatorId: str


class DeleteResourceRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str


class DeleteRestApiRequestRequestTypeDef(TypedDict):
    restApiId: str


class DeleteStageRequestRequestTypeDef(TypedDict):
    restApiId: str
    stageName: str


class DeleteUsagePlanKeyRequestRequestTypeDef(TypedDict):
    usagePlanId: str
    keyId: str


class DeleteUsagePlanRequestRequestTypeDef(TypedDict):
    usagePlanId: str


class DeleteVpcLinkRequestRequestTypeDef(TypedDict):
    vpcLinkId: str


class MethodSnapshotTypeDef(TypedDict):
    authorizationType: NotRequired[str]
    apiKeyRequired: NotRequired[bool]


class DocumentationVersionTypeDef(TypedDict):
    version: NotRequired[str]
    createdDate: NotRequired[datetime]
    description: NotRequired[str]


class DomainNameAccessAssociationTypeDef(TypedDict):
    domainNameAccessAssociationArn: NotRequired[str]
    domainNameArn: NotRequired[str]
    accessAssociationSourceType: NotRequired[Literal["VPCE"]]
    accessAssociationSource: NotRequired[str]
    tags: NotRequired[Dict[str, str]]


EndpointConfigurationOutputTypeDef = TypedDict(
    "EndpointConfigurationOutputTypeDef",
    {
        "types": NotRequired[List[EndpointTypeType]],
        "vpcEndpointIds": NotRequired[List[str]],
    },
)


class MutualTlsAuthenticationTypeDef(TypedDict):
    truststoreUri: NotRequired[str]
    truststoreVersion: NotRequired[str]
    truststoreWarnings: NotRequired[List[str]]


class FlushStageAuthorizersCacheRequestRequestTypeDef(TypedDict):
    restApiId: str
    stageName: str


class FlushStageCacheRequestRequestTypeDef(TypedDict):
    restApiId: str
    stageName: str


class GatewayResponseTypeDef(TypedDict):
    responseType: NotRequired[GatewayResponseTypeType]
    statusCode: NotRequired[str]
    responseParameters: NotRequired[Dict[str, str]]
    responseTemplates: NotRequired[Dict[str, str]]
    defaultResponse: NotRequired[bool]


class GenerateClientCertificateRequestRequestTypeDef(TypedDict):
    description: NotRequired[str]
    tags: NotRequired[Mapping[str, str]]


class GetApiKeyRequestRequestTypeDef(TypedDict):
    apiKey: str
    includeValue: NotRequired[bool]


class PaginatorConfigTypeDef(TypedDict):
    MaxItems: NotRequired[int]
    PageSize: NotRequired[int]
    StartingToken: NotRequired[str]


class GetApiKeysRequestRequestTypeDef(TypedDict):
    position: NotRequired[str]
    limit: NotRequired[int]
    nameQuery: NotRequired[str]
    customerId: NotRequired[str]
    includeValues: NotRequired[bool]


class GetAuthorizerRequestRequestTypeDef(TypedDict):
    restApiId: str
    authorizerId: str


class GetAuthorizersRequestRequestTypeDef(TypedDict):
    restApiId: str
    position: NotRequired[str]
    limit: NotRequired[int]


class GetBasePathMappingRequestRequestTypeDef(TypedDict):
    domainName: str
    basePath: str
    domainNameId: NotRequired[str]


class GetBasePathMappingsRequestRequestTypeDef(TypedDict):
    domainName: str
    domainNameId: NotRequired[str]
    position: NotRequired[str]
    limit: NotRequired[int]


class GetClientCertificateRequestRequestTypeDef(TypedDict):
    clientCertificateId: str


class GetClientCertificatesRequestRequestTypeDef(TypedDict):
    position: NotRequired[str]
    limit: NotRequired[int]


class GetDeploymentRequestRequestTypeDef(TypedDict):
    restApiId: str
    deploymentId: str
    embed: NotRequired[Sequence[str]]


class GetDeploymentsRequestRequestTypeDef(TypedDict):
    restApiId: str
    position: NotRequired[str]
    limit: NotRequired[int]


class GetDocumentationPartRequestRequestTypeDef(TypedDict):
    restApiId: str
    documentationPartId: str


GetDocumentationPartsRequestRequestTypeDef = TypedDict(
    "GetDocumentationPartsRequestRequestTypeDef",
    {
        "restApiId": str,
        "type": NotRequired[DocumentationPartTypeType],
        "nameQuery": NotRequired[str],
        "path": NotRequired[str],
        "position": NotRequired[str],
        "limit": NotRequired[int],
        "locationStatus": NotRequired[LocationStatusTypeType],
    },
)


class GetDocumentationVersionRequestRequestTypeDef(TypedDict):
    restApiId: str
    documentationVersion: str


class GetDocumentationVersionsRequestRequestTypeDef(TypedDict):
    restApiId: str
    position: NotRequired[str]
    limit: NotRequired[int]


class GetDomainNameAccessAssociationsRequestRequestTypeDef(TypedDict):
    position: NotRequired[str]
    limit: NotRequired[int]
    resourceOwner: NotRequired[ResourceOwnerType]


class GetDomainNameRequestRequestTypeDef(TypedDict):
    domainName: str
    domainNameId: NotRequired[str]


class GetDomainNamesRequestRequestTypeDef(TypedDict):
    position: NotRequired[str]
    limit: NotRequired[int]
    resourceOwner: NotRequired[ResourceOwnerType]


class GetExportRequestRequestTypeDef(TypedDict):
    restApiId: str
    stageName: str
    exportType: str
    parameters: NotRequired[Mapping[str, str]]
    accepts: NotRequired[str]


class GetGatewayResponseRequestRequestTypeDef(TypedDict):
    restApiId: str
    responseType: GatewayResponseTypeType


class GetGatewayResponsesRequestRequestTypeDef(TypedDict):
    restApiId: str
    position: NotRequired[str]
    limit: NotRequired[int]


class GetIntegrationRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    httpMethod: str


class GetIntegrationResponseRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    httpMethod: str
    statusCode: str


class GetMethodRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    httpMethod: str


class GetMethodResponseRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    httpMethod: str
    statusCode: str


class GetModelRequestRequestTypeDef(TypedDict):
    restApiId: str
    modelName: str
    flatten: NotRequired[bool]


class GetModelTemplateRequestRequestTypeDef(TypedDict):
    restApiId: str
    modelName: str


class GetModelsRequestRequestTypeDef(TypedDict):
    restApiId: str
    position: NotRequired[str]
    limit: NotRequired[int]


class GetRequestValidatorRequestRequestTypeDef(TypedDict):
    restApiId: str
    requestValidatorId: str


class GetRequestValidatorsRequestRequestTypeDef(TypedDict):
    restApiId: str
    position: NotRequired[str]
    limit: NotRequired[int]


class GetResourceRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    embed: NotRequired[Sequence[str]]


class GetResourcesRequestRequestTypeDef(TypedDict):
    restApiId: str
    position: NotRequired[str]
    limit: NotRequired[int]
    embed: NotRequired[Sequence[str]]


class GetRestApiRequestRequestTypeDef(TypedDict):
    restApiId: str


class GetRestApisRequestRequestTypeDef(TypedDict):
    position: NotRequired[str]
    limit: NotRequired[int]


class GetSdkRequestRequestTypeDef(TypedDict):
    restApiId: str
    stageName: str
    sdkType: str
    parameters: NotRequired[Mapping[str, str]]


GetSdkTypeRequestRequestTypeDef = TypedDict(
    "GetSdkTypeRequestRequestTypeDef",
    {
        "id": str,
    },
)


class GetSdkTypesRequestRequestTypeDef(TypedDict):
    position: NotRequired[str]
    limit: NotRequired[int]


class GetStageRequestRequestTypeDef(TypedDict):
    restApiId: str
    stageName: str


class GetStagesRequestRequestTypeDef(TypedDict):
    restApiId: str
    deploymentId: NotRequired[str]


class GetTagsRequestRequestTypeDef(TypedDict):
    resourceArn: str
    position: NotRequired[str]
    limit: NotRequired[int]


class GetUsagePlanKeyRequestRequestTypeDef(TypedDict):
    usagePlanId: str
    keyId: str


class GetUsagePlanKeysRequestRequestTypeDef(TypedDict):
    usagePlanId: str
    position: NotRequired[str]
    limit: NotRequired[int]
    nameQuery: NotRequired[str]


class GetUsagePlanRequestRequestTypeDef(TypedDict):
    usagePlanId: str


class GetUsagePlansRequestRequestTypeDef(TypedDict):
    position: NotRequired[str]
    keyId: NotRequired[str]
    limit: NotRequired[int]


class GetUsageRequestRequestTypeDef(TypedDict):
    usagePlanId: str
    startDate: str
    endDate: str
    keyId: NotRequired[str]
    position: NotRequired[str]
    limit: NotRequired[int]


class GetVpcLinkRequestRequestTypeDef(TypedDict):
    vpcLinkId: str


class GetVpcLinksRequestRequestTypeDef(TypedDict):
    position: NotRequired[str]
    limit: NotRequired[int]


class IntegrationResponseTypeDef(TypedDict):
    statusCode: NotRequired[str]
    selectionPattern: NotRequired[str]
    responseParameters: NotRequired[Dict[str, str]]
    responseTemplates: NotRequired[Dict[str, str]]
    contentHandling: NotRequired[ContentHandlingStrategyType]


class TlsConfigTypeDef(TypedDict):
    insecureSkipVerification: NotRequired[bool]


class MethodResponseTypeDef(TypedDict):
    statusCode: NotRequired[str]
    responseParameters: NotRequired[Dict[str, bool]]
    responseModels: NotRequired[Dict[str, str]]


class MethodSettingTypeDef(TypedDict):
    metricsEnabled: NotRequired[bool]
    loggingLevel: NotRequired[str]
    dataTraceEnabled: NotRequired[bool]
    throttlingBurstLimit: NotRequired[int]
    throttlingRateLimit: NotRequired[float]
    cachingEnabled: NotRequired[bool]
    cacheTtlInSeconds: NotRequired[int]
    cacheDataEncrypted: NotRequired[bool]
    requireAuthorizationForCacheControl: NotRequired[bool]
    unauthorizedCacheControlHeaderStrategy: NotRequired[UnauthorizedCacheControlHeaderStrategyType]


ModelTypeDef = TypedDict(
    "ModelTypeDef",
    {
        "id": NotRequired[str],
        "name": NotRequired[str],
        "description": NotRequired[str],
        "schema": NotRequired[str],
        "contentType": NotRequired[str],
    },
)
PatchOperationTypeDef = TypedDict(
    "PatchOperationTypeDef",
    {
        "op": NotRequired[OpType],
        "path": NotRequired[str],
        "value": NotRequired[str],
        "from": NotRequired[str],
    },
)


class PutGatewayResponseRequestRequestTypeDef(TypedDict):
    restApiId: str
    responseType: GatewayResponseTypeType
    statusCode: NotRequired[str]
    responseParameters: NotRequired[Mapping[str, str]]
    responseTemplates: NotRequired[Mapping[str, str]]


class PutIntegrationResponseRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    httpMethod: str
    statusCode: str
    selectionPattern: NotRequired[str]
    responseParameters: NotRequired[Mapping[str, str]]
    responseTemplates: NotRequired[Mapping[str, str]]
    contentHandling: NotRequired[ContentHandlingStrategyType]


class PutMethodRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    httpMethod: str
    authorizationType: str
    authorizerId: NotRequired[str]
    apiKeyRequired: NotRequired[bool]
    operationName: NotRequired[str]
    requestParameters: NotRequired[Mapping[str, bool]]
    requestModels: NotRequired[Mapping[str, str]]
    requestValidatorId: NotRequired[str]
    authorizationScopes: NotRequired[Sequence[str]]


class PutMethodResponseRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    httpMethod: str
    statusCode: str
    responseParameters: NotRequired[Mapping[str, bool]]
    responseModels: NotRequired[Mapping[str, str]]


class RejectDomainNameAccessAssociationRequestRequestTypeDef(TypedDict):
    domainNameAccessAssociationArn: str
    domainNameArn: str


RequestValidatorTypeDef = TypedDict(
    "RequestValidatorTypeDef",
    {
        "id": NotRequired[str],
        "name": NotRequired[str],
        "validateRequestBody": NotRequired[bool],
        "validateRequestParameters": NotRequired[bool],
    },
)


class SdkConfigurationPropertyTypeDef(TypedDict):
    name: NotRequired[str]
    friendlyName: NotRequired[str]
    description: NotRequired[str]
    required: NotRequired[bool]
    defaultValue: NotRequired[str]


class TagResourceRequestRequestTypeDef(TypedDict):
    resourceArn: str
    tags: Mapping[str, str]


class TestInvokeAuthorizerRequestRequestTypeDef(TypedDict):
    restApiId: str
    authorizerId: str
    headers: NotRequired[Mapping[str, str]]
    multiValueHeaders: NotRequired[Mapping[str, Sequence[str]]]
    pathWithQueryString: NotRequired[str]
    body: NotRequired[str]
    stageVariables: NotRequired[Mapping[str, str]]
    additionalContext: NotRequired[Mapping[str, str]]


class TestInvokeMethodRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    httpMethod: str
    pathWithQueryString: NotRequired[str]
    body: NotRequired[str]
    headers: NotRequired[Mapping[str, str]]
    multiValueHeaders: NotRequired[Mapping[str, Sequence[str]]]
    clientCertificateId: NotRequired[str]
    stageVariables: NotRequired[Mapping[str, str]]


class UntagResourceRequestRequestTypeDef(TypedDict):
    resourceArn: str
    tagKeys: Sequence[str]


UsagePlanKeyTypeDef = TypedDict(
    "UsagePlanKeyTypeDef",
    {
        "id": NotRequired[str],
        "type": NotRequired[str],
        "value": NotRequired[str],
        "name": NotRequired[str],
    },
)
VpcLinkTypeDef = TypedDict(
    "VpcLinkTypeDef",
    {
        "id": NotRequired[str],
        "name": NotRequired[str],
        "description": NotRequired[str],
        "targetArns": NotRequired[List[str]],
        "status": NotRequired[VpcLinkStatusType],
        "statusMessage": NotRequired[str],
        "tags": NotRequired[Dict[str, str]],
    },
)


class ApiKeyIdsTypeDef(TypedDict):
    ids: List[str]
    warnings: List[str]
    ResponseMetadata: ResponseMetadataTypeDef


ApiKeyResponseTypeDef = TypedDict(
    "ApiKeyResponseTypeDef",
    {
        "id": str,
        "value": str,
        "name": str,
        "customerId": str,
        "description": str,
        "enabled": bool,
        "createdDate": datetime,
        "lastUpdatedDate": datetime,
        "stageKeys": List[str],
        "tags": Dict[str, str],
        "ResponseMetadata": ResponseMetadataTypeDef,
    },
)
AuthorizerResponseTypeDef = TypedDict(
    "AuthorizerResponseTypeDef",
    {
        "id": str,
        "name": str,
        "type": AuthorizerTypeType,
        "providerARNs": List[str],
        "authType": str,
        "authorizerUri": str,
        "authorizerCredentials": str,
        "identitySource": str,
        "identityValidationExpression": str,
        "authorizerResultTtlInSeconds": int,
        "ResponseMetadata": ResponseMetadataTypeDef,
    },
)


class BasePathMappingResponseTypeDef(TypedDict):
    basePath: str
    restApiId: str
    stage: str
    ResponseMetadata: ResponseMetadataTypeDef


class ClientCertificateResponseTypeDef(TypedDict):
    clientCertificateId: str
    description: str
    pemEncodedCertificate: str
    createdDate: datetime
    expirationDate: datetime
    tags: Dict[str, str]
    ResponseMetadata: ResponseMetadataTypeDef


class DocumentationPartIdsTypeDef(TypedDict):
    ids: List[str]
    warnings: List[str]
    ResponseMetadata: ResponseMetadataTypeDef


class DocumentationVersionResponseTypeDef(TypedDict):
    version: str
    createdDate: datetime
    description: str
    ResponseMetadata: ResponseMetadataTypeDef


class DomainNameAccessAssociationResponseTypeDef(TypedDict):
    domainNameAccessAssociationArn: str
    domainNameArn: str
    accessAssociationSourceType: Literal["VPCE"]
    accessAssociationSource: str
    tags: Dict[str, str]
    ResponseMetadata: ResponseMetadataTypeDef


class EmptyResponseMetadataTypeDef(TypedDict):
    ResponseMetadata: ResponseMetadataTypeDef


class ExportResponseTypeDef(TypedDict):
    contentType: str
    contentDisposition: str
    body: StreamingBody
    ResponseMetadata: ResponseMetadataTypeDef


class GatewayResponseResponseTypeDef(TypedDict):
    responseType: GatewayResponseTypeType
    statusCode: str
    responseParameters: Dict[str, str]
    responseTemplates: Dict[str, str]
    defaultResponse: bool
    ResponseMetadata: ResponseMetadataTypeDef


class IntegrationResponseResponseTypeDef(TypedDict):
    statusCode: str
    selectionPattern: str
    responseParameters: Dict[str, str]
    responseTemplates: Dict[str, str]
    contentHandling: ContentHandlingStrategyType
    ResponseMetadata: ResponseMetadataTypeDef


class MethodResponseResponseTypeDef(TypedDict):
    statusCode: str
    responseParameters: Dict[str, bool]
    responseModels: Dict[str, str]
    ResponseMetadata: ResponseMetadataTypeDef


ModelResponseTypeDef = TypedDict(
    "ModelResponseTypeDef",
    {
        "id": str,
        "name": str,
        "description": str,
        "schema": str,
        "contentType": str,
        "ResponseMetadata": ResponseMetadataTypeDef,
    },
)
RequestValidatorResponseTypeDef = TypedDict(
    "RequestValidatorResponseTypeDef",
    {
        "id": str,
        "name": str,
        "validateRequestBody": bool,
        "validateRequestParameters": bool,
        "ResponseMetadata": ResponseMetadataTypeDef,
    },
)


class SdkResponseTypeDef(TypedDict):
    contentType: str
    contentDisposition: str
    body: StreamingBody
    ResponseMetadata: ResponseMetadataTypeDef


class TagsTypeDef(TypedDict):
    tags: Dict[str, str]
    ResponseMetadata: ResponseMetadataTypeDef


class TemplateTypeDef(TypedDict):
    value: str
    ResponseMetadata: ResponseMetadataTypeDef


class TestInvokeAuthorizerResponseTypeDef(TypedDict):
    clientStatus: int
    log: str
    latency: int
    principalId: str
    policy: str
    authorization: Dict[str, List[str]]
    claims: Dict[str, str]
    ResponseMetadata: ResponseMetadataTypeDef


class TestInvokeMethodResponseTypeDef(TypedDict):
    status: int
    body: str
    headers: Dict[str, str]
    multiValueHeaders: Dict[str, List[str]]
    log: str
    latency: int
    ResponseMetadata: ResponseMetadataTypeDef


UsagePlanKeyResponseTypeDef = TypedDict(
    "UsagePlanKeyResponseTypeDef",
    {
        "id": str,
        "type": str,
        "value": str,
        "name": str,
        "ResponseMetadata": ResponseMetadataTypeDef,
    },
)


class UsageTypeDef(TypedDict):
    usagePlanId: str
    startDate: str
    endDate: str
    position: str
    items: Dict[str, List[List[int]]]
    ResponseMetadata: ResponseMetadataTypeDef


VpcLinkResponseTypeDef = TypedDict(
    "VpcLinkResponseTypeDef",
    {
        "id": str,
        "name": str,
        "description": str,
        "targetArns": List[str],
        "status": VpcLinkStatusType,
        "statusMessage": str,
        "tags": Dict[str, str],
        "ResponseMetadata": ResponseMetadataTypeDef,
    },
)


class AccountTypeDef(TypedDict):
    cloudwatchRoleArn: str
    throttleSettings: ThrottleSettingsTypeDef
    features: List[str]
    apiKeyVersion: str
    ResponseMetadata: ResponseMetadataTypeDef


class ApiStageOutputTypeDef(TypedDict):
    apiId: NotRequired[str]
    stage: NotRequired[str]
    throttle: NotRequired[Dict[str, ThrottleSettingsTypeDef]]


class ApiStageTypeDef(TypedDict):
    apiId: NotRequired[str]
    stage: NotRequired[str]
    throttle: NotRequired[Mapping[str, ThrottleSettingsTypeDef]]


class ApiKeysTypeDef(TypedDict):
    warnings: List[str]
    position: str
    items: List[ApiKeyTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class AuthorizersTypeDef(TypedDict):
    position: str
    items: List[AuthorizerTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class BasePathMappingsTypeDef(TypedDict):
    position: str
    items: List[BasePathMappingTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


ImportApiKeysRequestRequestTypeDef = TypedDict(
    "ImportApiKeysRequestRequestTypeDef",
    {
        "body": BlobTypeDef,
        "format": Literal["csv"],
        "failOnWarnings": NotRequired[bool],
    },
)


class ImportDocumentationPartsRequestRequestTypeDef(TypedDict):
    restApiId: str
    body: BlobTypeDef
    mode: NotRequired[PutModeType]
    failOnWarnings: NotRequired[bool]


class ImportRestApiRequestRequestTypeDef(TypedDict):
    body: BlobTypeDef
    failOnWarnings: NotRequired[bool]
    parameters: NotRequired[Mapping[str, str]]


class PutRestApiRequestRequestTypeDef(TypedDict):
    restApiId: str
    body: BlobTypeDef
    mode: NotRequired[PutModeType]
    failOnWarnings: NotRequired[bool]
    parameters: NotRequired[Mapping[str, str]]


class CreateStageRequestRequestTypeDef(TypedDict):
    restApiId: str
    stageName: str
    deploymentId: str
    description: NotRequired[str]
    cacheClusterEnabled: NotRequired[bool]
    cacheClusterSize: NotRequired[CacheClusterSizeType]
    variables: NotRequired[Mapping[str, str]]
    documentationVersion: NotRequired[str]
    canarySettings: NotRequired[CanarySettingsTypeDef]
    tracingEnabled: NotRequired[bool]
    tags: NotRequired[Mapping[str, str]]


class ClientCertificatesTypeDef(TypedDict):
    position: str
    items: List[ClientCertificateTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class CreateApiKeyRequestRequestTypeDef(TypedDict):
    name: NotRequired[str]
    description: NotRequired[str]
    enabled: NotRequired[bool]
    generateDistinctId: NotRequired[bool]
    value: NotRequired[str]
    stageKeys: NotRequired[Sequence[StageKeyTypeDef]]
    customerId: NotRequired[str]
    tags: NotRequired[Mapping[str, str]]


class CreateDeploymentRequestRequestTypeDef(TypedDict):
    restApiId: str
    stageName: NotRequired[str]
    stageDescription: NotRequired[str]
    description: NotRequired[str]
    cacheClusterEnabled: NotRequired[bool]
    cacheClusterSize: NotRequired[CacheClusterSizeType]
    variables: NotRequired[Mapping[str, str]]
    canarySettings: NotRequired[DeploymentCanarySettingsTypeDef]
    tracingEnabled: NotRequired[bool]


class CreateDocumentationPartRequestRequestTypeDef(TypedDict):
    restApiId: str
    location: DocumentationPartLocationTypeDef
    properties: str


DocumentationPartResponseTypeDef = TypedDict(
    "DocumentationPartResponseTypeDef",
    {
        "id": str,
        "location": DocumentationPartLocationTypeDef,
        "properties": str,
        "ResponseMetadata": ResponseMetadataTypeDef,
    },
)
DocumentationPartTypeDef = TypedDict(
    "DocumentationPartTypeDef",
    {
        "id": NotRequired[str],
        "location": NotRequired[DocumentationPartLocationTypeDef],
        "properties": NotRequired[str],
    },
)


class CreateRestApiRequestRequestTypeDef(TypedDict):
    name: str
    description: NotRequired[str]
    version: NotRequired[str]
    cloneFrom: NotRequired[str]
    binaryMediaTypes: NotRequired[Sequence[str]]
    minimumCompressionSize: NotRequired[int]
    apiKeySource: NotRequired[ApiKeySourceTypeType]
    endpointConfiguration: NotRequired[EndpointConfigurationTypeDef]
    policy: NotRequired[str]
    tags: NotRequired[Mapping[str, str]]
    disableExecuteApiEndpoint: NotRequired[bool]


class CreateDomainNameRequestRequestTypeDef(TypedDict):
    domainName: str
    certificateName: NotRequired[str]
    certificateBody: NotRequired[str]
    certificatePrivateKey: NotRequired[str]
    certificateChain: NotRequired[str]
    certificateArn: NotRequired[str]
    regionalCertificateName: NotRequired[str]
    regionalCertificateArn: NotRequired[str]
    endpointConfiguration: NotRequired[EndpointConfigurationTypeDef]
    tags: NotRequired[Mapping[str, str]]
    securityPolicy: NotRequired[SecurityPolicyType]
    mutualTlsAuthentication: NotRequired[MutualTlsAuthenticationInputTypeDef]
    ownershipVerificationCertificateArn: NotRequired[str]
    policy: NotRequired[str]


DeploymentResponseTypeDef = TypedDict(
    "DeploymentResponseTypeDef",
    {
        "id": str,
        "description": str,
        "createdDate": datetime,
        "apiSummary": Dict[str, Dict[str, MethodSnapshotTypeDef]],
        "ResponseMetadata": ResponseMetadataTypeDef,
    },
)
DeploymentTypeDef = TypedDict(
    "DeploymentTypeDef",
    {
        "id": NotRequired[str],
        "description": NotRequired[str],
        "createdDate": NotRequired[datetime],
        "apiSummary": NotRequired[Dict[str, Dict[str, MethodSnapshotTypeDef]]],
    },
)


class DocumentationVersionsTypeDef(TypedDict):
    position: str
    items: List[DocumentationVersionTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class DomainNameAccessAssociationsTypeDef(TypedDict):
    position: str
    items: List[DomainNameAccessAssociationTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


RestApiResponseTypeDef = TypedDict(
    "RestApiResponseTypeDef",
    {
        "id": str,
        "name": str,
        "description": str,
        "createdDate": datetime,
        "version": str,
        "warnings": List[str],
        "binaryMediaTypes": List[str],
        "minimumCompressionSize": int,
        "apiKeySource": ApiKeySourceTypeType,
        "endpointConfiguration": EndpointConfigurationOutputTypeDef,
        "policy": str,
        "tags": Dict[str, str],
        "disableExecuteApiEndpoint": bool,
        "rootResourceId": str,
        "ResponseMetadata": ResponseMetadataTypeDef,
    },
)
RestApiTypeDef = TypedDict(
    "RestApiTypeDef",
    {
        "id": NotRequired[str],
        "name": NotRequired[str],
        "description": NotRequired[str],
        "createdDate": NotRequired[datetime],
        "version": NotRequired[str],
        "warnings": NotRequired[List[str]],
        "binaryMediaTypes": NotRequired[List[str]],
        "minimumCompressionSize": NotRequired[int],
        "apiKeySource": NotRequired[ApiKeySourceTypeType],
        "endpointConfiguration": NotRequired[EndpointConfigurationOutputTypeDef],
        "policy": NotRequired[str],
        "tags": NotRequired[Dict[str, str]],
        "disableExecuteApiEndpoint": NotRequired[bool],
        "rootResourceId": NotRequired[str],
    },
)


class DomainNameResponseTypeDef(TypedDict):
    domainName: str
    domainNameId: str
    domainNameArn: str
    certificateName: str
    certificateArn: str
    certificateUploadDate: datetime
    regionalDomainName: str
    regionalHostedZoneId: str
    regionalCertificateName: str
    regionalCertificateArn: str
    distributionDomainName: str
    distributionHostedZoneId: str
    endpointConfiguration: EndpointConfigurationOutputTypeDef
    domainNameStatus: DomainNameStatusType
    domainNameStatusMessage: str
    securityPolicy: SecurityPolicyType
    tags: Dict[str, str]
    mutualTlsAuthentication: MutualTlsAuthenticationTypeDef
    ownershipVerificationCertificateArn: str
    managementPolicy: str
    policy: str
    ResponseMetadata: ResponseMetadataTypeDef


class DomainNameTypeDef(TypedDict):
    domainName: NotRequired[str]
    domainNameId: NotRequired[str]
    domainNameArn: NotRequired[str]
    certificateName: NotRequired[str]
    certificateArn: NotRequired[str]
    certificateUploadDate: NotRequired[datetime]
    regionalDomainName: NotRequired[str]
    regionalHostedZoneId: NotRequired[str]
    regionalCertificateName: NotRequired[str]
    regionalCertificateArn: NotRequired[str]
    distributionDomainName: NotRequired[str]
    distributionHostedZoneId: NotRequired[str]
    endpointConfiguration: NotRequired[EndpointConfigurationOutputTypeDef]
    domainNameStatus: NotRequired[DomainNameStatusType]
    domainNameStatusMessage: NotRequired[str]
    securityPolicy: NotRequired[SecurityPolicyType]
    tags: NotRequired[Dict[str, str]]
    mutualTlsAuthentication: NotRequired[MutualTlsAuthenticationTypeDef]
    ownershipVerificationCertificateArn: NotRequired[str]
    managementPolicy: NotRequired[str]
    policy: NotRequired[str]


class GatewayResponsesTypeDef(TypedDict):
    position: str
    items: List[GatewayResponseTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class GetApiKeysRequestPaginateTypeDef(TypedDict):
    nameQuery: NotRequired[str]
    customerId: NotRequired[str]
    includeValues: NotRequired[bool]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class GetAuthorizersRequestPaginateTypeDef(TypedDict):
    restApiId: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class GetBasePathMappingsRequestPaginateTypeDef(TypedDict):
    domainName: str
    domainNameId: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class GetClientCertificatesRequestPaginateTypeDef(TypedDict):
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class GetDeploymentsRequestPaginateTypeDef(TypedDict):
    restApiId: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


GetDocumentationPartsRequestPaginateTypeDef = TypedDict(
    "GetDocumentationPartsRequestPaginateTypeDef",
    {
        "restApiId": str,
        "type": NotRequired[DocumentationPartTypeType],
        "nameQuery": NotRequired[str],
        "path": NotRequired[str],
        "locationStatus": NotRequired[LocationStatusTypeType],
        "PaginationConfig": NotRequired[PaginatorConfigTypeDef],
    },
)


class GetDocumentationVersionsRequestPaginateTypeDef(TypedDict):
    restApiId: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class GetDomainNamesRequestPaginateTypeDef(TypedDict):
    resourceOwner: NotRequired[ResourceOwnerType]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class GetGatewayResponsesRequestPaginateTypeDef(TypedDict):
    restApiId: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class GetModelsRequestPaginateTypeDef(TypedDict):
    restApiId: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class GetRequestValidatorsRequestPaginateTypeDef(TypedDict):
    restApiId: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class GetResourcesRequestPaginateTypeDef(TypedDict):
    restApiId: str
    embed: NotRequired[Sequence[str]]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class GetRestApisRequestPaginateTypeDef(TypedDict):
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class GetSdkTypesRequestPaginateTypeDef(TypedDict):
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class GetUsagePlanKeysRequestPaginateTypeDef(TypedDict):
    usagePlanId: str
    nameQuery: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class GetUsagePlansRequestPaginateTypeDef(TypedDict):
    keyId: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class GetUsageRequestPaginateTypeDef(TypedDict):
    usagePlanId: str
    startDate: str
    endDate: str
    keyId: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


class GetVpcLinksRequestPaginateTypeDef(TypedDict):
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]


IntegrationExtraResponseTypeDef = TypedDict(
    "IntegrationExtraResponseTypeDef",
    {
        "type": IntegrationTypeType,
        "httpMethod": str,
        "uri": str,
        "connectionType": ConnectionTypeType,
        "connectionId": str,
        "credentials": str,
        "requestParameters": Dict[str, str],
        "requestTemplates": Dict[str, str],
        "passthroughBehavior": str,
        "contentHandling": ContentHandlingStrategyType,
        "timeoutInMillis": int,
        "cacheNamespace": str,
        "cacheKeyParameters": List[str],
        "integrationResponses": Dict[str, IntegrationResponseTypeDef],
        "tlsConfig": TlsConfigTypeDef,
        "ResponseMetadata": ResponseMetadataTypeDef,
    },
)
IntegrationTypeDef = TypedDict(
    "IntegrationTypeDef",
    {
        "type": NotRequired[IntegrationTypeType],
        "httpMethod": NotRequired[str],
        "uri": NotRequired[str],
        "connectionType": NotRequired[ConnectionTypeType],
        "connectionId": NotRequired[str],
        "credentials": NotRequired[str],
        "requestParameters": NotRequired[Dict[str, str]],
        "requestTemplates": NotRequired[Dict[str, str]],
        "passthroughBehavior": NotRequired[str],
        "contentHandling": NotRequired[ContentHandlingStrategyType],
        "timeoutInMillis": NotRequired[int],
        "cacheNamespace": NotRequired[str],
        "cacheKeyParameters": NotRequired[List[str]],
        "integrationResponses": NotRequired[Dict[str, IntegrationResponseTypeDef]],
        "tlsConfig": NotRequired[TlsConfigTypeDef],
    },
)
PutIntegrationRequestRequestTypeDef = TypedDict(
    "PutIntegrationRequestRequestTypeDef",
    {
        "restApiId": str,
        "resourceId": str,
        "httpMethod": str,
        "type": IntegrationTypeType,
        "integrationHttpMethod": NotRequired[str],
        "uri": NotRequired[str],
        "connectionType": NotRequired[ConnectionTypeType],
        "connectionId": NotRequired[str],
        "credentials": NotRequired[str],
        "requestParameters": NotRequired[Mapping[str, str]],
        "requestTemplates": NotRequired[Mapping[str, str]],
        "passthroughBehavior": NotRequired[str],
        "cacheNamespace": NotRequired[str],
        "cacheKeyParameters": NotRequired[Sequence[str]],
        "contentHandling": NotRequired[ContentHandlingStrategyType],
        "timeoutInMillis": NotRequired[int],
        "tlsConfig": NotRequired[TlsConfigTypeDef],
    },
)


class StageResponseTypeDef(TypedDict):
    deploymentId: str
    clientCertificateId: str
    stageName: str
    description: str
    cacheClusterEnabled: bool
    cacheClusterSize: CacheClusterSizeType
    cacheClusterStatus: CacheClusterStatusType
    methodSettings: Dict[str, MethodSettingTypeDef]
    variables: Dict[str, str]
    documentationVersion: str
    accessLogSettings: AccessLogSettingsTypeDef
    canarySettings: CanarySettingsOutputTypeDef
    tracingEnabled: bool
    webAclArn: str
    tags: Dict[str, str]
    createdDate: datetime
    lastUpdatedDate: datetime
    ResponseMetadata: ResponseMetadataTypeDef


class StageTypeDef(TypedDict):
    deploymentId: NotRequired[str]
    clientCertificateId: NotRequired[str]
    stageName: NotRequired[str]
    description: NotRequired[str]
    cacheClusterEnabled: NotRequired[bool]
    cacheClusterSize: NotRequired[CacheClusterSizeType]
    cacheClusterStatus: NotRequired[CacheClusterStatusType]
    methodSettings: NotRequired[Dict[str, MethodSettingTypeDef]]
    variables: NotRequired[Dict[str, str]]
    documentationVersion: NotRequired[str]
    accessLogSettings: NotRequired[AccessLogSettingsTypeDef]
    canarySettings: NotRequired[CanarySettingsOutputTypeDef]
    tracingEnabled: NotRequired[bool]
    webAclArn: NotRequired[str]
    tags: NotRequired[Dict[str, str]]
    createdDate: NotRequired[datetime]
    lastUpdatedDate: NotRequired[datetime]


class ModelsTypeDef(TypedDict):
    position: str
    items: List[ModelTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class UpdateAccountRequestRequestTypeDef(TypedDict):
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateApiKeyRequestRequestTypeDef(TypedDict):
    apiKey: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateAuthorizerRequestRequestTypeDef(TypedDict):
    restApiId: str
    authorizerId: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateBasePathMappingRequestRequestTypeDef(TypedDict):
    domainName: str
    basePath: str
    domainNameId: NotRequired[str]
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateClientCertificateRequestRequestTypeDef(TypedDict):
    clientCertificateId: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateDeploymentRequestRequestTypeDef(TypedDict):
    restApiId: str
    deploymentId: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateDocumentationPartRequestRequestTypeDef(TypedDict):
    restApiId: str
    documentationPartId: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateDocumentationVersionRequestRequestTypeDef(TypedDict):
    restApiId: str
    documentationVersion: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateDomainNameRequestRequestTypeDef(TypedDict):
    domainName: str
    domainNameId: NotRequired[str]
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateGatewayResponseRequestRequestTypeDef(TypedDict):
    restApiId: str
    responseType: GatewayResponseTypeType
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateIntegrationRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    httpMethod: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateIntegrationResponseRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    httpMethod: str
    statusCode: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateMethodRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    httpMethod: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateMethodResponseRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    httpMethod: str
    statusCode: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateModelRequestRequestTypeDef(TypedDict):
    restApiId: str
    modelName: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateRequestValidatorRequestRequestTypeDef(TypedDict):
    restApiId: str
    requestValidatorId: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateResourceRequestRequestTypeDef(TypedDict):
    restApiId: str
    resourceId: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateRestApiRequestRequestTypeDef(TypedDict):
    restApiId: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateStageRequestRequestTypeDef(TypedDict):
    restApiId: str
    stageName: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateUsagePlanRequestRequestTypeDef(TypedDict):
    usagePlanId: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateUsageRequestRequestTypeDef(TypedDict):
    usagePlanId: str
    keyId: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class UpdateVpcLinkRequestRequestTypeDef(TypedDict):
    vpcLinkId: str
    patchOperations: NotRequired[Sequence[PatchOperationTypeDef]]


class RequestValidatorsTypeDef(TypedDict):
    position: str
    items: List[RequestValidatorTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


SdkTypeResponseTypeDef = TypedDict(
    "SdkTypeResponseTypeDef",
    {
        "id": str,
        "friendlyName": str,
        "description": str,
        "configurationProperties": List[SdkConfigurationPropertyTypeDef],
        "ResponseMetadata": ResponseMetadataTypeDef,
    },
)
SdkTypeTypeDef = TypedDict(
    "SdkTypeTypeDef",
    {
        "id": NotRequired[str],
        "friendlyName": NotRequired[str],
        "description": NotRequired[str],
        "configurationProperties": NotRequired[List[SdkConfigurationPropertyTypeDef]],
    },
)


class UsagePlanKeysTypeDef(TypedDict):
    position: str
    items: List[UsagePlanKeyTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class VpcLinksTypeDef(TypedDict):
    position: str
    items: List[VpcLinkTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


UsagePlanResponseTypeDef = TypedDict(
    "UsagePlanResponseTypeDef",
    {
        "id": str,
        "name": str,
        "description": str,
        "apiStages": List[ApiStageOutputTypeDef],
        "throttle": ThrottleSettingsTypeDef,
        "quota": QuotaSettingsTypeDef,
        "productCode": str,
        "tags": Dict[str, str],
        "ResponseMetadata": ResponseMetadataTypeDef,
    },
)
UsagePlanTypeDef = TypedDict(
    "UsagePlanTypeDef",
    {
        "id": NotRequired[str],
        "name": NotRequired[str],
        "description": NotRequired[str],
        "apiStages": NotRequired[List[ApiStageOutputTypeDef]],
        "throttle": NotRequired[ThrottleSettingsTypeDef],
        "quota": NotRequired[QuotaSettingsTypeDef],
        "productCode": NotRequired[str],
        "tags": NotRequired[Dict[str, str]],
    },
)
ApiStageUnionTypeDef = Union[ApiStageTypeDef, ApiStageOutputTypeDef]


class DocumentationPartsTypeDef(TypedDict):
    position: str
    items: List[DocumentationPartTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class DeploymentsTypeDef(TypedDict):
    position: str
    items: List[DeploymentTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class RestApisTypeDef(TypedDict):
    position: str
    items: List[RestApiTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class DomainNamesTypeDef(TypedDict):
    position: str
    items: List[DomainNameTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class MethodExtraResponseTypeDef(TypedDict):
    httpMethod: str
    authorizationType: str
    authorizerId: str
    apiKeyRequired: bool
    requestValidatorId: str
    operationName: str
    requestParameters: Dict[str, bool]
    requestModels: Dict[str, str]
    methodResponses: Dict[str, MethodResponseTypeDef]
    methodIntegration: IntegrationTypeDef
    authorizationScopes: List[str]
    ResponseMetadata: ResponseMetadataTypeDef


class MethodTypeDef(TypedDict):
    httpMethod: NotRequired[str]
    authorizationType: NotRequired[str]
    authorizerId: NotRequired[str]
    apiKeyRequired: NotRequired[bool]
    requestValidatorId: NotRequired[str]
    operationName: NotRequired[str]
    requestParameters: NotRequired[Dict[str, bool]]
    requestModels: NotRequired[Dict[str, str]]
    methodResponses: NotRequired[Dict[str, MethodResponseTypeDef]]
    methodIntegration: NotRequired[IntegrationTypeDef]
    authorizationScopes: NotRequired[List[str]]


class StagesTypeDef(TypedDict):
    item: List[StageTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class SdkTypesTypeDef(TypedDict):
    position: str
    items: List[SdkTypeTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class UsagePlansTypeDef(TypedDict):
    position: str
    items: List[UsagePlanTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef


class CreateUsagePlanRequestRequestTypeDef(TypedDict):
    name: str
    description: NotRequired[str]
    apiStages: NotRequired[Sequence[ApiStageUnionTypeDef]]
    throttle: NotRequired[ThrottleSettingsTypeDef]
    quota: NotRequired[QuotaSettingsTypeDef]
    tags: NotRequired[Mapping[str, str]]


ResourceResponseTypeDef = TypedDict(
    "ResourceResponseTypeDef",
    {
        "id": str,
        "parentId": str,
        "pathPart": str,
        "path": str,
        "resourceMethods": Dict[str, MethodTypeDef],
        "ResponseMetadata": ResponseMetadataTypeDef,
    },
)
ResourceTypeDef = TypedDict(
    "ResourceTypeDef",
    {
        "id": NotRequired[str],
        "parentId": NotRequired[str],
        "pathPart": NotRequired[str],
        "path": NotRequired[str],
        "resourceMethods": NotRequired[Dict[str, MethodTypeDef]],
    },
)


class ResourcesTypeDef(TypedDict):
    position: str
    items: List[ResourceTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef
