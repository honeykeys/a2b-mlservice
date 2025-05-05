"""
Type annotations for sts service type definitions.

[Documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_sts/type_defs/)

Usage::

    ```python
    from mypy_boto3_sts.type_defs import PolicyDescriptorTypeTypeDef

    data: PolicyDescriptorTypeTypeDef = ...
    ```

Copyright 2025 Vlad Emelianov
"""

from __future__ import annotations

import sys
from datetime import datetime

if sys.version_info >= (3, 9):
    from builtins import dict as Dict
    from collections.abc import Sequence
else:
    from typing import Dict, Sequence
if sys.version_info >= (3, 12):
    from typing import NotRequired, TypedDict
else:
    from typing_extensions import NotRequired, TypedDict

__all__ = (
    "AssumeRoleRequestRequestTypeDef",
    "AssumeRoleResponseTypeDef",
    "AssumeRoleWithSAMLRequestRequestTypeDef",
    "AssumeRoleWithSAMLResponseTypeDef",
    "AssumeRoleWithWebIdentityRequestRequestTypeDef",
    "AssumeRoleWithWebIdentityResponseTypeDef",
    "AssumeRootRequestRequestTypeDef",
    "AssumeRootResponseTypeDef",
    "AssumedRoleUserTypeDef",
    "CredentialsTypeDef",
    "DecodeAuthorizationMessageRequestRequestTypeDef",
    "DecodeAuthorizationMessageResponseTypeDef",
    "FederatedUserTypeDef",
    "GetAccessKeyInfoRequestRequestTypeDef",
    "GetAccessKeyInfoResponseTypeDef",
    "GetCallerIdentityResponseTypeDef",
    "GetFederationTokenRequestRequestTypeDef",
    "GetFederationTokenResponseTypeDef",
    "GetSessionTokenRequestRequestTypeDef",
    "GetSessionTokenResponseTypeDef",
    "PolicyDescriptorTypeTypeDef",
    "ProvidedContextTypeDef",
    "ResponseMetadataTypeDef",
    "TagTypeDef",
)

class PolicyDescriptorTypeTypeDef(TypedDict):
    arn: NotRequired[str]

class ProvidedContextTypeDef(TypedDict):
    ProviderArn: NotRequired[str]
    ContextAssertion: NotRequired[str]

class TagTypeDef(TypedDict):
    Key: str
    Value: str

class AssumedRoleUserTypeDef(TypedDict):
    AssumedRoleId: str
    Arn: str

class CredentialsTypeDef(TypedDict):
    AccessKeyId: str
    SecretAccessKey: str
    SessionToken: str
    Expiration: datetime

class ResponseMetadataTypeDef(TypedDict):
    RequestId: str
    HTTPStatusCode: int
    HTTPHeaders: Dict[str, str]
    RetryAttempts: int
    HostId: NotRequired[str]

class DecodeAuthorizationMessageRequestRequestTypeDef(TypedDict):
    EncodedMessage: str

class FederatedUserTypeDef(TypedDict):
    FederatedUserId: str
    Arn: str

class GetAccessKeyInfoRequestRequestTypeDef(TypedDict):
    AccessKeyId: str

class GetSessionTokenRequestRequestTypeDef(TypedDict):
    DurationSeconds: NotRequired[int]
    SerialNumber: NotRequired[str]
    TokenCode: NotRequired[str]

class AssumeRoleWithSAMLRequestRequestTypeDef(TypedDict):
    RoleArn: str
    PrincipalArn: str
    SAMLAssertion: str
    PolicyArns: NotRequired[Sequence[PolicyDescriptorTypeTypeDef]]
    Policy: NotRequired[str]
    DurationSeconds: NotRequired[int]

class AssumeRoleWithWebIdentityRequestRequestTypeDef(TypedDict):
    RoleArn: str
    RoleSessionName: str
    WebIdentityToken: str
    ProviderId: NotRequired[str]
    PolicyArns: NotRequired[Sequence[PolicyDescriptorTypeTypeDef]]
    Policy: NotRequired[str]
    DurationSeconds: NotRequired[int]

class AssumeRootRequestRequestTypeDef(TypedDict):
    TargetPrincipal: str
    TaskPolicyArn: PolicyDescriptorTypeTypeDef
    DurationSeconds: NotRequired[int]

class AssumeRoleRequestRequestTypeDef(TypedDict):
    RoleArn: str
    RoleSessionName: str
    PolicyArns: NotRequired[Sequence[PolicyDescriptorTypeTypeDef]]
    Policy: NotRequired[str]
    DurationSeconds: NotRequired[int]
    Tags: NotRequired[Sequence[TagTypeDef]]
    TransitiveTagKeys: NotRequired[Sequence[str]]
    ExternalId: NotRequired[str]
    SerialNumber: NotRequired[str]
    TokenCode: NotRequired[str]
    SourceIdentity: NotRequired[str]
    ProvidedContexts: NotRequired[Sequence[ProvidedContextTypeDef]]

class GetFederationTokenRequestRequestTypeDef(TypedDict):
    Name: str
    Policy: NotRequired[str]
    PolicyArns: NotRequired[Sequence[PolicyDescriptorTypeTypeDef]]
    DurationSeconds: NotRequired[int]
    Tags: NotRequired[Sequence[TagTypeDef]]

class AssumeRoleResponseTypeDef(TypedDict):
    Credentials: CredentialsTypeDef
    AssumedRoleUser: AssumedRoleUserTypeDef
    PackedPolicySize: int
    SourceIdentity: str
    ResponseMetadata: ResponseMetadataTypeDef

class AssumeRoleWithSAMLResponseTypeDef(TypedDict):
    Credentials: CredentialsTypeDef
    AssumedRoleUser: AssumedRoleUserTypeDef
    PackedPolicySize: int
    Subject: str
    SubjectType: str
    Issuer: str
    Audience: str
    NameQualifier: str
    SourceIdentity: str
    ResponseMetadata: ResponseMetadataTypeDef

class AssumeRoleWithWebIdentityResponseTypeDef(TypedDict):
    Credentials: CredentialsTypeDef
    SubjectFromWebIdentityToken: str
    AssumedRoleUser: AssumedRoleUserTypeDef
    PackedPolicySize: int
    Provider: str
    Audience: str
    SourceIdentity: str
    ResponseMetadata: ResponseMetadataTypeDef

class AssumeRootResponseTypeDef(TypedDict):
    Credentials: CredentialsTypeDef
    SourceIdentity: str
    ResponseMetadata: ResponseMetadataTypeDef

class DecodeAuthorizationMessageResponseTypeDef(TypedDict):
    DecodedMessage: str
    ResponseMetadata: ResponseMetadataTypeDef

class GetAccessKeyInfoResponseTypeDef(TypedDict):
    Account: str
    ResponseMetadata: ResponseMetadataTypeDef

class GetCallerIdentityResponseTypeDef(TypedDict):
    UserId: str
    Account: str
    Arn: str
    ResponseMetadata: ResponseMetadataTypeDef

class GetSessionTokenResponseTypeDef(TypedDict):
    Credentials: CredentialsTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class GetFederationTokenResponseTypeDef(TypedDict):
    Credentials: CredentialsTypeDef
    FederatedUser: FederatedUserTypeDef
    PackedPolicySize: int
    ResponseMetadata: ResponseMetadataTypeDef
