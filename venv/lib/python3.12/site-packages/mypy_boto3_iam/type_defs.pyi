"""
Type annotations for iam service type definitions.

[Documentation](https://youtype.github.io/boto3_stubs_docs/mypy_boto3_iam/type_defs/)

Usage::

    ```python
    from mypy_boto3_iam.type_defs import AccessDetailTypeDef

    data: AccessDetailTypeDef = ...
    ```

Copyright 2025 Vlad Emelianov
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Union

from .literals import (
    AccessAdvisorUsageGranularityTypeType,
    AssignmentStatusTypeType,
    ContextKeyTypeEnumType,
    DeletionTaskStatusTypeType,
    EncodingTypeType,
    EntityTypeType,
    FeatureTypeType,
    GlobalEndpointTokenVersionType,
    JobStatusTypeType,
    PolicyEvaluationDecisionTypeType,
    PolicyOwnerEntityTypeType,
    PolicyScopeTypeType,
    PolicySourceTypeType,
    PolicyTypeType,
    PolicyUsageTypeType,
    ReportStateTypeType,
    SortKeyTypeType,
    StatusTypeType,
    SummaryKeyTypeType,
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
    "AccessDetailTypeDef",
    "AccessKeyLastUsedTypeDef",
    "AccessKeyMetadataTypeDef",
    "AccessKeyTypeDef",
    "AddClientIDToOpenIDConnectProviderRequestRequestTypeDef",
    "AddRoleToInstanceProfileRequestInstanceProfileAddRoleTypeDef",
    "AddRoleToInstanceProfileRequestRequestTypeDef",
    "AddUserToGroupRequestGroupAddUserTypeDef",
    "AddUserToGroupRequestRequestTypeDef",
    "AddUserToGroupRequestUserAddGroupTypeDef",
    "AttachGroupPolicyRequestGroupAttachPolicyTypeDef",
    "AttachGroupPolicyRequestPolicyAttachGroupTypeDef",
    "AttachGroupPolicyRequestRequestTypeDef",
    "AttachRolePolicyRequestPolicyAttachRoleTypeDef",
    "AttachRolePolicyRequestRequestTypeDef",
    "AttachRolePolicyRequestRoleAttachPolicyTypeDef",
    "AttachUserPolicyRequestPolicyAttachUserTypeDef",
    "AttachUserPolicyRequestRequestTypeDef",
    "AttachUserPolicyRequestUserAttachPolicyTypeDef",
    "AttachedPermissionsBoundaryTypeDef",
    "AttachedPolicyTypeDef",
    "ChangePasswordRequestRequestTypeDef",
    "ChangePasswordRequestServiceResourceChangePasswordTypeDef",
    "ContextEntryTypeDef",
    "CreateAccessKeyRequestRequestTypeDef",
    "CreateAccessKeyResponseTypeDef",
    "CreateAccountAliasRequestRequestTypeDef",
    "CreateAccountAliasRequestServiceResourceCreateAccountAliasTypeDef",
    "CreateGroupRequestGroupCreateTypeDef",
    "CreateGroupRequestRequestTypeDef",
    "CreateGroupRequestServiceResourceCreateGroupTypeDef",
    "CreateGroupResponseTypeDef",
    "CreateInstanceProfileRequestRequestTypeDef",
    "CreateInstanceProfileRequestServiceResourceCreateInstanceProfileTypeDef",
    "CreateInstanceProfileResponseTypeDef",
    "CreateLoginProfileRequestLoginProfileCreateTypeDef",
    "CreateLoginProfileRequestRequestTypeDef",
    "CreateLoginProfileRequestUserCreateLoginProfileTypeDef",
    "CreateLoginProfileResponseTypeDef",
    "CreateOpenIDConnectProviderRequestRequestTypeDef",
    "CreateOpenIDConnectProviderResponseTypeDef",
    "CreatePolicyRequestRequestTypeDef",
    "CreatePolicyRequestServiceResourceCreatePolicyTypeDef",
    "CreatePolicyResponseTypeDef",
    "CreatePolicyVersionRequestPolicyCreateVersionTypeDef",
    "CreatePolicyVersionRequestRequestTypeDef",
    "CreatePolicyVersionResponseTypeDef",
    "CreateRoleRequestRequestTypeDef",
    "CreateRoleRequestServiceResourceCreateRoleTypeDef",
    "CreateRoleResponseTypeDef",
    "CreateSAMLProviderRequestRequestTypeDef",
    "CreateSAMLProviderRequestServiceResourceCreateSamlProviderTypeDef",
    "CreateSAMLProviderResponseTypeDef",
    "CreateServiceLinkedRoleRequestRequestTypeDef",
    "CreateServiceLinkedRoleResponseTypeDef",
    "CreateServiceSpecificCredentialRequestRequestTypeDef",
    "CreateServiceSpecificCredentialResponseTypeDef",
    "CreateUserRequestRequestTypeDef",
    "CreateUserRequestServiceResourceCreateUserTypeDef",
    "CreateUserRequestUserCreateTypeDef",
    "CreateUserResponseTypeDef",
    "CreateVirtualMFADeviceRequestRequestTypeDef",
    "CreateVirtualMFADeviceRequestServiceResourceCreateVirtualMfaDeviceTypeDef",
    "CreateVirtualMFADeviceResponseTypeDef",
    "DeactivateMFADeviceRequestRequestTypeDef",
    "DeleteAccessKeyRequestRequestTypeDef",
    "DeleteAccountAliasRequestRequestTypeDef",
    "DeleteGroupPolicyRequestRequestTypeDef",
    "DeleteGroupRequestRequestTypeDef",
    "DeleteInstanceProfileRequestRequestTypeDef",
    "DeleteLoginProfileRequestRequestTypeDef",
    "DeleteOpenIDConnectProviderRequestRequestTypeDef",
    "DeletePolicyRequestRequestTypeDef",
    "DeletePolicyVersionRequestRequestTypeDef",
    "DeleteRolePermissionsBoundaryRequestRequestTypeDef",
    "DeleteRolePolicyRequestRequestTypeDef",
    "DeleteRoleRequestRequestTypeDef",
    "DeleteSAMLProviderRequestRequestTypeDef",
    "DeleteSSHPublicKeyRequestRequestTypeDef",
    "DeleteServerCertificateRequestRequestTypeDef",
    "DeleteServiceLinkedRoleRequestRequestTypeDef",
    "DeleteServiceLinkedRoleResponseTypeDef",
    "DeleteServiceSpecificCredentialRequestRequestTypeDef",
    "DeleteSigningCertificateRequestRequestTypeDef",
    "DeleteUserPermissionsBoundaryRequestRequestTypeDef",
    "DeleteUserPolicyRequestRequestTypeDef",
    "DeleteUserRequestRequestTypeDef",
    "DeleteVirtualMFADeviceRequestRequestTypeDef",
    "DeletionTaskFailureReasonTypeTypeDef",
    "DetachGroupPolicyRequestGroupDetachPolicyTypeDef",
    "DetachGroupPolicyRequestPolicyDetachGroupTypeDef",
    "DetachGroupPolicyRequestRequestTypeDef",
    "DetachRolePolicyRequestPolicyDetachRoleTypeDef",
    "DetachRolePolicyRequestRequestTypeDef",
    "DetachRolePolicyRequestRoleDetachPolicyTypeDef",
    "DetachUserPolicyRequestPolicyDetachUserTypeDef",
    "DetachUserPolicyRequestRequestTypeDef",
    "DetachUserPolicyRequestUserDetachPolicyTypeDef",
    "DisableOrganizationsRootCredentialsManagementResponseTypeDef",
    "DisableOrganizationsRootSessionsResponseTypeDef",
    "EmptyResponseMetadataTypeDef",
    "EnableMFADeviceRequestMfaDeviceAssociateTypeDef",
    "EnableMFADeviceRequestRequestTypeDef",
    "EnableMFADeviceRequestUserEnableMfaTypeDef",
    "EnableOrganizationsRootCredentialsManagementResponseTypeDef",
    "EnableOrganizationsRootSessionsResponseTypeDef",
    "EntityDetailsTypeDef",
    "EntityInfoTypeDef",
    "ErrorDetailsTypeDef",
    "EvaluationResultTypeDef",
    "GenerateCredentialReportResponseTypeDef",
    "GenerateOrganizationsAccessReportRequestRequestTypeDef",
    "GenerateOrganizationsAccessReportResponseTypeDef",
    "GenerateServiceLastAccessedDetailsRequestRequestTypeDef",
    "GenerateServiceLastAccessedDetailsResponseTypeDef",
    "GetAccessKeyLastUsedRequestRequestTypeDef",
    "GetAccessKeyLastUsedResponseTypeDef",
    "GetAccountAuthorizationDetailsRequestPaginateTypeDef",
    "GetAccountAuthorizationDetailsRequestRequestTypeDef",
    "GetAccountAuthorizationDetailsResponseTypeDef",
    "GetAccountPasswordPolicyResponseTypeDef",
    "GetAccountSummaryResponseTypeDef",
    "GetContextKeysForCustomPolicyRequestRequestTypeDef",
    "GetContextKeysForPolicyResponseTypeDef",
    "GetContextKeysForPrincipalPolicyRequestRequestTypeDef",
    "GetCredentialReportResponseTypeDef",
    "GetGroupPolicyRequestRequestTypeDef",
    "GetGroupPolicyResponseTypeDef",
    "GetGroupRequestPaginateTypeDef",
    "GetGroupRequestRequestTypeDef",
    "GetGroupResponseTypeDef",
    "GetInstanceProfileRequestRequestTypeDef",
    "GetInstanceProfileRequestWaitTypeDef",
    "GetInstanceProfileResponseTypeDef",
    "GetLoginProfileRequestRequestTypeDef",
    "GetLoginProfileResponseTypeDef",
    "GetMFADeviceRequestRequestTypeDef",
    "GetMFADeviceResponseTypeDef",
    "GetOpenIDConnectProviderRequestRequestTypeDef",
    "GetOpenIDConnectProviderResponseTypeDef",
    "GetOrganizationsAccessReportRequestRequestTypeDef",
    "GetOrganizationsAccessReportResponseTypeDef",
    "GetPolicyRequestRequestTypeDef",
    "GetPolicyRequestWaitTypeDef",
    "GetPolicyResponseTypeDef",
    "GetPolicyVersionRequestRequestTypeDef",
    "GetPolicyVersionResponseTypeDef",
    "GetRolePolicyRequestRequestTypeDef",
    "GetRolePolicyResponseTypeDef",
    "GetRoleRequestRequestTypeDef",
    "GetRoleRequestWaitTypeDef",
    "GetRoleResponseTypeDef",
    "GetSAMLProviderRequestRequestTypeDef",
    "GetSAMLProviderResponseTypeDef",
    "GetSSHPublicKeyRequestRequestTypeDef",
    "GetSSHPublicKeyResponseTypeDef",
    "GetServerCertificateRequestRequestTypeDef",
    "GetServerCertificateResponseTypeDef",
    "GetServiceLastAccessedDetailsRequestRequestTypeDef",
    "GetServiceLastAccessedDetailsResponseTypeDef",
    "GetServiceLastAccessedDetailsWithEntitiesRequestRequestTypeDef",
    "GetServiceLastAccessedDetailsWithEntitiesResponseTypeDef",
    "GetServiceLinkedRoleDeletionStatusRequestRequestTypeDef",
    "GetServiceLinkedRoleDeletionStatusResponseTypeDef",
    "GetUserPolicyRequestRequestTypeDef",
    "GetUserPolicyResponseTypeDef",
    "GetUserRequestRequestTypeDef",
    "GetUserRequestWaitTypeDef",
    "GetUserResponseTypeDef",
    "GroupDetailTypeDef",
    "GroupTypeDef",
    "InstanceProfileTypeDef",
    "ListAccessKeysRequestPaginateTypeDef",
    "ListAccessKeysRequestRequestTypeDef",
    "ListAccessKeysResponseTypeDef",
    "ListAccountAliasesRequestPaginateTypeDef",
    "ListAccountAliasesRequestRequestTypeDef",
    "ListAccountAliasesResponseTypeDef",
    "ListAttachedGroupPoliciesRequestPaginateTypeDef",
    "ListAttachedGroupPoliciesRequestRequestTypeDef",
    "ListAttachedGroupPoliciesResponseTypeDef",
    "ListAttachedRolePoliciesRequestPaginateTypeDef",
    "ListAttachedRolePoliciesRequestRequestTypeDef",
    "ListAttachedRolePoliciesResponseTypeDef",
    "ListAttachedUserPoliciesRequestPaginateTypeDef",
    "ListAttachedUserPoliciesRequestRequestTypeDef",
    "ListAttachedUserPoliciesResponseTypeDef",
    "ListEntitiesForPolicyRequestPaginateTypeDef",
    "ListEntitiesForPolicyRequestRequestTypeDef",
    "ListEntitiesForPolicyResponseTypeDef",
    "ListGroupPoliciesRequestPaginateTypeDef",
    "ListGroupPoliciesRequestRequestTypeDef",
    "ListGroupPoliciesResponseTypeDef",
    "ListGroupsForUserRequestPaginateTypeDef",
    "ListGroupsForUserRequestRequestTypeDef",
    "ListGroupsForUserResponseTypeDef",
    "ListGroupsRequestPaginateTypeDef",
    "ListGroupsRequestRequestTypeDef",
    "ListGroupsResponseTypeDef",
    "ListInstanceProfileTagsRequestPaginateTypeDef",
    "ListInstanceProfileTagsRequestRequestTypeDef",
    "ListInstanceProfileTagsResponseTypeDef",
    "ListInstanceProfilesForRoleRequestPaginateTypeDef",
    "ListInstanceProfilesForRoleRequestRequestTypeDef",
    "ListInstanceProfilesForRoleResponseTypeDef",
    "ListInstanceProfilesRequestPaginateTypeDef",
    "ListInstanceProfilesRequestRequestTypeDef",
    "ListInstanceProfilesResponseTypeDef",
    "ListMFADeviceTagsRequestPaginateTypeDef",
    "ListMFADeviceTagsRequestRequestTypeDef",
    "ListMFADeviceTagsResponseTypeDef",
    "ListMFADevicesRequestPaginateTypeDef",
    "ListMFADevicesRequestRequestTypeDef",
    "ListMFADevicesResponseTypeDef",
    "ListOpenIDConnectProviderTagsRequestPaginateTypeDef",
    "ListOpenIDConnectProviderTagsRequestRequestTypeDef",
    "ListOpenIDConnectProviderTagsResponseTypeDef",
    "ListOpenIDConnectProvidersResponseTypeDef",
    "ListOrganizationsFeaturesResponseTypeDef",
    "ListPoliciesGrantingServiceAccessEntryTypeDef",
    "ListPoliciesGrantingServiceAccessRequestRequestTypeDef",
    "ListPoliciesGrantingServiceAccessResponseTypeDef",
    "ListPoliciesRequestPaginateTypeDef",
    "ListPoliciesRequestRequestTypeDef",
    "ListPoliciesResponseTypeDef",
    "ListPolicyTagsRequestPaginateTypeDef",
    "ListPolicyTagsRequestRequestTypeDef",
    "ListPolicyTagsResponseTypeDef",
    "ListPolicyVersionsRequestPaginateTypeDef",
    "ListPolicyVersionsRequestRequestTypeDef",
    "ListPolicyVersionsResponseTypeDef",
    "ListRolePoliciesRequestPaginateTypeDef",
    "ListRolePoliciesRequestRequestTypeDef",
    "ListRolePoliciesResponseTypeDef",
    "ListRoleTagsRequestPaginateTypeDef",
    "ListRoleTagsRequestRequestTypeDef",
    "ListRoleTagsResponseTypeDef",
    "ListRolesRequestPaginateTypeDef",
    "ListRolesRequestRequestTypeDef",
    "ListRolesResponseTypeDef",
    "ListSAMLProviderTagsRequestPaginateTypeDef",
    "ListSAMLProviderTagsRequestRequestTypeDef",
    "ListSAMLProviderTagsResponseTypeDef",
    "ListSAMLProvidersResponseTypeDef",
    "ListSSHPublicKeysRequestPaginateTypeDef",
    "ListSSHPublicKeysRequestRequestTypeDef",
    "ListSSHPublicKeysResponseTypeDef",
    "ListServerCertificateTagsRequestPaginateTypeDef",
    "ListServerCertificateTagsRequestRequestTypeDef",
    "ListServerCertificateTagsResponseTypeDef",
    "ListServerCertificatesRequestPaginateTypeDef",
    "ListServerCertificatesRequestRequestTypeDef",
    "ListServerCertificatesResponseTypeDef",
    "ListServiceSpecificCredentialsRequestRequestTypeDef",
    "ListServiceSpecificCredentialsResponseTypeDef",
    "ListSigningCertificatesRequestPaginateTypeDef",
    "ListSigningCertificatesRequestRequestTypeDef",
    "ListSigningCertificatesResponseTypeDef",
    "ListUserPoliciesRequestPaginateTypeDef",
    "ListUserPoliciesRequestRequestTypeDef",
    "ListUserPoliciesResponseTypeDef",
    "ListUserTagsRequestPaginateTypeDef",
    "ListUserTagsRequestRequestTypeDef",
    "ListUserTagsResponseTypeDef",
    "ListUsersRequestPaginateTypeDef",
    "ListUsersRequestRequestTypeDef",
    "ListUsersResponseTypeDef",
    "ListVirtualMFADevicesRequestPaginateTypeDef",
    "ListVirtualMFADevicesRequestRequestTypeDef",
    "ListVirtualMFADevicesResponseTypeDef",
    "LoginProfileTypeDef",
    "MFADeviceTypeDef",
    "ManagedPolicyDetailTypeDef",
    "OpenIDConnectProviderListEntryTypeDef",
    "OrganizationsDecisionDetailTypeDef",
    "PaginatorConfigTypeDef",
    "PasswordPolicyTypeDef",
    "PermissionsBoundaryDecisionDetailTypeDef",
    "PolicyDetailTypeDef",
    "PolicyDocumentDictTypeDef",
    "PolicyDocumentStatementTypeDef",
    "PolicyDocumentTypeDef",
    "PolicyGrantingServiceAccessTypeDef",
    "PolicyGroupTypeDef",
    "PolicyRoleTypeDef",
    "PolicyTypeDef",
    "PolicyUserTypeDef",
    "PolicyVersionTypeDef",
    "PositionTypeDef",
    "PutGroupPolicyRequestGroupCreatePolicyTypeDef",
    "PutGroupPolicyRequestGroupPolicyPutTypeDef",
    "PutGroupPolicyRequestRequestTypeDef",
    "PutRolePermissionsBoundaryRequestRequestTypeDef",
    "PutRolePolicyRequestRequestTypeDef",
    "PutRolePolicyRequestRolePolicyPutTypeDef",
    "PutUserPermissionsBoundaryRequestRequestTypeDef",
    "PutUserPolicyRequestRequestTypeDef",
    "PutUserPolicyRequestUserCreatePolicyTypeDef",
    "PutUserPolicyRequestUserPolicyPutTypeDef",
    "RemoveClientIDFromOpenIDConnectProviderRequestRequestTypeDef",
    "RemoveRoleFromInstanceProfileRequestInstanceProfileRemoveRoleTypeDef",
    "RemoveRoleFromInstanceProfileRequestRequestTypeDef",
    "RemoveUserFromGroupRequestGroupRemoveUserTypeDef",
    "RemoveUserFromGroupRequestRequestTypeDef",
    "RemoveUserFromGroupRequestUserRemoveGroupTypeDef",
    "ResetServiceSpecificCredentialRequestRequestTypeDef",
    "ResetServiceSpecificCredentialResponseTypeDef",
    "ResourceSpecificResultTypeDef",
    "ResponseMetadataTypeDef",
    "ResyncMFADeviceRequestMfaDeviceResyncTypeDef",
    "ResyncMFADeviceRequestRequestTypeDef",
    "RoleDetailTypeDef",
    "RoleLastUsedTypeDef",
    "RoleTypeDef",
    "RoleUsageTypeTypeDef",
    "SAMLProviderListEntryTypeDef",
    "SSHPublicKeyMetadataTypeDef",
    "SSHPublicKeyTypeDef",
    "ServerCertificateMetadataTypeDef",
    "ServerCertificateTypeDef",
    "ServiceLastAccessedTypeDef",
    "ServiceSpecificCredentialMetadataTypeDef",
    "ServiceSpecificCredentialTypeDef",
    "SetDefaultPolicyVersionRequestRequestTypeDef",
    "SetSecurityTokenServicePreferencesRequestRequestTypeDef",
    "SigningCertificateTypeDef",
    "SimulateCustomPolicyRequestPaginateTypeDef",
    "SimulateCustomPolicyRequestRequestTypeDef",
    "SimulatePolicyResponseTypeDef",
    "SimulatePrincipalPolicyRequestPaginateTypeDef",
    "SimulatePrincipalPolicyRequestRequestTypeDef",
    "StatementTypeDef",
    "TagInstanceProfileRequestRequestTypeDef",
    "TagMFADeviceRequestRequestTypeDef",
    "TagOpenIDConnectProviderRequestRequestTypeDef",
    "TagPolicyRequestRequestTypeDef",
    "TagRoleRequestRequestTypeDef",
    "TagSAMLProviderRequestRequestTypeDef",
    "TagServerCertificateRequestRequestTypeDef",
    "TagTypeDef",
    "TagUserRequestRequestTypeDef",
    "TrackedActionLastAccessedTypeDef",
    "UntagInstanceProfileRequestRequestTypeDef",
    "UntagMFADeviceRequestRequestTypeDef",
    "UntagOpenIDConnectProviderRequestRequestTypeDef",
    "UntagPolicyRequestRequestTypeDef",
    "UntagRoleRequestRequestTypeDef",
    "UntagSAMLProviderRequestRequestTypeDef",
    "UntagServerCertificateRequestRequestTypeDef",
    "UntagUserRequestRequestTypeDef",
    "UpdateAccessKeyRequestAccessKeyActivateTypeDef",
    "UpdateAccessKeyRequestAccessKeyDeactivateTypeDef",
    "UpdateAccessKeyRequestAccessKeyPairActivateTypeDef",
    "UpdateAccessKeyRequestAccessKeyPairDeactivateTypeDef",
    "UpdateAccessKeyRequestRequestTypeDef",
    "UpdateAccountPasswordPolicyRequestAccountPasswordPolicyUpdateTypeDef",
    "UpdateAccountPasswordPolicyRequestRequestTypeDef",
    "UpdateAccountPasswordPolicyRequestServiceResourceCreateAccountPasswordPolicyTypeDef",
    "UpdateAssumeRolePolicyRequestAssumeRolePolicyUpdateTypeDef",
    "UpdateAssumeRolePolicyRequestRequestTypeDef",
    "UpdateGroupRequestGroupUpdateTypeDef",
    "UpdateGroupRequestRequestTypeDef",
    "UpdateLoginProfileRequestLoginProfileUpdateTypeDef",
    "UpdateLoginProfileRequestRequestTypeDef",
    "UpdateOpenIDConnectProviderThumbprintRequestRequestTypeDef",
    "UpdateRoleDescriptionRequestRequestTypeDef",
    "UpdateRoleDescriptionResponseTypeDef",
    "UpdateRoleRequestRequestTypeDef",
    "UpdateSAMLProviderRequestRequestTypeDef",
    "UpdateSAMLProviderRequestSamlProviderUpdateTypeDef",
    "UpdateSAMLProviderResponseTypeDef",
    "UpdateSSHPublicKeyRequestRequestTypeDef",
    "UpdateServerCertificateRequestRequestTypeDef",
    "UpdateServerCertificateRequestServerCertificateUpdateTypeDef",
    "UpdateServiceSpecificCredentialRequestRequestTypeDef",
    "UpdateSigningCertificateRequestRequestTypeDef",
    "UpdateSigningCertificateRequestSigningCertificateActivateTypeDef",
    "UpdateSigningCertificateRequestSigningCertificateDeactivateTypeDef",
    "UpdateUserRequestRequestTypeDef",
    "UpdateUserRequestUserUpdateTypeDef",
    "UploadSSHPublicKeyRequestRequestTypeDef",
    "UploadSSHPublicKeyResponseTypeDef",
    "UploadServerCertificateRequestRequestTypeDef",
    "UploadServerCertificateRequestServiceResourceCreateServerCertificateTypeDef",
    "UploadServerCertificateResponseTypeDef",
    "UploadSigningCertificateRequestRequestTypeDef",
    "UploadSigningCertificateRequestServiceResourceCreateSigningCertificateTypeDef",
    "UploadSigningCertificateResponseTypeDef",
    "UserDetailTypeDef",
    "UserTypeDef",
    "VirtualMFADeviceTypeDef",
    "WaiterConfigTypeDef",
)

AccessDetailTypeDef = TypedDict(
    "AccessDetailTypeDef",
    {
        "ServiceName": str,
        "ServiceNamespace": str,
        "Region": NotRequired[str],
        "EntityPath": NotRequired[str],
        "LastAuthenticatedTime": NotRequired[datetime],
        "TotalAuthenticatedEntities": NotRequired[int],
    },
)
AccessKeyLastUsedTypeDef = TypedDict(
    "AccessKeyLastUsedTypeDef",
    {
        "ServiceName": str,
        "Region": str,
        "LastUsedDate": NotRequired[datetime],
    },
)

class AccessKeyMetadataTypeDef(TypedDict):
    UserName: NotRequired[str]
    AccessKeyId: NotRequired[str]
    Status: NotRequired[StatusTypeType]
    CreateDate: NotRequired[datetime]

class AccessKeyTypeDef(TypedDict):
    UserName: str
    AccessKeyId: str
    Status: StatusTypeType
    SecretAccessKey: str
    CreateDate: NotRequired[datetime]

class AddClientIDToOpenIDConnectProviderRequestRequestTypeDef(TypedDict):
    OpenIDConnectProviderArn: str
    ClientID: str

class AddRoleToInstanceProfileRequestInstanceProfileAddRoleTypeDef(TypedDict):
    RoleName: str

class AddRoleToInstanceProfileRequestRequestTypeDef(TypedDict):
    InstanceProfileName: str
    RoleName: str

class AddUserToGroupRequestGroupAddUserTypeDef(TypedDict):
    UserName: str

class AddUserToGroupRequestRequestTypeDef(TypedDict):
    GroupName: str
    UserName: str

class AddUserToGroupRequestUserAddGroupTypeDef(TypedDict):
    GroupName: str

class AttachGroupPolicyRequestGroupAttachPolicyTypeDef(TypedDict):
    PolicyArn: str

class AttachGroupPolicyRequestPolicyAttachGroupTypeDef(TypedDict):
    GroupName: str

class AttachGroupPolicyRequestRequestTypeDef(TypedDict):
    GroupName: str
    PolicyArn: str

class AttachRolePolicyRequestPolicyAttachRoleTypeDef(TypedDict):
    RoleName: str

class AttachRolePolicyRequestRequestTypeDef(TypedDict):
    RoleName: str
    PolicyArn: str

class AttachRolePolicyRequestRoleAttachPolicyTypeDef(TypedDict):
    PolicyArn: str

class AttachUserPolicyRequestPolicyAttachUserTypeDef(TypedDict):
    UserName: str

class AttachUserPolicyRequestRequestTypeDef(TypedDict):
    UserName: str
    PolicyArn: str

class AttachUserPolicyRequestUserAttachPolicyTypeDef(TypedDict):
    PolicyArn: str

class AttachedPermissionsBoundaryTypeDef(TypedDict):
    PermissionsBoundaryType: NotRequired[Literal["PermissionsBoundaryPolicy"]]
    PermissionsBoundaryArn: NotRequired[str]

class AttachedPolicyTypeDef(TypedDict):
    PolicyName: NotRequired[str]
    PolicyArn: NotRequired[str]

class ChangePasswordRequestRequestTypeDef(TypedDict):
    OldPassword: str
    NewPassword: str

class ChangePasswordRequestServiceResourceChangePasswordTypeDef(TypedDict):
    OldPassword: str
    NewPassword: str

class ContextEntryTypeDef(TypedDict):
    ContextKeyName: NotRequired[str]
    ContextKeyValues: NotRequired[Sequence[str]]
    ContextKeyType: NotRequired[ContextKeyTypeEnumType]

class CreateAccessKeyRequestRequestTypeDef(TypedDict):
    UserName: NotRequired[str]

class ResponseMetadataTypeDef(TypedDict):
    RequestId: str
    HTTPStatusCode: int
    HTTPHeaders: Dict[str, str]
    RetryAttempts: int
    HostId: NotRequired[str]

class CreateAccountAliasRequestRequestTypeDef(TypedDict):
    AccountAlias: str

class CreateAccountAliasRequestServiceResourceCreateAccountAliasTypeDef(TypedDict):
    AccountAlias: str

class CreateGroupRequestGroupCreateTypeDef(TypedDict):
    Path: NotRequired[str]

class CreateGroupRequestRequestTypeDef(TypedDict):
    GroupName: str
    Path: NotRequired[str]

class CreateGroupRequestServiceResourceCreateGroupTypeDef(TypedDict):
    GroupName: str
    Path: NotRequired[str]

class GroupTypeDef(TypedDict):
    Path: str
    GroupName: str
    GroupId: str
    Arn: str
    CreateDate: datetime

class TagTypeDef(TypedDict):
    Key: str
    Value: str

class CreateLoginProfileRequestLoginProfileCreateTypeDef(TypedDict):
    Password: NotRequired[str]
    PasswordResetRequired: NotRequired[bool]

class CreateLoginProfileRequestRequestTypeDef(TypedDict):
    UserName: NotRequired[str]
    Password: NotRequired[str]
    PasswordResetRequired: NotRequired[bool]

class CreateLoginProfileRequestUserCreateLoginProfileTypeDef(TypedDict):
    Password: NotRequired[str]
    PasswordResetRequired: NotRequired[bool]

class LoginProfileTypeDef(TypedDict):
    UserName: str
    CreateDate: datetime
    PasswordResetRequired: NotRequired[bool]

class CreatePolicyVersionRequestPolicyCreateVersionTypeDef(TypedDict):
    PolicyDocument: str
    SetAsDefault: NotRequired[bool]

class CreatePolicyVersionRequestRequestTypeDef(TypedDict):
    PolicyArn: str
    PolicyDocument: str
    SetAsDefault: NotRequired[bool]

class CreateServiceLinkedRoleRequestRequestTypeDef(TypedDict):
    AWSServiceName: str
    Description: NotRequired[str]
    CustomSuffix: NotRequired[str]

CreateServiceSpecificCredentialRequestRequestTypeDef = TypedDict(
    "CreateServiceSpecificCredentialRequestRequestTypeDef",
    {
        "UserName": str,
        "ServiceName": str,
    },
)
ServiceSpecificCredentialTypeDef = TypedDict(
    "ServiceSpecificCredentialTypeDef",
    {
        "CreateDate": datetime,
        "ServiceName": str,
        "ServiceUserName": str,
        "ServicePassword": str,
        "ServiceSpecificCredentialId": str,
        "UserName": str,
        "Status": StatusTypeType,
    },
)

class DeactivateMFADeviceRequestRequestTypeDef(TypedDict):
    SerialNumber: str
    UserName: NotRequired[str]

class DeleteAccessKeyRequestRequestTypeDef(TypedDict):
    AccessKeyId: str
    UserName: NotRequired[str]

class DeleteAccountAliasRequestRequestTypeDef(TypedDict):
    AccountAlias: str

class DeleteGroupPolicyRequestRequestTypeDef(TypedDict):
    GroupName: str
    PolicyName: str

class DeleteGroupRequestRequestTypeDef(TypedDict):
    GroupName: str

class DeleteInstanceProfileRequestRequestTypeDef(TypedDict):
    InstanceProfileName: str

class DeleteLoginProfileRequestRequestTypeDef(TypedDict):
    UserName: NotRequired[str]

class DeleteOpenIDConnectProviderRequestRequestTypeDef(TypedDict):
    OpenIDConnectProviderArn: str

class DeletePolicyRequestRequestTypeDef(TypedDict):
    PolicyArn: str

class DeletePolicyVersionRequestRequestTypeDef(TypedDict):
    PolicyArn: str
    VersionId: str

class DeleteRolePermissionsBoundaryRequestRequestTypeDef(TypedDict):
    RoleName: str

class DeleteRolePolicyRequestRequestTypeDef(TypedDict):
    RoleName: str
    PolicyName: str

class DeleteRoleRequestRequestTypeDef(TypedDict):
    RoleName: str

class DeleteSAMLProviderRequestRequestTypeDef(TypedDict):
    SAMLProviderArn: str

class DeleteSSHPublicKeyRequestRequestTypeDef(TypedDict):
    UserName: str
    SSHPublicKeyId: str

class DeleteServerCertificateRequestRequestTypeDef(TypedDict):
    ServerCertificateName: str

class DeleteServiceLinkedRoleRequestRequestTypeDef(TypedDict):
    RoleName: str

class DeleteServiceSpecificCredentialRequestRequestTypeDef(TypedDict):
    ServiceSpecificCredentialId: str
    UserName: NotRequired[str]

class DeleteSigningCertificateRequestRequestTypeDef(TypedDict):
    CertificateId: str
    UserName: NotRequired[str]

class DeleteUserPermissionsBoundaryRequestRequestTypeDef(TypedDict):
    UserName: str

class DeleteUserPolicyRequestRequestTypeDef(TypedDict):
    UserName: str
    PolicyName: str

class DeleteUserRequestRequestTypeDef(TypedDict):
    UserName: str

class DeleteVirtualMFADeviceRequestRequestTypeDef(TypedDict):
    SerialNumber: str

class RoleUsageTypeTypeDef(TypedDict):
    Region: NotRequired[str]
    Resources: NotRequired[List[str]]

class DetachGroupPolicyRequestGroupDetachPolicyTypeDef(TypedDict):
    PolicyArn: str

class DetachGroupPolicyRequestPolicyDetachGroupTypeDef(TypedDict):
    GroupName: str

class DetachGroupPolicyRequestRequestTypeDef(TypedDict):
    GroupName: str
    PolicyArn: str

class DetachRolePolicyRequestPolicyDetachRoleTypeDef(TypedDict):
    RoleName: str

class DetachRolePolicyRequestRequestTypeDef(TypedDict):
    RoleName: str
    PolicyArn: str

class DetachRolePolicyRequestRoleDetachPolicyTypeDef(TypedDict):
    PolicyArn: str

class DetachUserPolicyRequestPolicyDetachUserTypeDef(TypedDict):
    UserName: str

class DetachUserPolicyRequestRequestTypeDef(TypedDict):
    UserName: str
    PolicyArn: str

class DetachUserPolicyRequestUserDetachPolicyTypeDef(TypedDict):
    PolicyArn: str

class EnableMFADeviceRequestMfaDeviceAssociateTypeDef(TypedDict):
    AuthenticationCode1: str
    AuthenticationCode2: str

class EnableMFADeviceRequestRequestTypeDef(TypedDict):
    UserName: str
    SerialNumber: str
    AuthenticationCode1: str
    AuthenticationCode2: str

class EnableMFADeviceRequestUserEnableMfaTypeDef(TypedDict):
    SerialNumber: str
    AuthenticationCode1: str
    AuthenticationCode2: str

EntityInfoTypeDef = TypedDict(
    "EntityInfoTypeDef",
    {
        "Arn": str,
        "Name": str,
        "Type": PolicyOwnerEntityTypeType,
        "Id": str,
        "Path": NotRequired[str],
    },
)

class ErrorDetailsTypeDef(TypedDict):
    Message: str
    Code: str

class OrganizationsDecisionDetailTypeDef(TypedDict):
    AllowedByOrganizations: NotRequired[bool]

class PermissionsBoundaryDecisionDetailTypeDef(TypedDict):
    AllowedByPermissionsBoundary: NotRequired[bool]

class GenerateOrganizationsAccessReportRequestRequestTypeDef(TypedDict):
    EntityPath: str
    OrganizationsPolicyId: NotRequired[str]

class GenerateServiceLastAccessedDetailsRequestRequestTypeDef(TypedDict):
    Arn: str
    Granularity: NotRequired[AccessAdvisorUsageGranularityTypeType]

class GetAccessKeyLastUsedRequestRequestTypeDef(TypedDict):
    AccessKeyId: str

class PaginatorConfigTypeDef(TypedDict):
    MaxItems: NotRequired[int]
    PageSize: NotRequired[int]
    StartingToken: NotRequired[str]

class GetAccountAuthorizationDetailsRequestRequestTypeDef(TypedDict):
    Filter: NotRequired[Sequence[EntityTypeType]]
    MaxItems: NotRequired[int]
    Marker: NotRequired[str]

class PasswordPolicyTypeDef(TypedDict):
    MinimumPasswordLength: NotRequired[int]
    RequireSymbols: NotRequired[bool]
    RequireNumbers: NotRequired[bool]
    RequireUppercaseCharacters: NotRequired[bool]
    RequireLowercaseCharacters: NotRequired[bool]
    AllowUsersToChangePassword: NotRequired[bool]
    ExpirePasswords: NotRequired[bool]
    MaxPasswordAge: NotRequired[int]
    PasswordReusePrevention: NotRequired[int]
    HardExpiry: NotRequired[bool]

class GetContextKeysForCustomPolicyRequestRequestTypeDef(TypedDict):
    PolicyInputList: Sequence[str]

class GetContextKeysForPrincipalPolicyRequestRequestTypeDef(TypedDict):
    PolicySourceArn: str
    PolicyInputList: NotRequired[Sequence[str]]

class GetGroupPolicyRequestRequestTypeDef(TypedDict):
    GroupName: str
    PolicyName: str

class GetGroupRequestRequestTypeDef(TypedDict):
    GroupName: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class GetInstanceProfileRequestRequestTypeDef(TypedDict):
    InstanceProfileName: str

class WaiterConfigTypeDef(TypedDict):
    Delay: NotRequired[int]
    MaxAttempts: NotRequired[int]

class GetLoginProfileRequestRequestTypeDef(TypedDict):
    UserName: NotRequired[str]

class GetMFADeviceRequestRequestTypeDef(TypedDict):
    SerialNumber: str
    UserName: NotRequired[str]

class GetOpenIDConnectProviderRequestRequestTypeDef(TypedDict):
    OpenIDConnectProviderArn: str

class GetOrganizationsAccessReportRequestRequestTypeDef(TypedDict):
    JobId: str
    MaxItems: NotRequired[int]
    Marker: NotRequired[str]
    SortKey: NotRequired[SortKeyTypeType]

class GetPolicyRequestRequestTypeDef(TypedDict):
    PolicyArn: str

class GetPolicyVersionRequestRequestTypeDef(TypedDict):
    PolicyArn: str
    VersionId: str

class GetRolePolicyRequestRequestTypeDef(TypedDict):
    RoleName: str
    PolicyName: str

class GetRoleRequestRequestTypeDef(TypedDict):
    RoleName: str

class GetSAMLProviderRequestRequestTypeDef(TypedDict):
    SAMLProviderArn: str

class GetSSHPublicKeyRequestRequestTypeDef(TypedDict):
    UserName: str
    SSHPublicKeyId: str
    Encoding: EncodingTypeType

class SSHPublicKeyTypeDef(TypedDict):
    UserName: str
    SSHPublicKeyId: str
    Fingerprint: str
    SSHPublicKeyBody: str
    Status: StatusTypeType
    UploadDate: NotRequired[datetime]

class GetServerCertificateRequestRequestTypeDef(TypedDict):
    ServerCertificateName: str

class GetServiceLastAccessedDetailsRequestRequestTypeDef(TypedDict):
    JobId: str
    MaxItems: NotRequired[int]
    Marker: NotRequired[str]

class GetServiceLastAccessedDetailsWithEntitiesRequestRequestTypeDef(TypedDict):
    JobId: str
    ServiceNamespace: str
    MaxItems: NotRequired[int]
    Marker: NotRequired[str]

class GetServiceLinkedRoleDeletionStatusRequestRequestTypeDef(TypedDict):
    DeletionTaskId: str

class GetUserPolicyRequestRequestTypeDef(TypedDict):
    UserName: str
    PolicyName: str

class GetUserRequestRequestTypeDef(TypedDict):
    UserName: NotRequired[str]

class ListAccessKeysRequestRequestTypeDef(TypedDict):
    UserName: NotRequired[str]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListAccountAliasesRequestRequestTypeDef(TypedDict):
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListAttachedGroupPoliciesRequestRequestTypeDef(TypedDict):
    GroupName: str
    PathPrefix: NotRequired[str]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListAttachedRolePoliciesRequestRequestTypeDef(TypedDict):
    RoleName: str
    PathPrefix: NotRequired[str]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListAttachedUserPoliciesRequestRequestTypeDef(TypedDict):
    UserName: str
    PathPrefix: NotRequired[str]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListEntitiesForPolicyRequestRequestTypeDef(TypedDict):
    PolicyArn: str
    EntityFilter: NotRequired[EntityTypeType]
    PathPrefix: NotRequired[str]
    PolicyUsageFilter: NotRequired[PolicyUsageTypeType]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class PolicyGroupTypeDef(TypedDict):
    GroupName: NotRequired[str]
    GroupId: NotRequired[str]

class PolicyRoleTypeDef(TypedDict):
    RoleName: NotRequired[str]
    RoleId: NotRequired[str]

class PolicyUserTypeDef(TypedDict):
    UserName: NotRequired[str]
    UserId: NotRequired[str]

class ListGroupPoliciesRequestRequestTypeDef(TypedDict):
    GroupName: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListGroupsForUserRequestRequestTypeDef(TypedDict):
    UserName: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListGroupsRequestRequestTypeDef(TypedDict):
    PathPrefix: NotRequired[str]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListInstanceProfileTagsRequestRequestTypeDef(TypedDict):
    InstanceProfileName: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListInstanceProfilesForRoleRequestRequestTypeDef(TypedDict):
    RoleName: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListInstanceProfilesRequestRequestTypeDef(TypedDict):
    PathPrefix: NotRequired[str]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListMFADeviceTagsRequestRequestTypeDef(TypedDict):
    SerialNumber: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListMFADevicesRequestRequestTypeDef(TypedDict):
    UserName: NotRequired[str]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class MFADeviceTypeDef(TypedDict):
    UserName: str
    SerialNumber: str
    EnableDate: datetime

class ListOpenIDConnectProviderTagsRequestRequestTypeDef(TypedDict):
    OpenIDConnectProviderArn: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class OpenIDConnectProviderListEntryTypeDef(TypedDict):
    Arn: NotRequired[str]

class PolicyGrantingServiceAccessTypeDef(TypedDict):
    PolicyName: str
    PolicyType: PolicyTypeType
    PolicyArn: NotRequired[str]
    EntityType: NotRequired[PolicyOwnerEntityTypeType]
    EntityName: NotRequired[str]

class ListPoliciesGrantingServiceAccessRequestRequestTypeDef(TypedDict):
    Arn: str
    ServiceNamespaces: Sequence[str]
    Marker: NotRequired[str]

class ListPoliciesRequestRequestTypeDef(TypedDict):
    Scope: NotRequired[PolicyScopeTypeType]
    OnlyAttached: NotRequired[bool]
    PathPrefix: NotRequired[str]
    PolicyUsageFilter: NotRequired[PolicyUsageTypeType]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListPolicyTagsRequestRequestTypeDef(TypedDict):
    PolicyArn: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListPolicyVersionsRequestRequestTypeDef(TypedDict):
    PolicyArn: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListRolePoliciesRequestRequestTypeDef(TypedDict):
    RoleName: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListRoleTagsRequestRequestTypeDef(TypedDict):
    RoleName: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListRolesRequestRequestTypeDef(TypedDict):
    PathPrefix: NotRequired[str]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListSAMLProviderTagsRequestRequestTypeDef(TypedDict):
    SAMLProviderArn: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class SAMLProviderListEntryTypeDef(TypedDict):
    Arn: NotRequired[str]
    ValidUntil: NotRequired[datetime]
    CreateDate: NotRequired[datetime]

class ListSSHPublicKeysRequestRequestTypeDef(TypedDict):
    UserName: NotRequired[str]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class SSHPublicKeyMetadataTypeDef(TypedDict):
    UserName: str
    SSHPublicKeyId: str
    Status: StatusTypeType
    UploadDate: datetime

class ListServerCertificateTagsRequestRequestTypeDef(TypedDict):
    ServerCertificateName: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListServerCertificatesRequestRequestTypeDef(TypedDict):
    PathPrefix: NotRequired[str]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ServerCertificateMetadataTypeDef(TypedDict):
    Path: str
    ServerCertificateName: str
    ServerCertificateId: str
    Arn: str
    UploadDate: NotRequired[datetime]
    Expiration: NotRequired[datetime]

ListServiceSpecificCredentialsRequestRequestTypeDef = TypedDict(
    "ListServiceSpecificCredentialsRequestRequestTypeDef",
    {
        "UserName": NotRequired[str],
        "ServiceName": NotRequired[str],
    },
)
ServiceSpecificCredentialMetadataTypeDef = TypedDict(
    "ServiceSpecificCredentialMetadataTypeDef",
    {
        "UserName": str,
        "Status": StatusTypeType,
        "ServiceUserName": str,
        "CreateDate": datetime,
        "ServiceSpecificCredentialId": str,
        "ServiceName": str,
    },
)

class ListSigningCertificatesRequestRequestTypeDef(TypedDict):
    UserName: NotRequired[str]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class SigningCertificateTypeDef(TypedDict):
    UserName: str
    CertificateId: str
    CertificateBody: str
    Status: StatusTypeType
    UploadDate: NotRequired[datetime]

class ListUserPoliciesRequestRequestTypeDef(TypedDict):
    UserName: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListUserTagsRequestRequestTypeDef(TypedDict):
    UserName: str
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListUsersRequestRequestTypeDef(TypedDict):
    PathPrefix: NotRequired[str]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class ListVirtualMFADevicesRequestRequestTypeDef(TypedDict):
    AssignmentStatus: NotRequired[AssignmentStatusTypeType]
    Marker: NotRequired[str]
    MaxItems: NotRequired[int]

class PolicyDocumentStatementTypeDef(TypedDict):
    Effect: str
    Resource: str | List[str]
    Sid: str
    Action: str | List[str]

class PositionTypeDef(TypedDict):
    Line: NotRequired[int]
    Column: NotRequired[int]

class PutGroupPolicyRequestGroupCreatePolicyTypeDef(TypedDict):
    PolicyName: str
    PolicyDocument: str

class PutGroupPolicyRequestGroupPolicyPutTypeDef(TypedDict):
    PolicyDocument: str

class PutGroupPolicyRequestRequestTypeDef(TypedDict):
    GroupName: str
    PolicyName: str
    PolicyDocument: str

class PutRolePermissionsBoundaryRequestRequestTypeDef(TypedDict):
    RoleName: str
    PermissionsBoundary: str

class PutRolePolicyRequestRequestTypeDef(TypedDict):
    RoleName: str
    PolicyName: str
    PolicyDocument: str

class PutRolePolicyRequestRolePolicyPutTypeDef(TypedDict):
    PolicyDocument: str

class PutUserPermissionsBoundaryRequestRequestTypeDef(TypedDict):
    UserName: str
    PermissionsBoundary: str

class PutUserPolicyRequestRequestTypeDef(TypedDict):
    UserName: str
    PolicyName: str
    PolicyDocument: str

class PutUserPolicyRequestUserCreatePolicyTypeDef(TypedDict):
    PolicyName: str
    PolicyDocument: str

class PutUserPolicyRequestUserPolicyPutTypeDef(TypedDict):
    PolicyDocument: str

class RemoveClientIDFromOpenIDConnectProviderRequestRequestTypeDef(TypedDict):
    OpenIDConnectProviderArn: str
    ClientID: str

class RemoveRoleFromInstanceProfileRequestInstanceProfileRemoveRoleTypeDef(TypedDict):
    RoleName: str

class RemoveRoleFromInstanceProfileRequestRequestTypeDef(TypedDict):
    InstanceProfileName: str
    RoleName: str

class RemoveUserFromGroupRequestGroupRemoveUserTypeDef(TypedDict):
    UserName: str

class RemoveUserFromGroupRequestRequestTypeDef(TypedDict):
    GroupName: str
    UserName: str

class RemoveUserFromGroupRequestUserRemoveGroupTypeDef(TypedDict):
    GroupName: str

class ResetServiceSpecificCredentialRequestRequestTypeDef(TypedDict):
    ServiceSpecificCredentialId: str
    UserName: NotRequired[str]

class ResyncMFADeviceRequestMfaDeviceResyncTypeDef(TypedDict):
    AuthenticationCode1: str
    AuthenticationCode2: str

class ResyncMFADeviceRequestRequestTypeDef(TypedDict):
    UserName: str
    SerialNumber: str
    AuthenticationCode1: str
    AuthenticationCode2: str

class RoleLastUsedTypeDef(TypedDict):
    LastUsedDate: NotRequired[datetime]
    Region: NotRequired[str]

class TrackedActionLastAccessedTypeDef(TypedDict):
    ActionName: NotRequired[str]
    LastAccessedEntity: NotRequired[str]
    LastAccessedTime: NotRequired[datetime]
    LastAccessedRegion: NotRequired[str]

class SetDefaultPolicyVersionRequestRequestTypeDef(TypedDict):
    PolicyArn: str
    VersionId: str

class SetSecurityTokenServicePreferencesRequestRequestTypeDef(TypedDict):
    GlobalEndpointTokenVersion: GlobalEndpointTokenVersionType

class UntagInstanceProfileRequestRequestTypeDef(TypedDict):
    InstanceProfileName: str
    TagKeys: Sequence[str]

class UntagMFADeviceRequestRequestTypeDef(TypedDict):
    SerialNumber: str
    TagKeys: Sequence[str]

class UntagOpenIDConnectProviderRequestRequestTypeDef(TypedDict):
    OpenIDConnectProviderArn: str
    TagKeys: Sequence[str]

class UntagPolicyRequestRequestTypeDef(TypedDict):
    PolicyArn: str
    TagKeys: Sequence[str]

class UntagRoleRequestRequestTypeDef(TypedDict):
    RoleName: str
    TagKeys: Sequence[str]

class UntagSAMLProviderRequestRequestTypeDef(TypedDict):
    SAMLProviderArn: str
    TagKeys: Sequence[str]

class UntagServerCertificateRequestRequestTypeDef(TypedDict):
    ServerCertificateName: str
    TagKeys: Sequence[str]

class UntagUserRequestRequestTypeDef(TypedDict):
    UserName: str
    TagKeys: Sequence[str]

class UpdateAccessKeyRequestAccessKeyActivateTypeDef(TypedDict):
    Status: NotRequired[StatusTypeType]

class UpdateAccessKeyRequestAccessKeyDeactivateTypeDef(TypedDict):
    Status: NotRequired[StatusTypeType]

class UpdateAccessKeyRequestAccessKeyPairActivateTypeDef(TypedDict):
    Status: NotRequired[StatusTypeType]

class UpdateAccessKeyRequestAccessKeyPairDeactivateTypeDef(TypedDict):
    Status: NotRequired[StatusTypeType]

class UpdateAccessKeyRequestRequestTypeDef(TypedDict):
    AccessKeyId: str
    Status: StatusTypeType
    UserName: NotRequired[str]

class UpdateAccountPasswordPolicyRequestAccountPasswordPolicyUpdateTypeDef(TypedDict):
    MinimumPasswordLength: NotRequired[int]
    RequireSymbols: NotRequired[bool]
    RequireNumbers: NotRequired[bool]
    RequireUppercaseCharacters: NotRequired[bool]
    RequireLowercaseCharacters: NotRequired[bool]
    AllowUsersToChangePassword: NotRequired[bool]
    MaxPasswordAge: NotRequired[int]
    PasswordReusePrevention: NotRequired[int]
    HardExpiry: NotRequired[bool]

class UpdateAccountPasswordPolicyRequestRequestTypeDef(TypedDict):
    MinimumPasswordLength: NotRequired[int]
    RequireSymbols: NotRequired[bool]
    RequireNumbers: NotRequired[bool]
    RequireUppercaseCharacters: NotRequired[bool]
    RequireLowercaseCharacters: NotRequired[bool]
    AllowUsersToChangePassword: NotRequired[bool]
    MaxPasswordAge: NotRequired[int]
    PasswordReusePrevention: NotRequired[int]
    HardExpiry: NotRequired[bool]

class UpdateAccountPasswordPolicyRequestServiceResourceCreateAccountPasswordPolicyTypeDef(
    TypedDict
):
    MinimumPasswordLength: NotRequired[int]
    RequireSymbols: NotRequired[bool]
    RequireNumbers: NotRequired[bool]
    RequireUppercaseCharacters: NotRequired[bool]
    RequireLowercaseCharacters: NotRequired[bool]
    AllowUsersToChangePassword: NotRequired[bool]
    MaxPasswordAge: NotRequired[int]
    PasswordReusePrevention: NotRequired[int]
    HardExpiry: NotRequired[bool]

class UpdateAssumeRolePolicyRequestAssumeRolePolicyUpdateTypeDef(TypedDict):
    PolicyDocument: str

class UpdateAssumeRolePolicyRequestRequestTypeDef(TypedDict):
    RoleName: str
    PolicyDocument: str

class UpdateGroupRequestGroupUpdateTypeDef(TypedDict):
    NewPath: NotRequired[str]
    NewGroupName: NotRequired[str]

class UpdateGroupRequestRequestTypeDef(TypedDict):
    GroupName: str
    NewPath: NotRequired[str]
    NewGroupName: NotRequired[str]

class UpdateLoginProfileRequestLoginProfileUpdateTypeDef(TypedDict):
    Password: NotRequired[str]
    PasswordResetRequired: NotRequired[bool]

class UpdateLoginProfileRequestRequestTypeDef(TypedDict):
    UserName: str
    Password: NotRequired[str]
    PasswordResetRequired: NotRequired[bool]

class UpdateOpenIDConnectProviderThumbprintRequestRequestTypeDef(TypedDict):
    OpenIDConnectProviderArn: str
    ThumbprintList: Sequence[str]

class UpdateRoleDescriptionRequestRequestTypeDef(TypedDict):
    RoleName: str
    Description: str

class UpdateRoleRequestRequestTypeDef(TypedDict):
    RoleName: str
    Description: NotRequired[str]
    MaxSessionDuration: NotRequired[int]

class UpdateSAMLProviderRequestRequestTypeDef(TypedDict):
    SAMLMetadataDocument: str
    SAMLProviderArn: str

class UpdateSAMLProviderRequestSamlProviderUpdateTypeDef(TypedDict):
    SAMLMetadataDocument: str

class UpdateSSHPublicKeyRequestRequestTypeDef(TypedDict):
    UserName: str
    SSHPublicKeyId: str
    Status: StatusTypeType

class UpdateServerCertificateRequestRequestTypeDef(TypedDict):
    ServerCertificateName: str
    NewPath: NotRequired[str]
    NewServerCertificateName: NotRequired[str]

class UpdateServerCertificateRequestServerCertificateUpdateTypeDef(TypedDict):
    NewPath: NotRequired[str]
    NewServerCertificateName: NotRequired[str]

class UpdateServiceSpecificCredentialRequestRequestTypeDef(TypedDict):
    ServiceSpecificCredentialId: str
    Status: StatusTypeType
    UserName: NotRequired[str]

class UpdateSigningCertificateRequestRequestTypeDef(TypedDict):
    CertificateId: str
    Status: StatusTypeType
    UserName: NotRequired[str]

class UpdateSigningCertificateRequestSigningCertificateActivateTypeDef(TypedDict):
    Status: NotRequired[StatusTypeType]

class UpdateSigningCertificateRequestSigningCertificateDeactivateTypeDef(TypedDict):
    Status: NotRequired[StatusTypeType]

class UpdateUserRequestRequestTypeDef(TypedDict):
    UserName: str
    NewPath: NotRequired[str]
    NewUserName: NotRequired[str]

class UpdateUserRequestUserUpdateTypeDef(TypedDict):
    NewPath: NotRequired[str]
    NewUserName: NotRequired[str]

class UploadSSHPublicKeyRequestRequestTypeDef(TypedDict):
    UserName: str
    SSHPublicKeyBody: str

class UploadSigningCertificateRequestRequestTypeDef(TypedDict):
    CertificateBody: str
    UserName: NotRequired[str]

class UploadSigningCertificateRequestServiceResourceCreateSigningCertificateTypeDef(TypedDict):
    CertificateBody: str
    UserName: NotRequired[str]

class SimulateCustomPolicyRequestRequestTypeDef(TypedDict):
    PolicyInputList: Sequence[str]
    ActionNames: Sequence[str]
    PermissionsBoundaryPolicyInputList: NotRequired[Sequence[str]]
    ResourceArns: NotRequired[Sequence[str]]
    ResourcePolicy: NotRequired[str]
    ResourceOwner: NotRequired[str]
    CallerArn: NotRequired[str]
    ContextEntries: NotRequired[Sequence[ContextEntryTypeDef]]
    ResourceHandlingOption: NotRequired[str]
    MaxItems: NotRequired[int]
    Marker: NotRequired[str]

class SimulatePrincipalPolicyRequestRequestTypeDef(TypedDict):
    PolicySourceArn: str
    ActionNames: Sequence[str]
    PolicyInputList: NotRequired[Sequence[str]]
    PermissionsBoundaryPolicyInputList: NotRequired[Sequence[str]]
    ResourceArns: NotRequired[Sequence[str]]
    ResourcePolicy: NotRequired[str]
    ResourceOwner: NotRequired[str]
    CallerArn: NotRequired[str]
    ContextEntries: NotRequired[Sequence[ContextEntryTypeDef]]
    ResourceHandlingOption: NotRequired[str]
    MaxItems: NotRequired[int]
    Marker: NotRequired[str]

class CreateAccessKeyResponseTypeDef(TypedDict):
    AccessKey: AccessKeyTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class DeleteServiceLinkedRoleResponseTypeDef(TypedDict):
    DeletionTaskId: str
    ResponseMetadata: ResponseMetadataTypeDef

class DisableOrganizationsRootCredentialsManagementResponseTypeDef(TypedDict):
    OrganizationId: str
    EnabledFeatures: List[FeatureTypeType]
    ResponseMetadata: ResponseMetadataTypeDef

class DisableOrganizationsRootSessionsResponseTypeDef(TypedDict):
    OrganizationId: str
    EnabledFeatures: List[FeatureTypeType]
    ResponseMetadata: ResponseMetadataTypeDef

class EmptyResponseMetadataTypeDef(TypedDict):
    ResponseMetadata: ResponseMetadataTypeDef

class EnableOrganizationsRootCredentialsManagementResponseTypeDef(TypedDict):
    OrganizationId: str
    EnabledFeatures: List[FeatureTypeType]
    ResponseMetadata: ResponseMetadataTypeDef

class EnableOrganizationsRootSessionsResponseTypeDef(TypedDict):
    OrganizationId: str
    EnabledFeatures: List[FeatureTypeType]
    ResponseMetadata: ResponseMetadataTypeDef

class GenerateCredentialReportResponseTypeDef(TypedDict):
    State: ReportStateTypeType
    Description: str
    ResponseMetadata: ResponseMetadataTypeDef

class GenerateOrganizationsAccessReportResponseTypeDef(TypedDict):
    JobId: str
    ResponseMetadata: ResponseMetadataTypeDef

class GenerateServiceLastAccessedDetailsResponseTypeDef(TypedDict):
    JobId: str
    ResponseMetadata: ResponseMetadataTypeDef

class GetAccessKeyLastUsedResponseTypeDef(TypedDict):
    UserName: str
    AccessKeyLastUsed: AccessKeyLastUsedTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class GetAccountSummaryResponseTypeDef(TypedDict):
    SummaryMap: Dict[SummaryKeyTypeType, int]
    ResponseMetadata: ResponseMetadataTypeDef

class GetContextKeysForPolicyResponseTypeDef(TypedDict):
    ContextKeyNames: List[str]
    ResponseMetadata: ResponseMetadataTypeDef

class GetCredentialReportResponseTypeDef(TypedDict):
    Content: bytes
    ReportFormat: Literal["text/csv"]
    GeneratedTime: datetime
    ResponseMetadata: ResponseMetadataTypeDef

class GetMFADeviceResponseTypeDef(TypedDict):
    UserName: str
    SerialNumber: str
    EnableDate: datetime
    Certifications: Dict[str, str]
    ResponseMetadata: ResponseMetadataTypeDef

class ListAccessKeysResponseTypeDef(TypedDict):
    AccessKeyMetadata: List[AccessKeyMetadataTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListAccountAliasesResponseTypeDef(TypedDict):
    AccountAliases: List[str]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListAttachedGroupPoliciesResponseTypeDef(TypedDict):
    AttachedPolicies: List[AttachedPolicyTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListAttachedRolePoliciesResponseTypeDef(TypedDict):
    AttachedPolicies: List[AttachedPolicyTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListAttachedUserPoliciesResponseTypeDef(TypedDict):
    AttachedPolicies: List[AttachedPolicyTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListGroupPoliciesResponseTypeDef(TypedDict):
    PolicyNames: List[str]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListOrganizationsFeaturesResponseTypeDef(TypedDict):
    OrganizationId: str
    EnabledFeatures: List[FeatureTypeType]
    ResponseMetadata: ResponseMetadataTypeDef

class ListRolePoliciesResponseTypeDef(TypedDict):
    PolicyNames: List[str]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListUserPoliciesResponseTypeDef(TypedDict):
    PolicyNames: List[str]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class UpdateSAMLProviderResponseTypeDef(TypedDict):
    SAMLProviderArn: str
    ResponseMetadata: ResponseMetadataTypeDef

class CreateGroupResponseTypeDef(TypedDict):
    Group: GroupTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class ListGroupsForUserResponseTypeDef(TypedDict):
    Groups: List[GroupTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListGroupsResponseTypeDef(TypedDict):
    Groups: List[GroupTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class CreateInstanceProfileRequestRequestTypeDef(TypedDict):
    InstanceProfileName: str
    Path: NotRequired[str]
    Tags: NotRequired[Sequence[TagTypeDef]]

class CreateInstanceProfileRequestServiceResourceCreateInstanceProfileTypeDef(TypedDict):
    InstanceProfileName: str
    Path: NotRequired[str]
    Tags: NotRequired[Sequence[TagTypeDef]]

class CreateOpenIDConnectProviderRequestRequestTypeDef(TypedDict):
    Url: str
    ClientIDList: NotRequired[Sequence[str]]
    ThumbprintList: NotRequired[Sequence[str]]
    Tags: NotRequired[Sequence[TagTypeDef]]

class CreateOpenIDConnectProviderResponseTypeDef(TypedDict):
    OpenIDConnectProviderArn: str
    Tags: List[TagTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef

class CreatePolicyRequestRequestTypeDef(TypedDict):
    PolicyName: str
    PolicyDocument: str
    Path: NotRequired[str]
    Description: NotRequired[str]
    Tags: NotRequired[Sequence[TagTypeDef]]

class CreatePolicyRequestServiceResourceCreatePolicyTypeDef(TypedDict):
    PolicyName: str
    PolicyDocument: str
    Path: NotRequired[str]
    Description: NotRequired[str]
    Tags: NotRequired[Sequence[TagTypeDef]]

class CreateRoleRequestRequestTypeDef(TypedDict):
    RoleName: str
    AssumeRolePolicyDocument: str
    Path: NotRequired[str]
    Description: NotRequired[str]
    MaxSessionDuration: NotRequired[int]
    PermissionsBoundary: NotRequired[str]
    Tags: NotRequired[Sequence[TagTypeDef]]

class CreateRoleRequestServiceResourceCreateRoleTypeDef(TypedDict):
    RoleName: str
    AssumeRolePolicyDocument: str
    Path: NotRequired[str]
    Description: NotRequired[str]
    MaxSessionDuration: NotRequired[int]
    PermissionsBoundary: NotRequired[str]
    Tags: NotRequired[Sequence[TagTypeDef]]

class CreateSAMLProviderRequestRequestTypeDef(TypedDict):
    SAMLMetadataDocument: str
    Name: str
    Tags: NotRequired[Sequence[TagTypeDef]]

class CreateSAMLProviderRequestServiceResourceCreateSamlProviderTypeDef(TypedDict):
    SAMLMetadataDocument: str
    Name: str
    Tags: NotRequired[Sequence[TagTypeDef]]

class CreateSAMLProviderResponseTypeDef(TypedDict):
    SAMLProviderArn: str
    Tags: List[TagTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef

class CreateUserRequestRequestTypeDef(TypedDict):
    UserName: str
    Path: NotRequired[str]
    PermissionsBoundary: NotRequired[str]
    Tags: NotRequired[Sequence[TagTypeDef]]

class CreateUserRequestServiceResourceCreateUserTypeDef(TypedDict):
    UserName: str
    Path: NotRequired[str]
    PermissionsBoundary: NotRequired[str]
    Tags: NotRequired[Sequence[TagTypeDef]]

class CreateUserRequestUserCreateTypeDef(TypedDict):
    Path: NotRequired[str]
    PermissionsBoundary: NotRequired[str]
    Tags: NotRequired[Sequence[TagTypeDef]]

class CreateVirtualMFADeviceRequestRequestTypeDef(TypedDict):
    VirtualMFADeviceName: str
    Path: NotRequired[str]
    Tags: NotRequired[Sequence[TagTypeDef]]

class CreateVirtualMFADeviceRequestServiceResourceCreateVirtualMfaDeviceTypeDef(TypedDict):
    VirtualMFADeviceName: str
    Path: NotRequired[str]
    Tags: NotRequired[Sequence[TagTypeDef]]

class GetOpenIDConnectProviderResponseTypeDef(TypedDict):
    Url: str
    ClientIDList: List[str]
    ThumbprintList: List[str]
    CreateDate: datetime
    Tags: List[TagTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef

class GetSAMLProviderResponseTypeDef(TypedDict):
    SAMLMetadataDocument: str
    CreateDate: datetime
    ValidUntil: datetime
    Tags: List[TagTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef

class ListInstanceProfileTagsResponseTypeDef(TypedDict):
    Tags: List[TagTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListMFADeviceTagsResponseTypeDef(TypedDict):
    Tags: List[TagTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListOpenIDConnectProviderTagsResponseTypeDef(TypedDict):
    Tags: List[TagTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListPolicyTagsResponseTypeDef(TypedDict):
    Tags: List[TagTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListRoleTagsResponseTypeDef(TypedDict):
    Tags: List[TagTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListSAMLProviderTagsResponseTypeDef(TypedDict):
    Tags: List[TagTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListServerCertificateTagsResponseTypeDef(TypedDict):
    Tags: List[TagTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListUserTagsResponseTypeDef(TypedDict):
    Tags: List[TagTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class PolicyTypeDef(TypedDict):
    PolicyName: NotRequired[str]
    PolicyId: NotRequired[str]
    Arn: NotRequired[str]
    Path: NotRequired[str]
    DefaultVersionId: NotRequired[str]
    AttachmentCount: NotRequired[int]
    PermissionsBoundaryUsageCount: NotRequired[int]
    IsAttachable: NotRequired[bool]
    Description: NotRequired[str]
    CreateDate: NotRequired[datetime]
    UpdateDate: NotRequired[datetime]
    Tags: NotRequired[List[TagTypeDef]]

class TagInstanceProfileRequestRequestTypeDef(TypedDict):
    InstanceProfileName: str
    Tags: Sequence[TagTypeDef]

class TagMFADeviceRequestRequestTypeDef(TypedDict):
    SerialNumber: str
    Tags: Sequence[TagTypeDef]

class TagOpenIDConnectProviderRequestRequestTypeDef(TypedDict):
    OpenIDConnectProviderArn: str
    Tags: Sequence[TagTypeDef]

class TagPolicyRequestRequestTypeDef(TypedDict):
    PolicyArn: str
    Tags: Sequence[TagTypeDef]

class TagRoleRequestRequestTypeDef(TypedDict):
    RoleName: str
    Tags: Sequence[TagTypeDef]

class TagSAMLProviderRequestRequestTypeDef(TypedDict):
    SAMLProviderArn: str
    Tags: Sequence[TagTypeDef]

class TagServerCertificateRequestRequestTypeDef(TypedDict):
    ServerCertificateName: str
    Tags: Sequence[TagTypeDef]

class TagUserRequestRequestTypeDef(TypedDict):
    UserName: str
    Tags: Sequence[TagTypeDef]

class UploadServerCertificateRequestRequestTypeDef(TypedDict):
    ServerCertificateName: str
    CertificateBody: str
    PrivateKey: str
    Path: NotRequired[str]
    CertificateChain: NotRequired[str]
    Tags: NotRequired[Sequence[TagTypeDef]]

class UploadServerCertificateRequestServiceResourceCreateServerCertificateTypeDef(TypedDict):
    ServerCertificateName: str
    CertificateBody: str
    PrivateKey: str
    Path: NotRequired[str]
    CertificateChain: NotRequired[str]
    Tags: NotRequired[Sequence[TagTypeDef]]

class UserTypeDef(TypedDict):
    Path: str
    UserName: str
    UserId: str
    Arn: str
    CreateDate: datetime
    PasswordLastUsed: NotRequired[datetime]
    PermissionsBoundary: NotRequired[AttachedPermissionsBoundaryTypeDef]
    Tags: NotRequired[List[TagTypeDef]]

class CreateLoginProfileResponseTypeDef(TypedDict):
    LoginProfile: LoginProfileTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class GetLoginProfileResponseTypeDef(TypedDict):
    LoginProfile: LoginProfileTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class CreateServiceSpecificCredentialResponseTypeDef(TypedDict):
    ServiceSpecificCredential: ServiceSpecificCredentialTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class ResetServiceSpecificCredentialResponseTypeDef(TypedDict):
    ServiceSpecificCredential: ServiceSpecificCredentialTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class DeletionTaskFailureReasonTypeTypeDef(TypedDict):
    Reason: NotRequired[str]
    RoleUsageList: NotRequired[List[RoleUsageTypeTypeDef]]

class EntityDetailsTypeDef(TypedDict):
    EntityInfo: EntityInfoTypeDef
    LastAuthenticated: NotRequired[datetime]

class GetOrganizationsAccessReportResponseTypeDef(TypedDict):
    JobStatus: JobStatusTypeType
    JobCreationDate: datetime
    JobCompletionDate: datetime
    NumberOfServicesAccessible: int
    NumberOfServicesNotAccessed: int
    AccessDetails: List[AccessDetailTypeDef]
    IsTruncated: bool
    Marker: str
    ErrorDetails: ErrorDetailsTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class GetAccountAuthorizationDetailsRequestPaginateTypeDef(TypedDict):
    Filter: NotRequired[Sequence[EntityTypeType]]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class GetGroupRequestPaginateTypeDef(TypedDict):
    GroupName: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListAccessKeysRequestPaginateTypeDef(TypedDict):
    UserName: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListAccountAliasesRequestPaginateTypeDef(TypedDict):
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListAttachedGroupPoliciesRequestPaginateTypeDef(TypedDict):
    GroupName: str
    PathPrefix: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListAttachedRolePoliciesRequestPaginateTypeDef(TypedDict):
    RoleName: str
    PathPrefix: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListAttachedUserPoliciesRequestPaginateTypeDef(TypedDict):
    UserName: str
    PathPrefix: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListEntitiesForPolicyRequestPaginateTypeDef(TypedDict):
    PolicyArn: str
    EntityFilter: NotRequired[EntityTypeType]
    PathPrefix: NotRequired[str]
    PolicyUsageFilter: NotRequired[PolicyUsageTypeType]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListGroupPoliciesRequestPaginateTypeDef(TypedDict):
    GroupName: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListGroupsForUserRequestPaginateTypeDef(TypedDict):
    UserName: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListGroupsRequestPaginateTypeDef(TypedDict):
    PathPrefix: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListInstanceProfileTagsRequestPaginateTypeDef(TypedDict):
    InstanceProfileName: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListInstanceProfilesForRoleRequestPaginateTypeDef(TypedDict):
    RoleName: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListInstanceProfilesRequestPaginateTypeDef(TypedDict):
    PathPrefix: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListMFADeviceTagsRequestPaginateTypeDef(TypedDict):
    SerialNumber: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListMFADevicesRequestPaginateTypeDef(TypedDict):
    UserName: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListOpenIDConnectProviderTagsRequestPaginateTypeDef(TypedDict):
    OpenIDConnectProviderArn: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListPoliciesRequestPaginateTypeDef(TypedDict):
    Scope: NotRequired[PolicyScopeTypeType]
    OnlyAttached: NotRequired[bool]
    PathPrefix: NotRequired[str]
    PolicyUsageFilter: NotRequired[PolicyUsageTypeType]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListPolicyTagsRequestPaginateTypeDef(TypedDict):
    PolicyArn: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListPolicyVersionsRequestPaginateTypeDef(TypedDict):
    PolicyArn: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListRolePoliciesRequestPaginateTypeDef(TypedDict):
    RoleName: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListRoleTagsRequestPaginateTypeDef(TypedDict):
    RoleName: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListRolesRequestPaginateTypeDef(TypedDict):
    PathPrefix: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListSAMLProviderTagsRequestPaginateTypeDef(TypedDict):
    SAMLProviderArn: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListSSHPublicKeysRequestPaginateTypeDef(TypedDict):
    UserName: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListServerCertificateTagsRequestPaginateTypeDef(TypedDict):
    ServerCertificateName: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListServerCertificatesRequestPaginateTypeDef(TypedDict):
    PathPrefix: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListSigningCertificatesRequestPaginateTypeDef(TypedDict):
    UserName: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListUserPoliciesRequestPaginateTypeDef(TypedDict):
    UserName: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListUserTagsRequestPaginateTypeDef(TypedDict):
    UserName: str
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListUsersRequestPaginateTypeDef(TypedDict):
    PathPrefix: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class ListVirtualMFADevicesRequestPaginateTypeDef(TypedDict):
    AssignmentStatus: NotRequired[AssignmentStatusTypeType]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class SimulateCustomPolicyRequestPaginateTypeDef(TypedDict):
    PolicyInputList: Sequence[str]
    ActionNames: Sequence[str]
    PermissionsBoundaryPolicyInputList: NotRequired[Sequence[str]]
    ResourceArns: NotRequired[Sequence[str]]
    ResourcePolicy: NotRequired[str]
    ResourceOwner: NotRequired[str]
    CallerArn: NotRequired[str]
    ContextEntries: NotRequired[Sequence[ContextEntryTypeDef]]
    ResourceHandlingOption: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class SimulatePrincipalPolicyRequestPaginateTypeDef(TypedDict):
    PolicySourceArn: str
    ActionNames: Sequence[str]
    PolicyInputList: NotRequired[Sequence[str]]
    PermissionsBoundaryPolicyInputList: NotRequired[Sequence[str]]
    ResourceArns: NotRequired[Sequence[str]]
    ResourcePolicy: NotRequired[str]
    ResourceOwner: NotRequired[str]
    CallerArn: NotRequired[str]
    ContextEntries: NotRequired[Sequence[ContextEntryTypeDef]]
    ResourceHandlingOption: NotRequired[str]
    PaginationConfig: NotRequired[PaginatorConfigTypeDef]

class GetAccountPasswordPolicyResponseTypeDef(TypedDict):
    PasswordPolicy: PasswordPolicyTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class GetInstanceProfileRequestWaitTypeDef(TypedDict):
    InstanceProfileName: str
    WaiterConfig: NotRequired[WaiterConfigTypeDef]

class GetPolicyRequestWaitTypeDef(TypedDict):
    PolicyArn: str
    WaiterConfig: NotRequired[WaiterConfigTypeDef]

class GetRoleRequestWaitTypeDef(TypedDict):
    RoleName: str
    WaiterConfig: NotRequired[WaiterConfigTypeDef]

class GetUserRequestWaitTypeDef(TypedDict):
    UserName: NotRequired[str]
    WaiterConfig: NotRequired[WaiterConfigTypeDef]

class GetSSHPublicKeyResponseTypeDef(TypedDict):
    SSHPublicKey: SSHPublicKeyTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class UploadSSHPublicKeyResponseTypeDef(TypedDict):
    SSHPublicKey: SSHPublicKeyTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class ListEntitiesForPolicyResponseTypeDef(TypedDict):
    PolicyGroups: List[PolicyGroupTypeDef]
    PolicyUsers: List[PolicyUserTypeDef]
    PolicyRoles: List[PolicyRoleTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListMFADevicesResponseTypeDef(TypedDict):
    MFADevices: List[MFADeviceTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListOpenIDConnectProvidersResponseTypeDef(TypedDict):
    OpenIDConnectProviderList: List[OpenIDConnectProviderListEntryTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef

class ListPoliciesGrantingServiceAccessEntryTypeDef(TypedDict):
    ServiceNamespace: NotRequired[str]
    Policies: NotRequired[List[PolicyGrantingServiceAccessTypeDef]]

class ListSAMLProvidersResponseTypeDef(TypedDict):
    SAMLProviderList: List[SAMLProviderListEntryTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef

class ListSSHPublicKeysResponseTypeDef(TypedDict):
    SSHPublicKeys: List[SSHPublicKeyMetadataTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListServerCertificatesResponseTypeDef(TypedDict):
    ServerCertificateMetadataList: List[ServerCertificateMetadataTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ServerCertificateTypeDef(TypedDict):
    ServerCertificateMetadata: ServerCertificateMetadataTypeDef
    CertificateBody: str
    CertificateChain: NotRequired[str]
    Tags: NotRequired[List[TagTypeDef]]

class UploadServerCertificateResponseTypeDef(TypedDict):
    ServerCertificateMetadata: ServerCertificateMetadataTypeDef
    Tags: List[TagTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef

class ListServiceSpecificCredentialsResponseTypeDef(TypedDict):
    ServiceSpecificCredentials: List[ServiceSpecificCredentialMetadataTypeDef]
    ResponseMetadata: ResponseMetadataTypeDef

class ListSigningCertificatesResponseTypeDef(TypedDict):
    Certificates: List[SigningCertificateTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class UploadSigningCertificateResponseTypeDef(TypedDict):
    Certificate: SigningCertificateTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class PolicyDocumentDictTypeDef(TypedDict):
    Version: str
    Statement: List[PolicyDocumentStatementTypeDef]

class StatementTypeDef(TypedDict):
    SourcePolicyId: NotRequired[str]
    SourcePolicyType: NotRequired[PolicySourceTypeType]
    StartPosition: NotRequired[PositionTypeDef]
    EndPosition: NotRequired[PositionTypeDef]

ServiceLastAccessedTypeDef = TypedDict(
    "ServiceLastAccessedTypeDef",
    {
        "ServiceName": str,
        "ServiceNamespace": str,
        "LastAuthenticated": NotRequired[datetime],
        "LastAuthenticatedEntity": NotRequired[str],
        "LastAuthenticatedRegion": NotRequired[str],
        "TotalAuthenticatedEntities": NotRequired[int],
        "TrackedActionsLastAccessed": NotRequired[List[TrackedActionLastAccessedTypeDef]],
    },
)

class CreatePolicyResponseTypeDef(TypedDict):
    Policy: PolicyTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class GetPolicyResponseTypeDef(TypedDict):
    Policy: PolicyTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class ListPoliciesResponseTypeDef(TypedDict):
    Policies: List[PolicyTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class CreateUserResponseTypeDef(TypedDict):
    User: UserTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class GetGroupResponseTypeDef(TypedDict):
    Group: GroupTypeDef
    Users: List[UserTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class GetUserResponseTypeDef(TypedDict):
    User: UserTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class ListUsersResponseTypeDef(TypedDict):
    Users: List[UserTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class VirtualMFADeviceTypeDef(TypedDict):
    SerialNumber: str
    Base32StringSeed: NotRequired[bytes]
    QRCodePNG: NotRequired[bytes]
    User: NotRequired[UserTypeDef]
    EnableDate: NotRequired[datetime]
    Tags: NotRequired[List[TagTypeDef]]

class GetServiceLinkedRoleDeletionStatusResponseTypeDef(TypedDict):
    Status: DeletionTaskStatusTypeType
    Reason: DeletionTaskFailureReasonTypeTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class GetServiceLastAccessedDetailsWithEntitiesResponseTypeDef(TypedDict):
    JobStatus: JobStatusTypeType
    JobCreationDate: datetime
    JobCompletionDate: datetime
    EntityDetailsList: List[EntityDetailsTypeDef]
    IsTruncated: bool
    Marker: str
    Error: ErrorDetailsTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class ListPoliciesGrantingServiceAccessResponseTypeDef(TypedDict):
    PoliciesGrantingServiceAccess: List[ListPoliciesGrantingServiceAccessEntryTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class GetServerCertificateResponseTypeDef(TypedDict):
    ServerCertificate: ServerCertificateTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

PolicyDocumentTypeDef = Union[str, PolicyDocumentDictTypeDef]

class ResourceSpecificResultTypeDef(TypedDict):
    EvalResourceName: str
    EvalResourceDecision: PolicyEvaluationDecisionTypeType
    MatchedStatements: NotRequired[List[StatementTypeDef]]
    MissingContextValues: NotRequired[List[str]]
    EvalDecisionDetails: NotRequired[Dict[str, PolicyEvaluationDecisionTypeType]]
    PermissionsBoundaryDecisionDetail: NotRequired[PermissionsBoundaryDecisionDetailTypeDef]

class GetServiceLastAccessedDetailsResponseTypeDef(TypedDict):
    JobStatus: JobStatusTypeType
    JobType: AccessAdvisorUsageGranularityTypeType
    JobCreationDate: datetime
    ServicesLastAccessed: List[ServiceLastAccessedTypeDef]
    JobCompletionDate: datetime
    IsTruncated: bool
    Marker: str
    Error: ErrorDetailsTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class CreateVirtualMFADeviceResponseTypeDef(TypedDict):
    VirtualMFADevice: VirtualMFADeviceTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class ListVirtualMFADevicesResponseTypeDef(TypedDict):
    VirtualMFADevices: List[VirtualMFADeviceTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class GetGroupPolicyResponseTypeDef(TypedDict):
    GroupName: str
    PolicyName: str
    PolicyDocument: PolicyDocumentTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class GetRolePolicyResponseTypeDef(TypedDict):
    RoleName: str
    PolicyName: str
    PolicyDocument: PolicyDocumentTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class GetUserPolicyResponseTypeDef(TypedDict):
    UserName: str
    PolicyName: str
    PolicyDocument: PolicyDocumentTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class PolicyDetailTypeDef(TypedDict):
    PolicyName: NotRequired[str]
    PolicyDocument: NotRequired[PolicyDocumentTypeDef]

class PolicyVersionTypeDef(TypedDict):
    Document: NotRequired[PolicyDocumentTypeDef]
    VersionId: NotRequired[str]
    IsDefaultVersion: NotRequired[bool]
    CreateDate: NotRequired[datetime]

class RoleTypeDef(TypedDict):
    Path: str
    RoleName: str
    RoleId: str
    Arn: str
    CreateDate: datetime
    AssumeRolePolicyDocument: NotRequired[PolicyDocumentTypeDef]
    Description: NotRequired[str]
    MaxSessionDuration: NotRequired[int]
    PermissionsBoundary: NotRequired[AttachedPermissionsBoundaryTypeDef]
    Tags: NotRequired[List[TagTypeDef]]
    RoleLastUsed: NotRequired[RoleLastUsedTypeDef]

class EvaluationResultTypeDef(TypedDict):
    EvalActionName: str
    EvalDecision: PolicyEvaluationDecisionTypeType
    EvalResourceName: NotRequired[str]
    MatchedStatements: NotRequired[List[StatementTypeDef]]
    MissingContextValues: NotRequired[List[str]]
    OrganizationsDecisionDetail: NotRequired[OrganizationsDecisionDetailTypeDef]
    PermissionsBoundaryDecisionDetail: NotRequired[PermissionsBoundaryDecisionDetailTypeDef]
    EvalDecisionDetails: NotRequired[Dict[str, PolicyEvaluationDecisionTypeType]]
    ResourceSpecificResults: NotRequired[List[ResourceSpecificResultTypeDef]]

class GroupDetailTypeDef(TypedDict):
    Path: NotRequired[str]
    GroupName: NotRequired[str]
    GroupId: NotRequired[str]
    Arn: NotRequired[str]
    CreateDate: NotRequired[datetime]
    GroupPolicyList: NotRequired[List[PolicyDetailTypeDef]]
    AttachedManagedPolicies: NotRequired[List[AttachedPolicyTypeDef]]

class UserDetailTypeDef(TypedDict):
    Path: NotRequired[str]
    UserName: NotRequired[str]
    UserId: NotRequired[str]
    Arn: NotRequired[str]
    CreateDate: NotRequired[datetime]
    UserPolicyList: NotRequired[List[PolicyDetailTypeDef]]
    GroupList: NotRequired[List[str]]
    AttachedManagedPolicies: NotRequired[List[AttachedPolicyTypeDef]]
    PermissionsBoundary: NotRequired[AttachedPermissionsBoundaryTypeDef]
    Tags: NotRequired[List[TagTypeDef]]

class CreatePolicyVersionResponseTypeDef(TypedDict):
    PolicyVersion: PolicyVersionTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class GetPolicyVersionResponseTypeDef(TypedDict):
    PolicyVersion: PolicyVersionTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class ListPolicyVersionsResponseTypeDef(TypedDict):
    Versions: List[PolicyVersionTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ManagedPolicyDetailTypeDef(TypedDict):
    PolicyName: NotRequired[str]
    PolicyId: NotRequired[str]
    Arn: NotRequired[str]
    Path: NotRequired[str]
    DefaultVersionId: NotRequired[str]
    AttachmentCount: NotRequired[int]
    PermissionsBoundaryUsageCount: NotRequired[int]
    IsAttachable: NotRequired[bool]
    Description: NotRequired[str]
    CreateDate: NotRequired[datetime]
    UpdateDate: NotRequired[datetime]
    PolicyVersionList: NotRequired[List[PolicyVersionTypeDef]]

class CreateRoleResponseTypeDef(TypedDict):
    Role: RoleTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class CreateServiceLinkedRoleResponseTypeDef(TypedDict):
    Role: RoleTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class GetRoleResponseTypeDef(TypedDict):
    Role: RoleTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class InstanceProfileTypeDef(TypedDict):
    Path: str
    InstanceProfileName: str
    InstanceProfileId: str
    Arn: str
    CreateDate: datetime
    Roles: List[RoleTypeDef]
    Tags: NotRequired[List[TagTypeDef]]

class ListRolesResponseTypeDef(TypedDict):
    Roles: List[RoleTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class UpdateRoleDescriptionResponseTypeDef(TypedDict):
    Role: RoleTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class SimulatePolicyResponseTypeDef(TypedDict):
    EvaluationResults: List[EvaluationResultTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class CreateInstanceProfileResponseTypeDef(TypedDict):
    InstanceProfile: InstanceProfileTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class GetInstanceProfileResponseTypeDef(TypedDict):
    InstanceProfile: InstanceProfileTypeDef
    ResponseMetadata: ResponseMetadataTypeDef

class ListInstanceProfilesForRoleResponseTypeDef(TypedDict):
    InstanceProfiles: List[InstanceProfileTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class ListInstanceProfilesResponseTypeDef(TypedDict):
    InstanceProfiles: List[InstanceProfileTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef

class RoleDetailTypeDef(TypedDict):
    Path: NotRequired[str]
    RoleName: NotRequired[str]
    RoleId: NotRequired[str]
    Arn: NotRequired[str]
    CreateDate: NotRequired[datetime]
    AssumeRolePolicyDocument: NotRequired[PolicyDocumentTypeDef]
    InstanceProfileList: NotRequired[List[InstanceProfileTypeDef]]
    RolePolicyList: NotRequired[List[PolicyDetailTypeDef]]
    AttachedManagedPolicies: NotRequired[List[AttachedPolicyTypeDef]]
    PermissionsBoundary: NotRequired[AttachedPermissionsBoundaryTypeDef]
    Tags: NotRequired[List[TagTypeDef]]
    RoleLastUsed: NotRequired[RoleLastUsedTypeDef]

class GetAccountAuthorizationDetailsResponseTypeDef(TypedDict):
    UserDetailList: List[UserDetailTypeDef]
    GroupDetailList: List[GroupDetailTypeDef]
    RoleDetailList: List[RoleDetailTypeDef]
    Policies: List[ManagedPolicyDetailTypeDef]
    IsTruncated: bool
    Marker: str
    ResponseMetadata: ResponseMetadataTypeDef
