from boto3.resources.model import ResourceModel
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.model import ServiceModel

from .base import NestedDocumenter

class SubResourceDocumenter(NestedDocumenter):
    def document_sub_resources(self, section: DocumentStructure) -> None: ...

def document_sub_resource(
    section: DocumentStructure,
    resource_name: str,
    sub_resource_model: ResourceModel,
    service_model: ServiceModel,
    include_signature: bool = ...,
) -> None: ...
