from pydantic import BaseModel, Field
from typing import Dict, Optional
import hashlib
import json

class DocumentModel(BaseModel):
    page_content: str
    metadata: Optional[Dict] = None

    def generate_digest(self) -> str:
        content = self.page_content + json.dumps(self.metadata or {})
        return hashlib.md5(content.encode()).hexdigest()

class DocumentResponse(BaseModel):
    page_content: str
    metadata: Dict = Field(default_factory=dict)
