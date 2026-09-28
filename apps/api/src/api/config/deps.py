"""Router dependencies."""

from typing import Annotated

from core.blobs import BlobStore, get_blob_store
from core.db import get_session
from core.embedding import Embedder, get_embedder
from fastapi import Depends
from sqlalchemy.orm import Session

SessionDep = Annotated[Session, Depends(get_session)]
BlobStoreDep = Annotated[BlobStore, Depends(get_blob_store)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
