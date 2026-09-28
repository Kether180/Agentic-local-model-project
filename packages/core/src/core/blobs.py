"""Object storage over a single boto3 S3 client.

MinIO is S3-compatible, so the only difference is `endpoint_url` — one client covers both.
"""

from functools import lru_cache

import boto3
from botocore.exceptions import ClientError
from types_boto3_s3.client import S3Client

from core.config import settings


class BlobStore:
    def __init__(self, bucket: str | None = None, client: S3Client | None = None) -> None:
        cfg = settings.blob
        self.bucket = bucket or cfg.bucket
        self._client: S3Client = client or boto3.client(
            "s3",
            endpoint_url=cfg.endpoint_url,
            aws_access_key_id=cfg.access_key,
            aws_secret_access_key=cfg.secret_key,
            region_name=cfg.region,
        )

    def save(self, key: str, data: bytes, content_type: str = "application/pdf") -> str:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return key

    def get(self, key: str) -> bytes:
        return self._client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as err:
            code = err.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        return True


@lru_cache
def get_blob_store() -> BlobStore:
    return BlobStore()
