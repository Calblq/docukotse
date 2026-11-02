"""S3-compatible file storage (RustFS in docker-compose; works with AWS S3 or MinIO too)."""

import logging
from functools import lru_cache
from typing import Protocol

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class Storage(Protocol):
    def put_bytes(self, key: str, data: bytes, content_type: str) -> None: ...
    def get_bytes(self, key: str) -> bytes: ...
    def delete(self, key: str) -> None: ...
    def presigned_url(self, key: str, filename: str, expires: int = 600) -> str: ...


def _client(endpoint: str, settings: Settings):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name="us-east-1",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=5,
            read_timeout=30,
            retries={"max_attempts": 2},
        ),
    )


class S3Storage:
    def __init__(self, settings: Settings) -> None:
        self.bucket = settings.s3_bucket
        self.client = _client(settings.s3_endpoint, settings)
        # Signed URLs are opened by the browser, so they must use the public host.
        self.public_client = _client(settings.s3_public_endpoint, settings)
        self._bucket_ready = False

    def ensure_bucket(self) -> None:
        """Create the bucket if missing. Called lazily on the first upload."""
        if self._bucket_ready:
            return
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)
            logger.info("Created bucket %s", self.bucket)
        self._bucket_ready = True

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        self.ensure_bucket()
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def get_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def presigned_url(self, key: str, filename: str, expires: int = 600) -> str:
        return self.public_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": self.bucket,
                "Key": key,
                "ResponseContentDisposition": f'inline; filename="{filename}"',
            },
            ExpiresIn=expires,
        )


@lru_cache
def get_storage() -> Storage:
    return S3Storage(get_settings())
