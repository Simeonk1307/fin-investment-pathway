import os
from minio import Minio
import io
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class MinioStorage:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        self.client = Minio(
            os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=False,
        )
        self.bucket = os.getenv("MINIO_BUCKET", "filings")

        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def save_text(self, filepath: str, text: str) -> str:
        data = text.encode("utf-8")
        stream = io.BytesIO(data)
        self.client.put_object(
            bucket_name=self.bucket,
            object_name=filepath,
            data=stream,
            length=len(data),
            content_type="text/plain",
        )
        return filepath

    def read_text(self, filepath: str) -> str:
        try:
            resp = self.client.get_object(self.bucket, filepath)
            content = resp.read().decode("utf-8", errors="ignore")
            resp.close()
            resp.release_conn()
            return content
        except Exception as e:
            logger.error(f"Error reading text from {filepath}: {e}")
            return ""

    def _clean_path(self, path: str) -> str:
        if not path:
            return ""
        path = path.strip().strip('"\'')
        path = path.replace('\\"', '').replace("\\'", "")
        path = path.replace('"', '').replace("'", "")
        return path.rstrip('\\/')

    def read_filing(self, storage_url: str) -> str:
        """Read filing content from storage_url"""
        try:
            if not storage_url:
                return ""
            
            url = str(storage_url).strip().strip('"').strip("'")
            url = url.replace('\\"', '').replace("\\'", "")
            
            if "minio://" in url:
                path = url.replace("minio://", "")
                if path.startswith(f"{self.bucket}/"):
                    path = path.replace(f"{self.bucket}/", "", 1)
            else:
                path = url
                if path.startswith(f"{self.bucket}/"):
                    path = path.replace(f"{self.bucket}/", "", 1)
            
            path = self._clean_path(path)
            
            logger.info(f"Reading filing: bucket={self.bucket}, path={path}")
            
            content = self.read_text(path)
            
            if content:
                logger.info(f"Successfully read {len(content)} characters")
            else:
                logger.warning(f"Empty content for path: {path}")
            
            return content
            
        except Exception as e:
            logger.error(f"Error reading filing from {storage_url}: {e}")
            return ""