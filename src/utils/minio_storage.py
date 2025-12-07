from minio import Minio
import io, os
from dotenv import load_dotenv

load_dotenv()

class MinioStorage:
    def __init__(self):
        self.client = Minio(
            os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=False
        )
        self.bucket = os.getenv("MINIO_BUCKET", "filings")

        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def save_text(self, filepath: str, text: str):
        data = text.encode("utf-8")
        stream = io.BytesIO(data)
        self.client.put_object(
            bucket_name=self.bucket,
            object_name=filepath,
            data=stream,
            length=len(data),
            content_type="text/plain"
        )
        return filepath  # Pathway stores this string

    def read_text(self, filepath: str):
        resp = self.client.get_object(self.bucket, filepath)
        content = resp.read().decode("utf-8")
        resp.close()
        resp.release_conn()
        return content
