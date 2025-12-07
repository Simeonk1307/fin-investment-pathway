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

    def save_text(self, filename: str, text: str):
        data = text.encode("utf-8")
        stream = io.BytesIO(data)
        self.client.put_object(
            self.bucket, filename, stream, len(data), content_type="text/plain"
        )
        return f"s3://{self.bucket}/{filename}"
    
    def read_from_minio(self, bucket: str, filename: str):
        resp = self.client.get_object(bucket, filename)
        content = resp.read().decode("utf-8")

        return content

