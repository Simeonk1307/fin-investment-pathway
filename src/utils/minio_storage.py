import re
import os
from minio import Minio
import io
from dotenv import load_dotenv

load_dotenv()


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
        except Exception:
            return ""

    def read_filing_extract(self, filepath: str, max_total: int = 2000) -> str:
        """Read filing and extract key SEC sections only."""
        filepath = self._clean_path(filepath)
        if not filepath:
            return ""

        raw = self.read_text(filepath)
        if not raw:
            return ""

        return self._extract_key_sections(raw, max_total)

    def _clean_path(self, path: str) -> str:
        if not path:
            return ""
        path = path.strip().strip('"\'')
        path = path.replace('\\"', '').replace("\\'", "")
        path = path.replace('"', '').replace("'", "")
        return path.rstrip('\\/')

    def _extract_key_sections(self, text: str, max_total: int = 2000) -> str:
        """Extract important sections from SEC filing."""
        if not text:
            return ""

        # Clean HTML/XML
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        sections = [
            (r"Item\s*1A[.\s:]*Risk\s*Factors(.*?)(?=Item\s*\d|$)", "RISKS"),
            (r"Item\s*7[.\s:]*Management.s Discussion(.*?)(?=Item\s*\d|$)", "MD&A"),
            (r"Item\s*1[.\s:]*Business(.*?)(?=Item\s*\d|$)", "BUSINESS"),
            (r"Item\s*2\.01|Item\s*5\.02(.*?)(?=Item\s*\d|$)", "MATERIAL_EVENTS"),
        ]

        extracted = []
        chars_used = 0
        per_section = max_total // 3

        for pattern, label in sections:
            if chars_used >= max_total:
                break
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()[:per_section]
                if len(content) > 50:
                    extracted.append(f"[{label}]: {content}")
                    chars_used += len(content)

        if not extracted:
            return f"[EXCERPT]: {text[:max_total]}"

        return "\n\n".join(extracted)