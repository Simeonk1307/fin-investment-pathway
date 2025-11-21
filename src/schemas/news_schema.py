import pathway as pw

class NewsSchema(pw.Schema):
        article_id: str
        title: str
        description: str
        content: str
        url: str
        published_at: str
        language: str
        source_name: str
        source_url: str