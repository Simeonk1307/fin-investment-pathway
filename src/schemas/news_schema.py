import pathway as pw

class FinnHubNewsSchema(pw.Schema):
        id: int
        headline: str
        description: str
        url: str
        source: str
        published_at: str
        category: str
        company: str



class GNewsSchema(pw.Schema):
        id: str
        headline: str
        description: str
        content: str
        url: str
        published_at: str
        language: str
        source: str