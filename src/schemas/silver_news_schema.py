import pathway as pw

class FinnHubNewsSchema(pw.Schema):
        category: str
        datetime: int
        headline: str
        news_id: int # from id
        image: str
        related: str
        source: str
        summary: str
        url: str



# class GNewsSchema(pw.Schema):
#         id: str
#         headline: str
#         description: str
#         content: str
#         url: str
#         published_at: str
#         language: str
#         source: str