import pathway as pw

class YFinanceEquitySchema(pw.Schema):
    ticker: str # from id
    price: float
    time: int
    exchange: str
    quote_type: int
    market_hours: int
    change_percent: float
    day_volume: int
    change: float
    price_hint: int


# Something unusual
# 
#  {
#     "id": "META",
#     "price": 643.2875,
#     "time": "1764609222000",
#     "exchange": "NMS",
#     "quote_type": 8,
#     "market_hours": 1,
#     "change_percent": -0.71958274,
#     "day_volume": "5534552",
#     "change": -4.6625366,
#     "last_size": "140",
#     "price_hint": "2"
# }

# {
#     "id": "META",
#     "price": 643.2875,
#     "time": "1764609222000",
#     "exchange": "NMS",
#     "quote_type": 8,
#     "market_hours": 1,
#     "change_percent": -0.71958274,
#     "day_volume": "5534576",
#     "change": -4.6625366,
#     "price_hint": "2"
# }

# {
#     "id": "AVGO",
#     "price": 391.1499,
#     "time": "1764609222000",
#     "exchange": "NMS",
#     "quote_type": 8,
#     "market_hours": 1,
#     "change_percent": -2.930834,
#     "day_volume": "9681293",
#     "change": -11.810089,
#     "last_size": "50",
#     "price_hint": "2"
# }

# {
#     "id": "AVGO",
#     "price": 391.1499,
#     "time": "1764609222000",
#     "exchange": "NMS",
#     "quote_type": 8,
#     "market_hours": 1,
#     "change_percent": -2.930834,
#     "day_volume": "9681342",
#     "change": -11.810089,
#     "price_hint": "2"
# }