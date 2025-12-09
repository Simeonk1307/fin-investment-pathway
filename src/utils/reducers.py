import pathway as pw
import pathway as pw
# from src.utils.indicators import Indicators
# from src.utils.lstm import Predictor

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

class StdDevAccumulator(pw.BaseCustomAccumulator):
    def __init__(self, cnt, sum, sum_sq):
        self.cnt = cnt
        self.sum = sum
        self.sum_sq = sum_sq

    @classmethod
    def from_row(cls, row):
        [val] = row
        return cls(1, val, val**2)

    def update(self, other):
        self.cnt += other.cnt
        self.sum += other.sum
        self.sum_sq += other.sum_sq

    def compute_result(self) -> float:
        mean = self.sum / self.cnt
        mean_sq = self.sum_sq / self.cnt
        return mean_sq - mean**2
    
    def retract(self, other):
        self.cnt -= other.cnt
        self.sum -= other.sum
        self.sum_sq -= other.sum_sq
    
stddev = pw.reducers.udf_reducer(StdDevAccumulator)

class RangeAccumulator(pw.BaseCustomAccumulator):
    def __init__(self,maxi, mini):
        self.maxi = maxi
        self.mini = mini

    @classmethod
    def from_row(cls, row):
        [val] = row
        return cls(val, val)

    def update(self, other):
        self.maxi = max(self.maxi, other.maxi)
        self.mini = min(self.mini, other.mini)

    def compute_result(self) -> float:
        return self.maxi - self.mini
    
range_calc = pw.reducers.udf_reducer(RangeAccumulator)

# check get_weight_timestamp usage in input_pipeline.py
class WeightedSentimentScoreAccumulator(pw.BaseCustomAccumulator):
  
  def __init__(self, negative,neutral_,positive,weight):
    self.negative = negative
    self.neutral_ = neutral_
    self.positive = positive
    self.weight = weight

  @classmethod
  def from_row(self, row):
    [(negative,neutral_,positive),weight] = row
    return WeightedSentimentScoreAccumulator(negative, neutral_, positive, weight)

  def update(self, other):
    self.negative += other.negative*self.weight
    self.neutral_ += other.neutral_*self.weight
    self.positive += other.positive*self.weight

  def compute_result(self) -> tuple[float, float, float]:
    return (self.negative, self.neutral_, self.positive)
  
score_accum = pw.reducers.udf_reducer(WeightedSentimentScoreAccumulator)

class SimpleSentimentScoreAccumulator(pw.BaseCustomAccumulator):
  
  def __init__(self, negative,neutral_,positive,cnt):
    self.negative = negative
    self.neutral_ = neutral_
    self.positive = positive
    self.cnt = cnt

  @classmethod
  def from_row(self, row):
    [(negative,neutral_,positive)] = row
    return SimpleSentimentScoreAccumulator(negative, neutral_, positive, 1)

  def update(self, other):
    self.negative += other.negative
    self.neutral_ += other.neutral_
    self.positive += other.positive
    self.cnt+=1

  def compute_result(self) -> tuple[float, float, float]:
    return (self.negative/self.cnt, self.neutral_/self.cnt, self.positive/self.cnt)
  
score_accum = pw.reducers.udf_reducer(SimpleSentimentScoreAccumulator)