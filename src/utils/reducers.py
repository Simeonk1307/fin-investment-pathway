import pathway as pw

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