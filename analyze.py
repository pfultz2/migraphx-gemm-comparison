import csv, cmath

#delta = ck - rocblas

def read_data(f):
    with open(f, newline='') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['g', 'm', 'n', 'k', 'delta1', 'delta2'])
        for row in reader:
            yield row

def get_ratio(row, batch_factor=62):
    # batch = cmath.sqrt(float(row['g'])).real
    batch = float(row['g']) / batch_factor
    return (batch * float(row['m']) * float(row['n'])) / float(row['k'])

def mean(r):
    return sum(r) / len(r)

class Validate:
    def __init__(self):
        self.rows = []

    def collect(self, files):
        for f in files:
            for row in read_data(f):
                self.rows.append(row)

    def find_loss(self, row, c):
        delta = float(row['delta1'])
        r = c(row)
        if r == 'ck':
            if delta > 0:
                return delta
        elif r == 'rocblas':
            if delta < 0:
                return -1.0 * delta
        return 0.0

    def find_losses(self, c):
        return [self.find_loss(row, c) for row in self.rows]

    def average_perf_loss(self, c):
        return mean(self.find_losses(c))

    def calc_perf_loss(self, c):
        losses = self.find_losses(c)
        print("Total = {0}, Max = {1}, Mean = {2}".format(sum(losses), max(losses), mean(losses)))


class Analyzer:
    def __init__(self):
        self.ratio = []
        self.delta = []

    def collect(self, files):
        for f in files:
            for row in read_data(f):
                ratio = get_ratio(row)
                self.ratio.append(ratio)
                self.delta.append(float(row['delta1']))

    def calc_regression(self):
        n = len(self.ratio)
        sumxy = sum(x*y for x,y in zip(self.ratio, self.delta))
        sumx = sum(self.ratio)
        sumx2 = sum(x*x for x in self.ratio)
        sumy = sum(self.delta)
        m = (n * sumxy - sumx*sumy) / (n * sumx2 - sumx*sumx)
        b = (sumy - m*sumx) / n
        return m, b

    def positive_value_range(self):
        ratios = [ratio for ratio, delta in zip(self.ratio, self.delta) if delta > 0 and delta < 0.001]
        return min(ratios), max(ratios), mean(ratios)

def large_k(row):
    if int(row['k']) > 2048:
        return 'rocblas'
    return 'ck'

def large_k_ratio(row, threshold=7, batch_factor=62):
    ratio = get_ratio(row, batch_factor)
    if ratio > threshold:
        return 'ck'
    return 'rocblas'

def large_k_ratio_average_loss(validate, *args, **kwargs):
    return validate.average_perf_loss(lambda c:large_k_ratio(c, *args, **kwargs))

def minimize(r, f):
    result = None
    min_val = None
    for x in r:
        y = f(x)
        if not min_val or y < min_val:
            result = x
            min_val = y
    return result, min_val

def minimize_mean(validate):
    losses = [validate.average_perf_loss(lambda c:large_k_ratio(c, i)) for i in range(2048)]
    m = min(losses)
    return losses.index(m), m


files = ['gemm_mi250_1_half.csv', 'gemm_mi250_64_half.csv']
analyze = Analyzer()
analyze.collect(files)
low, high, average = analyze.positive_value_range()
print(f"low = {low}, high = {high}, average = {average}")

m, b = analyze.calc_regression()
print(f"y = {m} * x + {b}")
threshold = -b / m
print(f"threshold = {threshold}")

validate = Validate()
validate.collect(files)

# values = [(threshold, batch_factor) for threshold in range(1, 2048) for batch_factor in range(1, 64)]
# min_result, min_value = minimize(values, lambda x:large_k_ratio_average_loss(validate, x[0], x[1]))
# print(f"Min result = {min_result}, Min average loss = {min_value}")

print("***************")
print("large_k_ratio 7, 62")
validate.calc_perf_loss(lambda c: large_k_ratio(c, 7, 62))
print("large_k_ratio 8, 64")
validate.calc_perf_loss(lambda c: large_k_ratio(c, 8, 64))
print("large_k_ratio 7, 64")
validate.calc_perf_loss(lambda c: large_k_ratio(c, 7, 64))
print("large_k_ratio 6, 64")
validate.calc_perf_loss(lambda c: large_k_ratio(c, 6, 64))
print("large_k_ratio 128, 2")
validate.calc_perf_loss(lambda c: large_k_ratio(c, 128, 2))
print("large_k")
validate.calc_perf_loss(large_k)


