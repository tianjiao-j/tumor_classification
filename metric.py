# Y = {0, 1}^k
# Z = {0, 1}^k
def union(y, z):
    '''y or z'''
    return set(y) | set(z)


def intersection(y, z):
    '''y and z'''
    return set(y) & set(z)


def union_sub_inter(y, z):
    '''y or z - y and z'''
    return union(y, z) - intersection(y, z)


class Metric():
    def __init__(self, Y, Z, labels):
        self.Y = Y  # actual
        self.Z = Z  # predicted
        self.n = len(Y)
        self.labels = labels
        self.k = len(self.labels)

        # exact match ratio

    def EMR(self):
        flags = [set(yi) == set(zi) for yi, zi in zip(self.Y, self.Z)]
        return sum(flags) / self.n

    def Accuracy(self):
        pred_accs = [len(intersection(yi, zi)) / len(union(yi, zi)) for yi, zi in zip(self.Y, self.Z)]
        return sum(pred_accs) / self.n

    def Precision(self):
        precisions = [len(intersection(yi, zi)) / len(zi) for yi, zi in zip(self.Y, self.Z)]
        return sum(precisions) / self.n

    def Recall(self):
        recalls = [len(intersection(yi, zi)) / len(yi) for yi, zi in zip(self.Y, self.Z)]
        return sum(recalls) / self.n

    def F1(self):
        f1s = 0
        for yi, zi in zip(self.Y, self.Z):
            f1 = 2 * len(intersection(yi, zi)) / (len(yi) + len(zi))
            f1s += f1
        return f1s / self.n

    # hamming loss
    def HL(self):
        hls = 0
        for yi, zi in zip(self.Y, self.Z):
            hls += len(union_sub_inter(yi, zi))
        return hls / (self.k * self.n)

    def macroPrecision(self):
        P_list = []
        for k in range(self.k):
            label = self.labels[k]
            numerator = 0.
            denominator = 0.
            for i in range(self.n):
                yi, zi = self.Y[i], self.Z[i]
                if label in yi and label in zi:
                    numerator += 1
                if label in zi:
                    denominator += 1
            if denominator == 0:
                res = 0
            else:
                res = numerator / denominator
            P_list.append(res)
        return sum(P_list) / self.k

    def macroRecall(self):
        R_list = []
        for k in range(self.k):
            label = self.labels[k]
            numerator = 0.
            denominator = 0.
            for i in range(self.n):
                yi, zi = self.Y[i], self.Z[i]
                if label in yi and label in zi:
                    numerator += 1
                if label in yi:
                    denominator += 1
            if denominator == 0:
                res = 0
            else:
                res = numerator / denominator
            R_list.append(res)
        return sum(R_list) / self.k

    def macroF1(self):
        F1_list = []
        for k in range(self.k):
            label = self.labels[k]
            numerator = 0.
            denominator = 0.
            for i in range(self.n):
                yi, zi = self.Y[i], self.Z[i]
                if label in yi and label in zi:
                    numerator += 1
                if label in yi:
                    denominator += 1
                if label in zi:
                    denominator += 1
            if denominator == 0:
                res = 0
            else:
                res = 2 * numerator / denominator
            F1_list.append(res)
        return sum(F1_list) / self.k

    def microPrecision(self):
        res = 0.
        numerator = 0.
        denominator = 0.
        for yi, zi in zip(self.Y, self.Z):
            numerator += len(intersection(yi, zi))
            denominator += len(zi)
        return numerator / denominator

    def microRecall(self):
        res = 0.
        numerator = 0.
        denominator = 0.
        for yi, zi in zip(self.Y, self.Z):
            numerator += len(intersection(yi, zi))
            denominator += len(yi)
        return numerator / denominator

    def microF1(self):
        res = 0.
        numerator = 0.
        denominator = 0.
        for yi, zi in zip(self.Y, self.Z):
            numerator += len(intersection(yi, zi))
            temp = len(yi) + len(zi)
            denominator += temp
        return 2 * numerator / denominator


if __name__ == '__main__':
    labels = [0, 1, 2, 3, 4, 5, 6]
    n = 100
    import random

    Y = [random.sample(labels, 2) for _ in range(n)]
    Z = [random.sample(labels, 2) for _ in range(n)]
    M = Metric(Y, Z, labels)
    print(M.Precision())
    print(M.Recall())
    print(M.F1())
    print(M.EMR())
    print(M.microF1())
    print(M.microRecall())
    print(M.macroPrecision())
    print(M.HL())
    print(M.macroPrecision())
    print(M.macroF1())
    print(M.macroRecall())
