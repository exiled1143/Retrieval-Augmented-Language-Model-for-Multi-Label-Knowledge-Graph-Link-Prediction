from KnowledgeGraph.metrics import Metrics
from typing import List


class PrecisionAtKTotal(Metrics):
    """precision@K metrics"""

    def __init__(self, topN: int):
        self._topN = topN
        self._results = []
        self._tp = 0
        self._fp = 0

    def calculate(self, targets: List[int], predictions: List[int]):
        predictions = set(predictions[:self._topN])
        targets = set(targets)
        tp = len(predictions.intersection(targets))
        fp = len(predictions)-tp
        self._tp += tp
        self._fp += fp

    def show_result(self):
        print("="*5)
        print(f"TP:{self._tp}, FP:{self._fp}")
        if self._tp+self._fp == 0:
            print(f"precision at {self._topN}:0")
        else:    
            print(f"precision at {self._topN}:{self._tp/(self._tp+self._fp)}")
        print("="*5)

