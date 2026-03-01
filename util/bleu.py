# util/bleu.py
import math
from collections import Counter

'''
BLEU = BP * (Π p_n ^ w_n)

BLEU = BP * exp( Σ w_n * log p_n )
where,
- p_n: modified n-gram precision
- BP: brevity penalty
'''

def ngram_counts(tokens, n):
    return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

def modified_precision(candidate, reference, n):
    cand_ng = ngram_counts(candidate, n) # 候補文のn-gramカウント
    ref_ng  = ngram_counts(reference, n) # 参照文のn-gramカウント

    overlap = {ng: min(count, ref_ng.get(ng, 0)) for ng, count in cand_ng.items()} # 重複分のカウント

    return sum(overlap.values()), max(sum(cand_ng.values()), 1) # 合計重複数, 合計候補のn-gram数

def brevity_penalty(candidate, reference):
    c = len(candidate)
    r = len(reference)
    if c == 0:
        return 0
    return math.exp(1 - r / c) if c < r else 1

def compute_bleu(candidate, reference, max_n=4):
    # token を int のままでも OK（<SOS>=2 や <EOS>=3 は後で除去）
    weights = [1/max_n] * max_n  # ex : [0.25, 0.25, 0.25, 0.25]

    score = 0
    # 1-gram ~ n-gram まで計算
    for n in range(1, max_n+1):
        overlap, total = modified_precision(candidate, reference, n)
        if total == 0:
            return 0
        precision = overlap / total
        if precision == 0:
            return 0
        score += weights[n-1] * math.log(precision)

    return brevity_penalty(candidate, reference) * math.exp(score)
