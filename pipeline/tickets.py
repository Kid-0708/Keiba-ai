"""買い目の決め方（シミュレーターと同じルール）と、組み合わせの確率（ハービル式）"""
import itertools
import numpy as np

EVTH = 1.2
BETS = ["tan", "fuku", "umaren", "wide", "umatan", "trio", "tri"]
BET_NAME = {"tan": "単勝", "fuku": "複勝", "umaren": "馬連", "wide": "ワイド", "umatan": "馬単", "trio": "3連複", "tri": "3連単"}


def trifectas(u, p):
    """上位3着の並び (i,j,k) と確率"""
    p = np.asarray(p, float); p = p / p.sum()
    n = len(u); out = []
    for i in range(n):
        for j in range(n):
            if j == i: continue
            pij = p[i] * p[j] / (1 - p[i])
            for k in range(n):
                if k in (i, j): continue
                out.append((u[i], u[j], u[k], pij * p[k] / (1 - p[i] - p[j])))
    return out


def place_probs(u, p):
    """各馬の 2着以内・3着以内 の確率"""
    t2 = dict.fromkeys(u, 0.0); t3 = dict.fromkeys(u, 0.0)
    for a, b, c, q in trifectas(u, p):
        t2[a] += q; t2[b] += q; t3[a] += q; t3[b] += q; t3[c] += q
    return t2, t3


def marks(u, p, odds):
    o = [x for _, x in sorted(zip(p, u), reverse=True)]
    m = {}
    for i, s in enumerate(["◎", "○", "▲", "△", "△"]):
        if i < len(o): m[o[i]] = s
    pm = dict(zip(u, p)); od = dict(zip(u, odds))
    cand = [x for x in o[5:] if od.get(x) and pm[x] >= 0.05 and pm[x] * od[x] >= EVTH]
    if cand: m[max(cand, key=lambda x: pm[x] * od[x])] = "☆"
    return m


def tickets(u, p):
    """券種ごとの買い目（組番の文字列リスト）"""
    o = [x for _, x in sorted(zip(p, u), reverse=True)]
    A, B, C = o[0], o[1], o[2]
    D = o[3] if len(o) > 3 else None
    E = o[4] if len(o) > 4 else None
    _, t3 = place_probs(u, p)
    best3 = max(u, key=lambda x: t3[x])
    j = lambda *xs: "-".join(str(x) for x in xs)
    js = lambda *xs: "-".join(str(x) for x in sorted(xs))
    out = {
        "tan": [j(A)], "fuku": [j(best3)],
        "umaren": [js(A, B), js(A, C)] + ([js(A, D)] if D else []),
        "wide": [js(A, B), js(A, C), js(B, C)],
        "umatan": [j(A, B), j(A, C), j(B, A)],
        "trio": [js(*t) for t in [(A, B, C), (A, B, D), (A, B, E), (A, C, D), (A, C, E)] if None not in t],
        "tri": [j(*t) for t in [(A, B, C), (A, B, D), (A, B, E), (A, C, B), (A, C, D), (A, C, E)] if None not in t],
    }
    return out


def hit_return(bet, ticket_list, payouts):
    """payouts: {(bet, combo): yen}。払戻金の合計（100円ずつ買った場合）"""
    ret = 0
    for t in ticket_list:
        key = t
        if bet in ("umaren", "wide", "trio"):
            key = "-".join(sorted(t.split("-"), key=int))
        ret += payouts.get((bet, key), 0)
    return ret
