"""Core card utilities, a vectorized numpy hand evaluator and Monte Carlo equity.

Shared by the trainer and the submitted bot (Python 3.10+/3.11, numpy only).

Cards are ints 0..51: card = rank * 4 + suit, rank 0..12 (2..A), suit 0..3 (s h d c).
"""
import numpy as np

RANKS = "23456789TJQKA"
SUITS = "shdc"


def card_to_int(c):
    return RANKS.index(c[0]) * 4 + SUITS.index(c[1])


def int_to_card(i):
    return RANKS[i // 4] + SUITS[i % 4]


# ------------------------------------------------------------ lookup tables
_HB = np.full(1 << 14, -1, dtype=np.int64)       # highest set bit
for _m in range(1, 1 << 14):
    _HB[_m] = _m.bit_length() - 1


def _topk_table(k):
    t = np.zeros(1 << 13, dtype=np.int64)
    for m in range(1 << 13):
        v, n, r = 0, 0, 12
        while r >= 0 and n < k:
            if m >> r & 1:
                v = (v << 4) | r
                n += 1
            r -= 1
        v <<= 4 * (k - n)
        t[m] = v
    return t


_TOP1, _TOP2, _TOP3, _TOP5 = (_topk_table(k) for k in (1, 2, 3, 5))


def _straight_high(mask):
    """mask: int array of 13-bit rank masks -> straight high rank (0..12) or -1."""
    m2 = (mask << 1) | ((mask >> 12) & 1)          # bit0 = ace-low
    s = m2 & (m2 >> 1) & (m2 >> 2) & (m2 >> 3) & (m2 >> 4)
    hb = _HB[s & 0x3FFF]
    return np.where(s > 0, hb + 3, -1)


_ONE = np.int64(1)
_AR13 = np.arange(13, dtype=np.int64)
_AR4 = np.arange(4, dtype=np.int64)
_PW = (np.int64(1) << _AR13)


def evaluate(cards, chunk=8192):
    """Chunked wrapper around _evaluate to bound memory."""
    cards = np.asarray(cards, dtype=np.int64)
    if len(cards) <= chunk:
        return _evaluate(cards)
    return np.concatenate([_evaluate(cards[i:i + chunk]) for i in range(0, len(cards), chunk)])


def _evaluate(cards):
    """cards: int array (N, k), 5 <= k <= 7, distinct. Returns int64 scores (N,), higher wins."""
    cards = np.asarray(cards, dtype=np.int64)
    r = cards >> 2
    s = cards & 3
    n = cards.shape[0]
    bits = _ONE << r                                   # (N,k)
    counts = (r[:, :, None] == _AR13).sum(1)            # (N,13)
    rmask = (counts > 0) @ _PW
    m2 = (counts >= 2) @ _PW
    m3 = (counts >= 3) @ _PW
    m4 = (counts >= 4) @ _PW

    # flush
    sm = s[:, :, None] == _AR4                           # (N,k,4)
    suit_cnt = sm.sum(1)
    suit_masks = (bits[:, :, None] * sm).sum(1)
    fs = suit_cnt.argmax(1)
    has_flush = suit_cnt.max(1) >= 5
    fmask = suit_masks[np.arange(n), fs]

    sf_high = np.where(has_flush, _straight_high(fmask), -1)
    st_high = _straight_high(rmask)

    q = _HB[m4]
    t = _HB[m3]
    p_hi = _HB[m2]
    # pair rank for a full house: best rank with >=2 that isn't the trips rank
    fh_pair = _HB[m2 & ~np.where(t >= 0, _ONE << np.maximum(t, 0), 0)]
    p2 = _HB[m2 & ~np.where(p_hi >= 0, _ONE << np.maximum(p_hi, 0), 0)]
    npairs = (counts >= 2).sum(1)

    def clear(mask, rank):
        return mask & ~np.where(rank >= 0, _ONE << np.maximum(rank, 0), 0)

    C = np.int64(1 << 20)
    score = (0 * C) + _TOP5[rmask]                                       # high card
    pair_v = 1 * C + (np.maximum(p_hi, 0) << 12) + _TOP3[clear(rmask, p_hi)]
    score = np.where(npairs >= 1, pair_v, score)
    tp_v = 2 * C + (np.maximum(p_hi, 0) << 8) + (np.maximum(p2, 0) << 4) \
        + _TOP1[clear(clear(rmask, p_hi), p2)]
    score = np.where(npairs >= 2, tp_v, score)
    trips_v = 3 * C + (np.maximum(t, 0) << 8) + _TOP2[clear(rmask, t)]
    score = np.where(t >= 0, trips_v, score)
    score = np.where(st_high >= 0, 4 * C + st_high, score)
    score = np.where(has_flush, 5 * C + _TOP5[fmask], score)
    fh = (t >= 0) & (fh_pair >= 0)
    score = np.where(fh, 6 * C + (np.maximum(t, 0) << 4) + np.maximum(fh_pair, 0), score)
    score = np.where(q >= 0, 7 * C + (np.maximum(q, 0) << 4) + _TOP1[clear(rmask, q)], score)
    score = np.where(sf_high >= 0, 8 * C + sf_high, score)
    return score


def evaluate_one(cards):
    return int(evaluate(np.asarray([cards]))[0])


# ------------------------------------------------------------ Monte Carlo equity
_PAIRS3 = ((0, 1), (0, 2), (1, 2))


def _sample(rng, avail, n, k):
    """n rows of k distinct cards drawn without replacement from `avail`."""
    keys = rng.random((n, len(avail)))
    idx = np.argpartition(keys, k - 1, axis=1)[:, :k] if k < len(avail) else np.argsort(keys, 1)
    return avail[idx]


def _best(hole_cols, board):
    """hole_cols: (N, h) with h in {2, 3}; board (N,5). Best showdown score using 2 of the hole cards."""
    if hole_cols.shape[1] == 2:
        return evaluate(np.concatenate([hole_cols, board], 1))
    out = None
    for a, b in _PAIRS3:
        sc = evaluate(np.concatenate([hole_cols[:, [a, b]], board], 1))
        out = sc if out is None else np.maximum(out, sc)
    return out


def equity(my, board, dead=(), opp_n=2, n=256, rng=None):
    """Equity (win + tie/2) of `my` (2 or 3 card ints; 3 = keeps best 2 at showdown)
    against a uniformly random opponent holding `opp_n` cards (2 or 3, same rule),
    with the board completed at random."""
    rng = rng if rng is not None else np.random.default_rng()
    used = set(my) | set(board) | set(dead)
    avail = np.array([c for c in range(52) if c not in used], dtype=np.int64)
    need_board = 5 - len(board)
    draw = _sample(rng, avail, n, opp_n + need_board)
    opp = draw[:, :opp_n]
    full_board = np.concatenate([np.tile(np.asarray(board, dtype=np.int64), (n, 1)),
                                 draw[:, opp_n:]], 1) if board else draw[:, opp_n:]
    mine = np.tile(np.asarray(my, dtype=np.int64), (n, 1))
    a = _best(mine, full_board)
    b = _best(opp, full_board)
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / n)


STRONG_Q = (0.6, 0.85)   # strong ranges: opponent holdings in the top 40% / top 15% of current made-hand strength


def equity2(my, board, dead=(), opp_n=2, n=256, rng=None):
    """(equity vs random, vs top-40% range, vs top-15% range) for one situation."""
    rng = rng if rng is not None else np.random.default_rng()
    e, s1, s2 = equity_batch(np.asarray([list(my)]), np.asarray([list(board)]),
                             np.asarray([list(dead)]), opp_n, n, rng, strong=True)
    return float(e[0]), float(s1[0]), float(s2[0])


def equity_batch(my, board, dead, opp_n, n, rng, strong=False):
    """Vectorized equity for M independent situations with the same shapes.
    my (M,h), board (M,b), dead (M,d) int arrays (d may be 0). Returns (M,) floats.
    Same estimator as `equity` (uniform random opponent, random runout).
    With strong=True also returns equity against only those sampled opponent holdings whose
    made hand on the *current* board is in the top 40% (a range-aware strength estimate:
    "how good am I against the hands that would bet/raise here")."""
    my = np.asarray(my, dtype=np.int64)
    m = len(my)
    board = np.asarray(board, dtype=np.int64)
    board = board.reshape(m, -1) if board.size else np.zeros((m, 0), dtype=np.int64)
    dead = np.asarray(dead, dtype=np.int64)
    dead = dead.reshape(m, -1) if dead.size else np.zeros((m, 0), dtype=np.int64)
    need = opp_n + 5 - board.shape[1]
    used = np.concatenate([my, board, dead], 1)                       # (M,u)
    keys = rng.random((m, n, 52))
    blocked = np.zeros((m, 52), dtype=bool)
    blocked[np.arange(m)[:, None], used] = True
    keys[np.broadcast_to(blocked[:, None, :], keys.shape)] = 2.0
    draw = np.argpartition(keys, need - 1, axis=2)[:, :, :need].astype(np.int64)  # (M,n,need)
    draw = draw.reshape(m * n, need)
    opp = draw[:, :opp_n]
    fb = np.concatenate([np.repeat(board, n, 0), draw[:, opp_n:]], 1)
    a = _best(np.repeat(my, n, 0), fb)
    b = _best(opp, fb)
    win = ((a > b) + 0.5 * (a == b)).reshape(m, n)
    res = win.mean(1)
    if not strong:
        return res
    if board.shape[1] < 3:
        return res, res.copy(), res.copy()
    now = _best(opp, np.repeat(board, n, 0)).reshape(m, n)          # opponent's made hand now
    srt = np.sort(now, axis=1)
    out = [res]
    for q in STRONG_Q:
        thr = srt[:, min(n - 1, int(q * n))][:, None]
        mask = now >= thr
        out.append((win * mask).sum(1) / np.maximum(mask.sum(1), 1))
    return tuple(out)


from itertools import permutations as _perms
_SUIT_PERMS = list(_perms(range(4)))


def iso_key(cards3):
    """Canonical suit-isomorphism key for a preflop hand (ints)."""
    return min(tuple(sorted(((c >> 2) * 4 + p[c & 3]) for c in cards3)) for p in _SUIT_PERMS)
