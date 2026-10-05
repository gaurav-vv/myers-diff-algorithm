"""Myers' O(ND) diff (linear-space variant) for lines and characters.

Usage:
    main.py lines     A B   # Part A: minimal line diff
    main.py highlight A B   # Part B: line diff + changed-character ranges

Overview
--------
The core routine `diff_marks(a, b)` takes two sequences and returns two
boolean lists:
    del_a[i] is True  -> a[i] is deleted   (only in A)
    ins_b[j] is True  -> b[j] is inserted  (only in B)
Every unmarked element is a "keep", and the kept elements of A and B line up
one-to-one in order. Printing is a simple walk over those marks, which also
gives us the delete-first rule for free.

The marks are computed with Myers' "middle snake" divide and conquer
(section 4b of the 1986 paper): run the greedy D-path search from both ends
at once, stop where the forward and reverse paths overlap, split the problem
at that point and recurse. It uses O(N + M) memory instead of storing one V
array per value of d.
"""

import sys


# ---------------------------------------------------------------------------
# Reading input
# ---------------------------------------------------------------------------

def read_lines(path):
    """Read a file as raw bytes and split it into lines (brief, section 1)."""
    with open(path, "rb") as f:
        data = f.read()
    parts = data.split(b"\n")
    if parts[-1] == b"":
        # A trailing newline (or an empty file) does not make an extra line.
        parts.pop()
    return parts  # any '\r' stays inside the line


# ---------------------------------------------------------------------------
# Myers' middle snake
# ---------------------------------------------------------------------------

def middle_snake(a, b):
    """Find a split point (x, y) that lies on a shortest edit path of a -> b.

    Both a and b are non-empty, and a[0] != b[0], a[-1] != b[-1] (the caller
    trims common prefix and suffix first).

    v1[offset + k] is the furthest x reached on diagonal k (k = x - y) by the
    forward search from (0, 0). v2[offset + k] is the same for the reverse
    search from (n, m), run on the reversed sequences, so x there counts
    elements from the end. The loops walk the array index o = offset + k
    directly, which saves arithmetic in the hot loop.

    Entries never visited stay -1. Diagonals -d-1 and d+1 are never visited
    before step d, so the classic "k == -d / k == d" boundary tests are not
    needed: at k = -d the left neighbour is -1 and we move down, at k = d
    the right neighbour is -1 and we move right.

    Returns (x, y), or None if no split was found (should not happen).
    """
    n = len(a)
    m = len(b)
    ar = a[::-1]                    # reversed copies for the reverse search
    br = b[::-1]
    max_d = (n + m + 1) // 2
    offset = max_d + 1
    size = 2 * max_d + 4
    v1 = [-1] * size
    v2 = [-1] * size
    v1[offset + 1] = 0
    v2[offset + 1] = 0
    delta = n - m
    # If delta is odd the paths can first meet during a forward step,
    # otherwise during a reverse step.
    front = (delta & 1) == 1
    # Index of diagonal (delta - k) in the other array is (o_mirror - o).
    o_mirror = 2 * offset + delta
    # k1start/k1end (and k2...) shrink the diagonal range once a path runs
    # off the edge of the edit graph, so we never walk outside the grid.
    k1start = k1end = k2start = k2end = 0

    for d in range(max_d + 1):
        # ---- forward search: D-paths starting at (0, 0) ----
        for o in range(offset - d + k1start, offset + d + 1 - k1end, 2):
            # Choose the neighbour diagonal that reaches further:
            # down from k+1 (an insertion) or right from k-1 (a deletion).
            x = v1[o + 1]
            left = v1[o - 1]
            if left >= x:
                x = left + 1
            y = x - o + offset
            # Follow the snake: a run of equal elements (diagonal moves).
            while x < n and y < m and a[x] == b[y]:
                x += 1
                y += 1
            v1[o] = x
            if x > n:
                k1end += 2          # ran off the right edge
            elif y > m:
                k1start += 2        # ran off the bottom edge
            elif front:
                # Does the reverse (d-1)-path on the mirrored diagonal reach
                # past us? Then the two paths overlap: middle snake found.
                o2 = o_mirror - o
                if 0 <= o2 < size:
                    x2 = v2[o2]
                    if x2 != -1 and x >= n - x2:
                        return x, y

        # ---- reverse search: D-paths starting at (n, m) ----
        for o in range(offset - d + k2start, offset + d + 1 - k2end, 2):
            x = v2[o + 1]
            left = v2[o - 1]
            if left >= x:
                x = left + 1
            y = x - o + offset
            # Snake backwards from the end of both sequences.
            while x < n and y < m and ar[x] == br[y]:
                x += 1
                y += 1
            v2[o] = x
            if x > n:
                k2end += 2
            elif y > m:
                k2start += 2
            elif not front:
                # Does the forward d-path on the mirrored diagonal overlap us?
                o1 = o_mirror - o
                if 0 <= o1 < size:
                    x1 = v1[o1]
                    if x1 != -1 and x1 >= n - x:
                        return x1, x1 - o1 + offset
    return None


def diff_marks(a, b):
    """Return (del_a, ins_b) marking a minimal edit script from a to b."""
    del_a = [False] * len(a)
    ins_b = [False] * len(b)

    # Work stack of sub-problems: (a_lo, a_hi, b_lo, b_hi).
    # An explicit stack avoids Python's recursion limit.
    stack = [(0, len(a), 0, len(b))]
    while stack:
        a_lo, a_hi, b_lo, b_hi = stack.pop()

        # Trim the common prefix and suffix: those are keeps.
        while a_lo < a_hi and b_lo < b_hi and a[a_lo] == b[b_lo]:
            a_lo += 1
            b_lo += 1
        while a_lo < a_hi and b_lo < b_hi and a[a_hi - 1] == b[b_hi - 1]:
            a_hi -= 1
            b_hi -= 1

        if a_lo == a_hi:            # nothing left in A: insert the rest of B
            for j in range(b_lo, b_hi):
                ins_b[j] = True
            continue
        if b_lo == b_hi:            # nothing left in B: delete the rest of A
            for i in range(a_lo, a_hi):
                del_a[i] = True
            continue

        split = middle_snake(a[a_lo:a_hi], b[b_lo:b_hi])
        if split is None:           # safety net; not expected
            for i in range(a_lo, a_hi):
                del_a[i] = True
            for j in range(b_lo, b_hi):
                ins_b[j] = True
            continue
        x, y = split
        # Solve both halves. Each half is strictly smaller than the whole.
        stack.append((a_lo + x, a_hi, b_lo + y, b_hi))
        stack.append((a_lo, a_lo + x, b_lo, b_lo + y))

    return del_a, ins_b


def diff_lines(a, b):
    """Line diff with a speed-up that keeps the result minimal.

    1. Map every distinct line to a small integer, so comparisons are cheap.
    2. A line that never appears in the other file can never be a keep, so it
       must be a delete (or insert) in every edit script. Mark it directly and
       run Myers only on the remaining lines. This does not change the number
       of edits, but makes very different files much faster.
    """
    ids = {}
    a_ids = [ids.setdefault(line, len(ids)) for line in a]
    b_ids = [ids.setdefault(line, len(ids)) for line in b]

    in_a = set(a_ids)
    in_b = set(b_ids)
    a_keep_idx = [i for i, v in enumerate(a_ids) if v in in_b]
    b_keep_idx = [j for j, v in enumerate(b_ids) if v in in_a]

    del_a = [True] * len(a)
    ins_b = [True] * len(b)
    sub_del, sub_ins = diff_marks([a_ids[i] for i in a_keep_idx],
                                  [b_ids[j] for j in b_keep_idx])
    for pos, i in enumerate(a_keep_idx):
        del_a[i] = sub_del[pos]
    for pos, j in enumerate(b_keep_idx):
        ins_b[j] = sub_ins[pos]
    return del_a, ins_b


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def blocks(del_a, ins_b):
    """Walk the marks and yield ('keep', i, j) or ('change', dels, inss).

    A change block is a maximal run of deleted A lines plus inserted B lines
    with no keep between them. Yielding the deletes first is exactly the
    delete-first rule.
    """
    n = len(del_a)
    m = len(ins_b)
    i = j = 0
    while i < n or j < m:
        i0 = i
        while i < n and del_a[i]:
            i += 1
        j0 = j
        while j < m and ins_b[j]:
            j += 1
        if i > i0 or j > j0:
            yield ("change", range(i0, i), range(j0, j))
        elif i < n and j < m:
            yield ("keep", i, j)
            i += 1
            j += 1
        else:  # cannot happen when the marks are consistent
            break


def ranges_text(marks):
    """Turn [False, True, True, False, True] into '1-3,4-5' ('.' if none)."""
    out = []
    start = None
    for idx, flag in enumerate(marks):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            out.append("%d-%d" % (start, idx))
            start = None
    if start is not None:
        out.append("%d-%d" % (start, len(marks)))
    return ",".join(out) if out else "."


def render(a, b, highlight):
    del_a, ins_b = diff_lines(a, b)
    out = []
    append = out.append
    for kind, p, q in blocks(del_a, ins_b):
        if kind == "keep":
            append(b" " + a[p] + b"\n")
            continue
        dels = list(p)
        inss = list(q)
        for i in dels:
            append(b"-" + a[i] + b"\n")
        for t, j in enumerate(inss):
            append(b"+" + b[j] + b"\n")
            if highlight and t < len(dels):
                # Pair the t-th '-' line with the t-th '+' line and run the
                # same Myers diff on their characters (Unicode code points).
                old = a[dels[t]].decode("utf-8")
                new = b[j].decode("utf-8")
                c_del, c_ins = diff_marks(old, new)
                line = "? %s | %s\n" % (ranges_text(c_del), ranges_text(c_ins))
                append(line.encode("ascii"))
    return b"".join(out)


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] not in ("lines", "highlight"):
        print("usage: main.py lines|highlight A_PATH B_PATH", file=sys.stderr)
        return 2
    command, a_path, b_path = sys.argv[1:]
    try:
        a = read_lines(a_path)
        b = read_lines(b_path)
    except OSError as e:
        print("error: cannot read file: %s" % e, file=sys.stderr)
        return 2
    sys.stdout.buffer.write(render(a, b, command == "highlight"))
    sys.stdout.buffer.flush()
    return 0


raise SystemExit(main())
