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

    v1[k] is the furthest x reached on diagonal k (k = x - y) by the forward
    search; v2[k] is the same for the reverse search, measured from the end.
    The arrays are indexed with an offset so negative k works.
    Returns (x, y), or None if no split was found (should not happen).
    """
    n = len(a)
    m = len(b)
    max_d = (n + m + 1) // 2
    offset = max_d
    size = 2 * max_d + 2
    v1 = [-1] * size
    v2 = [-1] * size
    v1[offset + 1] = 0
    v2[offset + 1] = 0
    delta = n - m
    # If delta is odd the paths can first meet during a forward step,
    # otherwise during a reverse step.
    front = (delta & 1) == 1
    # k1start/k1end (and k2...) shrink the diagonal range once a path runs
    # off the edge of the edit graph, so we never walk outside the grid.
    k1start = k1end = k2start = k2end = 0

    for d in range(max_d + 1):
        # ---- forward search: D-paths starting at (0, 0) ----
        for k1 in range(-d + k1start, d + 1 - k1end, 2):
            k1_off = offset + k1
            # Choose the neighbour diagonal that reaches further:
            # come down from k+1 (an insertion) or right from k-1 (a deletion).
            if k1 == -d or (k1 != d and v1[k1_off - 1] < v1[k1_off + 1]):
                x1 = v1[k1_off + 1]
            else:
                x1 = v1[k1_off - 1] + 1
            y1 = x1 - k1
            # Follow the snake: a run of equal elements (diagonal moves).
            while x1 < n and y1 < m and a[x1] == b[y1]:
                x1 += 1
                y1 += 1
            v1[k1_off] = x1
            if x1 > n:
                k1end += 2          # ran off the right edge
            elif y1 > m:
                k1start += 2        # ran off the bottom edge
            elif front:
                # Does the reverse (d-1)-path on this diagonal overlap us?
                k2_off = offset + delta - k1
                if 0 <= k2_off < size and v2[k2_off] != -1:
                    if x1 >= n - v2[k2_off]:
                        return x1, y1

        # ---- reverse search: D-paths starting at (n, m) ----
        for k2 in range(-d + k2start, d + 1 - k2end, 2):
            k2_off = offset + k2
            if k2 == -d or (k2 != d and v2[k2_off - 1] < v2[k2_off + 1]):
                x2 = v2[k2_off + 1]
            else:
                x2 = v2[k2_off - 1] + 1
            y2 = x2 - k2
            # Snake backwards from the end of both sequences.
            while x2 < n and y2 < m and a[n - x2 - 1] == b[m - y2 - 1]:
                x2 += 1
                y2 += 1
            v2[k2_off] = x2
            if x2 > n:
                k2end += 2
            elif y2 > m:
                k2start += 2
            elif not front:
                # Does the forward d-path on this diagonal overlap us?
                k1_off = offset + delta - k2
                if 0 <= k1_off < size and v1[k1_off] != -1:
                    x1 = v1[k1_off]
                    y1 = x1 - (k1_off - offset)
                    if x1 >= n - x2:
                        return x1, y1
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
    """Line diff: map every distinct line to a small integer, so comparing
    two lines is one integer comparison, then run Myers on the integers."""
    ids = {}
    a_ids = [ids.setdefault(line, len(ids)) for line in a]
    b_ids = [ids.setdefault(line, len(ids)) for line in b]
    return diff_marks(a_ids, b_ids)


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


def render(a, b):
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
        for j in inss:
            append(b"+" + b[j] + b"\n")
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
    sys.stdout.buffer.write(render(a, b))
    sys.stdout.buffer.flush()
    return 0


raise SystemExit(main())
