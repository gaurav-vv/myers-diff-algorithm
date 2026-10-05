"""Myers' diff: line diff (Part A) and changed-character highlight (Part B).

Usage:
    main.py lines     A B   # Part A: minimal line diff
    main.py highlight A B   # Part B: line diff + changed-character ranges
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
    # TODO: compute the diff of a and b and print the listing.
    return 0


raise SystemExit(main())
