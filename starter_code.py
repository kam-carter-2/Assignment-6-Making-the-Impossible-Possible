
"""
Dynamic Programming Assignment - Longest Common Subsequence
Compare naive recursion, memoization, and bottom-up tabulation.
"""

import json
import time
from functools import lru_cache


# ============================================================================
# PART 1: NAIVE RECURSIVE SOLUTION
# ============================================================================

def lcs_recursive(seq1, seq2):
    """Return the LCS length using pure recursion (exponential time)."""

    # An empty sequence has no common subsequence.
    if not seq1 or not seq2:
        return 0

    # Matching final characters can be included in the LCS.
    if seq1[-1] == seq2[-1]:
        return 1 + lcs_recursive(seq1[:-1], seq2[:-1])

    # Otherwise, try removing the final character from either sequence.
    return max(
        lcs_recursive(seq1[:-1], seq2),
        lcs_recursive(seq1, seq2[:-1])
    )


# ============================================================================
# PART 2: MEMOIZATION (TOP-DOWN WITH CACHING)
# ============================================================================

def lcs_memoization(seq1, seq2):
    """Return the LCS length using recursion and cached subproblems."""

    @lru_cache(maxsize=None)
    def solve(i, j):
        # No characters remain in one of the prefixes.
        if i == 0 or j == 0:
            return 0

        # Matching characters extend the LCS of smaller prefixes.
        if seq1[i - 1] == seq2[j - 1]:
            return 1 + solve(i - 1, j - 1)

        # Reuse cached results for both alternatives.
        return max(solve(i - 1, j), solve(i, j - 1))

    return solve(len(seq1), len(seq2))


# ============================================================================
# PART 3: TABULATION (BOTTOM-UP WITH TABLE)
# ============================================================================

def lcs_tabulation(seq1, seq2):
    """Return the LCS length using an iterative dynamic programming table."""

    m, n = len(seq1), len(seq2)

    # dp[i][j] stores the LCS length for the first i and j characters.
    # Initializing to zero handles empty-prefix base cases.
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Build solutions from shorter prefixes to longer prefixes.
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


# ============================================================================
# TESTING & TIMING
# ============================================================================

def load_sequence(filename):
    """Load DNA sequence data from the sequences directory."""
    with open(f"sequences/{filename}", "r") as f:
        return json.load(f)


def test_small_cases():
    """Test all implementations on small known cases."""
    print("=" * 70)
    print("TESTING ON SMALL CASES")
    print("=" * 70)

    test_cases = [
        ("AGGTAB", "GXTXAYB", 4),
        ("ABCDGH", "AEDFHR", 3),
        ("ABC", "AC", 2),
        ("", "ABC", 0),
    ]

    for seq1, seq2, expected in test_cases:
        print(f"\nTest: {seq1!r} vs {seq2!r}")
        print(f"Expected LCS length: {expected}")

        for name, function in [
            ("Recursive", lcs_recursive),
            ("Memoization", lcs_memoization),
            ("Tabulation", lcs_tabulation),
        ]:
            try:
                result = function(seq1, seq2)
                status = "PASS" if result == expected else "FAIL"
                print(f"  {name}: {result} {status}")
            except Exception as error:
                print(f"  {name}: ERROR - {error}")


def time_recursive():
    """Time naive recursion on small sequences only."""
    print("\n" + "=" * 70)
    print("TIMING RECURSIVE SOLUTION")
    print("=" * 70)

    for size in [10, 20, 50]:
        data = load_sequence(f"dna_{size}.json")
        seq1, seq2 = data["sequence1"], data["sequence2"]

        print(f"\nSequence length: {size}")
        start = time.perf_counter()

        try:
            result = lcs_recursive(seq1, seq2)
            elapsed = time.perf_counter() - start
            print(f"  LCS length: {result}")
            print(f"  Time: {elapsed:.4f} seconds")
        except RecursionError:
            print("  ERROR: recursion limit reached")
            break

        if elapsed > 5:
            print("  Stopping: naive recursion is too slow.")
            break


def compare_all_approaches():
    """Benchmark each approach on the provided sequence sizes."""
    print("\n" + "=" * 76)
    print("COMPARING ALL APPROACHES")
    print("=" * 76)
    print(f"{'Size':<10} {'Recursive':<18} {'Memoization':<18} {'Tabulation':<18}")
    print("-" * 76)

    for size in [10, 20, 50, 100, 200, 500, 1000]:
        data = load_sequence(f"dna_{size}.json")
        seq1, seq2 = data["sequence1"], data["sequence2"]
        results = {}

        # Naive recursion is intentionally skipped for larger sequences.
        if size <= 20:
            try:
                start = time.perf_counter()
                lcs_recursive(seq1, seq2)
                results["recursive"] = f"{time.perf_counter() - start:.4f}s"
            except (RecursionError, Exception):
                results["recursive"] = "ERROR"
        else:
            results["recursive"] = "Skipped"

        # Memoization is fast, but recursive depth may limit large inputs.
        try:
            start = time.perf_counter()
            lcs_memoization(seq1, seq2)
            results["memoization"] = f"{time.perf_counter() - start:.4f}s"
        except (RecursionError, Exception):
            results["memoization"] = "ERROR"

        # Tabulation avoids recursive-call depth limitations.
        try:
            start = time.perf_counter()
            lcs_tabulation(seq1, seq2)
            results["tabulation"] = f"{time.perf_counter() - start:.4f}s"
        except Exception:
            results["tabulation"] = "ERROR"

        print(
            f"{size:<10} "
            f"{results['recursive']:<18} "
            f"{results['memoization']:<18} "
            f"{results['tabulation']:<18}"
        )


if __name__ == "__main__":
    print("DYNAMIC PROGRAMMING ASSIGNMENT - LCS\n")

    test_small_cases()
    compare_all_approaches()
