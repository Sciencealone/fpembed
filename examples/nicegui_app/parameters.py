"""Parameter validation and compression calculation utilities."""

import math


def is_power_of_two(n: int) -> bool:
    """Check if n is a power of 2.

    Uses bitwise operation: a power of 2 has exactly one bit set.

    Args:
        n: Integer to check.

    Returns:
        True if n is a power of 2, False otherwise.

    Examples:
        >>> is_power_of_two(4)
        True
        >>> is_power_of_two(7)
        False
    """
    return n > 0 and (n & (n - 1)) == 0


def calculate_valid_compressions(
    fp_size: int,
    min_comp: int,
    max_divisor: int,
) -> list:
    """Calculate valid compression factors for a given fingerprint size.

    Validation rules:
        1. Must be power of 2.
        2. Must divide fp_size evenly.
        3. Must be >= min_comp.
        4. Must be <= fp_size / max_divisor.

    Args:
        fp_size: Fingerprint size parameter.
        min_comp: Minimum compression factor.
        max_divisor: Divisor for maximum compression.

    Returns:
        Sorted list of valid compression factors.

    Examples:
        >>> calculate_valid_compressions(1024, 4, 2)
        [4, 8, 16, 32, 64, 128, 256, 512]
    """
    max_comp = fp_size // max_divisor
    valid = []
    compression = min_comp
    while compression <= max_comp:
        if is_power_of_two(compression) and fp_size % compression == 0:
            valid.append(compression)
        compression *= 2
    return sorted(valid)


def filter_compressions_by_granularity(
    fp_size: int,
    granularity_pct: int,
    min_comp: int,
    max_divisor: int,
) -> list[int]:
    """Select an equally-spaced subset of valid compressions by granularity %.

    Always includes 0 (FP, no compression) in the returned list.

    Args:
        fp_size: The sampled fingerprint size.
        granularity_pct: 10-100 in steps of 10.
        min_comp: Minimum compression (from config, e.g. 4).
        max_divisor: Max compression divisor (from config, e.g. 2).

    Returns:
        List of valid compression values including 0.

    Examples:
        >>> filter_compressions_by_granularity(1024, 100, 4, 2)
        [0, 4, 8, 16, 32, 64, 128, 256, 512]
        >>> filter_compressions_by_granularity(1024, 50, 4, 2)
        [0, 4, 16, 64, 256]
    """
    all_compressions = calculate_valid_compressions(fp_size, min_comp, max_divisor)

    if not all_compressions:
        return [0]

    if granularity_pct >= 100:
        return [0] + all_compressions

    total = len(all_compressions)
    subset_size = max(1, math.ceil(total * granularity_pct / 100))

    if subset_size >= total:
        return [0] + all_compressions

    indices = [
        round(i * (total - 1) / (subset_size - 1)) if subset_size > 1 else 0
        for i in range(subset_size)
    ]
    return [0] + [all_compressions[i] for i in indices]
