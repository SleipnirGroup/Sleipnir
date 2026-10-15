def near(expected: float, actual: float, tolerance: float) -> bool:
    return abs(expected - actual) < tolerance
