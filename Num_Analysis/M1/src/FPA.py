import math

# ==========================================
# Task 1: The Harmonic Series
# ==========================================

def harmonic_sum_forward(N):
    """
    Calculates sum(1/n) from n=1 to N.
    """
    total = 0.0
    for n in range(1, N + 1):
        total += 1.0 / n
    return total

def harmonic_sum_backward(N):
    """
    Calculates sum(1/n) from n=N down to 1.
    """
    total = 0.0
    for n in range(N, 0, -1):
        total += 1.0 / n
    return total

# ==========================================
# Task 2: Stable Variance
# ==========================================

def variance_naive(data):
    """
    Calculates variance using the unstable one-pass formula:
    Var = (Sum(x^2) / N) - (Mean)^2
    """
    N = len(data)
    if N == 0: return 0.0

    sum_of_squares = sum(x**2 for x in data)
    mean = sum(data) / N
    
    return (sum_of_squares / N) - (mean**2)

def variance_stable(data):
    """
    Calculates variance using the numerically stable Welford's algorithm.
    """
    N = len(data)
    if N == 0: return 0.0

    mean = 0.0
    M2 = 0.0
    
    for count, x in enumerate(data, 1):
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        M2 += delta * delta2
        
    return M2 / N

# ==========================================
# Task 3: Robust Quadratic Solver
# ==========================================

def solve_quadratic(a, b, c):
    """
    Solves ax^2 + bx + c = 0 handling catastrophic cancellation
    using the alternate quadratic formula.
    """
    discriminant = math.sqrt(b**2 - 4*a*c)

    if b > 0:
        x1 = (-b - discriminant) / (2 * a)
        x2 = (2 * c) / (-b - discriminant)

    elif b < 0:
        x1 = (-b + discriminant) / (2 * a)
        x2 = (2 * c) / (-b + discriminant)

    else:
        x1 = discriminant / (2 * a)
        x2 = -discriminant / (2 * a)

    return (x1, x2)


# ==========================================
# Main Execution Block (Testing)
# ==========================================
if __name__ == "__main__":
    print("--- Task 1: Harmonic Series (N=1,000,000) ---")
    fwd = harmonic_sum_forward(1000000)
    bwd = harmonic_sum_backward(1000000)
    print(f"Forward Sum:  {fwd:.20f}")
    print(f"Backward Sum: {bwd:.20f}")
    print(f"Difference:   {abs(fwd - bwd):.20f}")

    print("\n--- Task 2: Variance Calculation ---")
    test_data = [1e9 + 0.1, 1e9 + 0.2, 1e9 + 0.3]

    v_naive = variance_naive(test_data)
    v_stable = variance_stable(test_data)

    print(f"Naive Variance:  {v_naive}")
    print(f"Stable Variance: {v_stable}")

    print("\n--- Task 3: Quadratic Formula ---")
    a, b, c = 1.0, 10**8, 1.0
    r1, r2 = solve_quadratic(a, b, c)
    print(f"Roots for a={a}, b={b}, c={c}:")
    print(f"Root 1 (Standard formula, safe): {r1}")
    print(f"Root 2 (Alternate formula, safe): {r2}")
