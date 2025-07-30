'''
bezout's identity 
d = ax + by is only solvable when 
d is also divisible by gcd(a, b)

a, b -> jug capacities 
x, y -> integers (no of fills/empties/transfers)

'''

def gcd(a: int, b: int) -> int:
    if b == 0:
        return a
    return gcd(b, a % b)

def water_jug_solution(from_cap: int, to_cap: int, target: int, log: bool = False):
    from_jug = from_cap
    to_jug = 0
    steps = 1

    while from_jug != target and to_jug != target:
        if log: 
            print(f"Current Capacities {from_jug}/{from_cap} and {to_jug}/{to_cap}")

        transfer = min(from_jug, to_cap - to_jug)

        to_jug += transfer
        from_jug -= transfer
        steps += 1

        if log: 
            print(f"Current Capacities {from_jug}/{from_cap} and {to_jug}/{to_cap}")

        if from_jug == target or to_jug == target:
            break

        if from_jug == 0:
            from_jug = from_cap
            steps += 1

        if to_jug == to_cap:
            to_jug = 0
            steps += 1

    return steps


def is_solvable(a: int, b: int, d: int) -> bool:
    if d > max(a, b):
        return False
    return d % gcd(a, b) == 0


if __name__ == "__main__":
    n = int(input("Enter first jar capacity: "))
    m = int(input("Enter second jar capacity: "))
    d = int(input("Enter the amount to be measured: "))

    if n > m:
        n, m = m, n  

    if not is_solvable(n, m, d):
        print("Measurement not possible with given jar sizes.")
    else:
        steps1 = water_jug_solution(n, m, d, log=True)
        print("-" * 50)
        steps2 = water_jug_solution(m, n, d, log=True)
        print("-" * 50)
        print(f"Minimum steps required: {min(steps1, steps2)}")
