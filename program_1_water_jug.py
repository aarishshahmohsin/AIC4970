"""
bezout's identity
d = ax + by is only solvable when
d is also divisible by gcd(a, b)

a, b -> jug capacities
x, y -> integers (no of fills/empties/transfers)

"""

"""
let S = {ax + by | x, y belongs to Z, ax + by > 0}

S belongs to N, and s is non empty 
by well ordering principle(a set of non negative integers contains a smallest element) -> d 
then d = ax_0 + by_0 (x_0, y_0)

d | a and d | b 
proof:

a = qd + r 
r = a - qd = a - q(ax0 + by0) = a (1 - qx0) + b(-qy0) 

if r > 0 then -> r belongs to S, if r < d then it contradicts that d is the smallest non neg int in S.

therefore, r = 0. and hence d | a and d | b 

d is a common divisor of a and b 
since d = ax + by, any other common divisor of a and b must divide d. 

therefore, d = gcd(a, b) 

"""


def gcd(a: int, b: int) -> int:
    if b == 0:
        return a
    return gcd(b, a % b)


def water_jug_solution(from_cap: int, to_cap: int, target: int, log: bool = False, flag = False):
    from_jug = from_cap
    to_jug = 0
    steps = 1

    while from_jug != target and to_jug != target:
        if log:
            if flag:
                print(f"1. Current Capacities {from_jug/10}/{from_cap/10} and {to_jug/10}/{to_cap/10}")
            else:
                print(f"1. Current Capacities {from_jug}/{from_cap} and {to_jug}/{to_cap}")

        transfer = min(from_jug, to_cap - to_jug)

        to_jug += transfer
        from_jug -= transfer
        steps += 1

        if log:
            if flag:
                print(f"2. Current Capacities {from_jug/10}/{from_cap/10} and {to_jug/10}/{to_cap/10}")
            else:
                print(f"2. Current Capacities {from_jug}/{from_cap} and {to_jug}/{to_cap}")

        if from_jug == target or to_jug == target:
            break

        if from_jug == 0:
            from_jug = from_cap
            steps += 1

        if to_jug == to_cap:
            to_jug = 0
            steps += 1

    if from_jug != target: from_jug = 0 
    if to_jug != target: to_jug = 0
    if flag:
        print(f"2. Current Capacities {from_jug/10}/{from_cap/10} and {to_jug/10}/{to_cap/10}")
    else:
        print(f"3. Current Capacities {from_jug}/{from_cap} and {to_jug}/{to_cap}")
    return steps


def is_solvable(a: int, b: int, d: int) -> bool:
    if d > max(a, b):
        return False
    return d % gcd(a, b) == 0


if __name__ == "__main__":
    n = float(input("Enter first jar capacity: "))
    m = float(input("Enter second jar capacity: "))

    d = float(input("Enter the amount to be measured: "))
    flag = 0 

    if float(m) != int(m) or float(n) != int(n):
        flag = 1
        n *= 10 
        m *= 10 
        d *= 10 
        
    if m == 0 or n == 0:
        print("container has 0 capacity")
        exit(1)


    if not is_solvable(n, m, d):
        print("Measurement not possible with given jar sizes.")
    else:
        steps1 = water_jug_solution(n, m, d, log=True)
        # print("-" * 50)
        # steps2 = water_jug_solution(m, n, d, log=True)
        # print("-" * 50)
        # print(f"Minimum steps required: {min(steps1, steps2)}")
