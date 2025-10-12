from math import exp, sqrt, pi

def cdf(z, n=1e5):
    if z <= 0:
        return round(sum([exp(-0.5 * (k / n) ** 2) for k in range(int(-4 * n), int(z * n))]) / sqrt(2 * pi) / n, 4)
    return 1 - cdf(1 - z)

for z in range(-349, 1):
    print(f'{z / 100:.2f} : {cdf(z / 100):.4f}')
