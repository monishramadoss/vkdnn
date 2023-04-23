

b = 0
s = 4
N = 2 * 4 * 3 * 3
default_value = 1

with open("ssbo.txt", 'w') as f:
    for i in range(0, N):
        f.write(f"ssbo {b} subdata float {i*s} {default_value}\n")
