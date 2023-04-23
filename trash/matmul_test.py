

TILE_DIM = 16
ATile = [[0 for _ in range(TILE_DIM)] for _ in range(TILE_DIM)]
BTile = [[0 for _ in range(TILE_DIM)] for _ in range(TILE_DIM)]

def matmul(m, n, k, A, B, C, thrX, thrY, col, row):
    elementC = 0
    for t in range((n-1) / TILE_DIM):
        if row < m and t * TILE_DIM + thrX < n:
            ATile[thrX][thrY] = A[row * n + t * TILE_DIM + thrX]
        else:   
            ATile[thrX][thrY] = 0
        if t * TILE_DIM + thrY < n and col < k:
            BTile[thrY][thrX] = B[(t * TILE_DIM + thrY) + col]
        else:
            BTile[thrY][thrX] = 0
        for i in range(TILE_DIM):
            elementC += ATile[thrY][i] * BTile[i][thrX]
    if row < m and col < k:
        C[row*k + col] = elementC

def main():
    m = 32
    n = 32
    k = 32
    A = [0 for _ in range(m*n)]
    B = [0 for _ in range(n*k)]
    C = [0 for _ in range(m*k)]



    pass

if __name__ == "__main__":
    main()