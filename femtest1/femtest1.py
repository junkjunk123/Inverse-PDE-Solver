#This test uses FEM-based techniques to solve the Poisson equation \Delta u = f
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

def solve(nx, ny, f):
    nodes, elements = mesh(nx, ny)
    K = build_stiffness(nodes, elements)
    F = build_load(nodes, elements, f)
    F, K = boundary_conditions(nodes, F, K)
    K = csr_matrix(K)
    u = spsolve(K, F)
    return u

def build_load(nodes, elements, f):
    F = np.zeros(len(nodes))

    for e in elements:
        x1 = nodes[e[0]]
        x2 = nodes[e[1]]
        x3 = nodes[e[2]]
        centroid = (x1 + x2 + x3) / 3
        approx_integral = f(centroid[0], centroid[1]) * area(x1, x2, x3) / 3
        F[e[0]] += approx_integral
        F[e[1]] += approx_integral
        F[e[2]] += approx_integral
    return F

def boundary_conditions(nodes, F, K):
    for i in range(len(nodes)):
        n = nodes[i]
        if n[0] == 0 or n[0] == 1 or n[1] == 0 or n[1] == 1:
            for j in range(len(K[i])):
                if j != i:
                    K[i][j] = 0
                    K[j][i] = 0
                else:
                    K[i][j] = 1
            F[i] = 0
    return F, K

def build_stiffness(nodes, elements):
    K = np.zeros((len(nodes), len(nodes)))

    def compute_local_matrix(x1, x2, x3):
        K_ij = stiffness(0, 1, x1, x2, x3)
        K_jk = stiffness(1, 2, x1, x2, x3)
        K_ik = stiffness(0, 2, x1, x2, x3)
        K_ii = stiffness(0, 0, x1, x2, x3)
        K_jj = stiffness(1, 1, x1, x2, x3)
        K_kk = stiffness(2, 2, x1, x2, x3)

        return np.array([
            [K_ii, K_ij, K_ik],
            [K_ij, K_jj, K_jk],
            [K_ik, K_jk, K_kk]
        ])

    for i in range(len(elements)):
        local_K = compute_local_matrix(nodes[elements[i][0]], nodes[elements[i][1]], nodes[elements[i][2]])

        for a in [0, 1, 2]:
            for b in [0, 1, 2]:
                K[elements[i][a], elements[i][b]] += local_K[a][b]
    return K


def mesh(nx, ny):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    nodes = []
    for i in range(nx):
        for j in range(ny):
            nodes.append([x[i], y[j]])

    def idx(i, j):
        return j * nx + i

    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = idx(i, j)
            n1 = idx(i + 1, j)
            n2 = idx(i, j + 1)
            n3 = idx(i + 1, j + 1)

            elements.append([n0, n1, n3])
            elements.append([n0, n3, n2]) #keep counter-clockwise orientation for integration
    return np.array(nodes), np.array(elements)

def area(x1, x2, x3): #area of a triangle
    triangle_matrix = np.array([
        [1, x1[0], x1[1]],
        [1, x2[0], x2[1]],
        [1, x3[0], x3[1]]
    ])
    det = np.linalg.det(triangle_matrix)
    return 0.5 * np.abs(det)

def basis_grad(i, x1, x2, x3, area):
    mult = 1 / (2 * area)
    if i == 0:
        return np.array([x2[1] - x3[1], x3[0] - x2[0]]) * mult
    elif i == 1:
        return np.array([x3[1] - x1[1], x1[0] - x3[0]]) * mult
    return np.array([x1[1] - x2[1], x2[0] - x1[0]]) * mult

def stiffness(i, j, x1, x2, x3):
    a = area(x1, x2, x3)
    grad_i = basis_grad(i, x1, x2, x3, a)
    grad_j = basis_grad(j, x1, x2, x3, a)
    return np.dot(grad_i, grad_j) * a

if __name__ == '__main__':
    f = lambda x, y: np.exp(-50 * ((x - 0.5) ** 2  + (y - 0.5) ** 2))
    u = solve(30, 30, f)
    print(u)