#This test uses FEM-based techniques to solve the vector-valued Poisson equation \Delta u = f, u : R^2 -> R^2
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

def solve(nx, ny, f):
    nodes, elements = mesh(nx, ny)
    K = build_stiffness(nodes, elements)
    F = build_load(nodes, elements, f)
    F, K = boundary_conditions(nodes, F, K)
    K = csr_matrix(K)
    U = spsolve(K, F)
    return U

def build_load(nodes, elements, f):
    F = np.zeros(len(nodes) * 2)

    for e in elements:
        x1 = nodes[e[0]]
        x2 = nodes[e[1]]
        x3 = nodes[e[2]]
        centroid = (x1 + x2 + x3) / 3
        forcing = f(centroid[0], centroid[1])
        approx_integral_x, approx_integral_y = np.array([forcing[0], forcing[1]]) * area(x1, x2, x3) / 3
        for k in range(3):
            F[e[k]] += approx_integral_x
            F[e[k] + len(nodes)] += approx_integral_y
    return F

def boundary_conditions(nodes, F, K):
    n = len(nodes)
    for i in range(n):
        node = nodes[i]
        if node[0] == 0 or node[0] == 1 or node[1] == 0 or node[1] == 1:
            for dof in [i, i + n]:
                for j in range(len(K[dof])):
                    if j != dof:
                        K[dof][j] = 0
                        K[j][dof] = 0
                    else:
                        K[dof][j] = 1
                F[dof] = 0
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
    n = len(K)
    return np.block([
        [K, np.zeros((n, n))],
        [np.zeros((n, n)), K]
    ])


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
    pi = np.pi
    f = lambda x, y: (2 * pi ** 2 * np.sin(pi * x) * np.sin(pi * y),
                      2 * pi ** 2 * np.sin(pi * x) * np.sin(pi * y))
    u = solve(30, 30, f)

    # Check against true solution at interior nodes
    nodes, _ = mesh(30, 30)
    n = len(nodes)
    u_true_x = np.array([np.sin(pi * nd[0]) * np.sin(pi * nd[1]) for nd in nodes])
    u_true_y = u_true_x.copy()

    u_x = u[:n]
    u_y = u[n:]

    print("Max error x:", np.max(np.abs(u_x - u_true_x)))
    print("Max error y:", np.max(np.abs(u_y - u_true_y)))