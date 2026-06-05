#This test uses FEM with P1-P1 mesh to solve the Stokes Equations -\mu \Delta u + \nabla p = f; div(u) = 0, for u : R^2 -> R^2 and p : R^2 -> R
#Note that the pressure evolution in the solution exhibits numerical instability (lack of smoothness, rapid oscillation), which demonstrates the effects of
#violating the Ladyzhenskaya–Babuška–Brezzi (LBB) condition.
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def solve(nx, ny, f):
    nodes, elements = mesh(nx, ny)
    K = build_stiffness(nodes, elements)
    F = build_load(nodes, elements, f)
    F, K = boundary_conditions(nodes, F, K)

    K_sparse = csr_matrix(K)
    U = spsolve(K_sparse, F)

    # Extract solutions
    n = len(nodes)
    u_x = U[0:n]
    u_y = U[n:2 * n]
    p = U[2 * n:3 * n]

    return u_x, u_y, p


def mesh(nx, ny):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    nodes = []
    for i in range(nx):
        for j in range(ny):
            nodes.append([x[i], y[j]])

    def idx(i, j):
        return i * ny + j

    elements = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            n0 = idx(i, j)
            n1 = idx(i + 1, j)
            n2 = idx(i, j + 1)
            n3 = idx(i + 1, j + 1)

            # Counter-clockwise triangles
            elements.append([n0, n1, n3])
            elements.append([n0, n3, n2])

    return np.array(nodes), np.array(elements)


def area(x1, x2, x3):
    triangle_matrix = np.array([
        [1, x1[0], x1[1]],
        [1, x2[0], x2[1]],
        [1, x3[0], x3[1]]
    ])
    det = np.linalg.det(triangle_matrix)
    return 0.5 * np.abs(det)


def basis_grad(i, x1, x2, x3, a):
    mult = 1 / (2 * a)
    if i == 0:
        return np.array([x2[1] - x3[1], x3[0] - x2[0]]) * mult
    elif i == 1:
        return np.array([x3[1] - x1[1], x1[0] - x3[0]]) * mult
    return np.array([x1[1] - x2[1], x2[0] - x1[0]]) * mult


def compute_local_matrices(x1, x2, x3):
    a = area(x1, x2, x3)
    Ak = np.zeros((3, 3))
    Bx = np.zeros((3, 3))
    By = np.zeros((3, 3))

    for i in range(3):
        grad_i = basis_grad(i, x1, x2, x3, a)
        for j in range(3):
            grad_j = basis_grad(j, x1, x2, x3, a)

            # Viscosity / Laplacian block entries
            Ak[i, j] = np.dot(grad_i, grad_j) * a

            # Divergence block entries:
            # Row j represents pressure test function, Col i represents velocity trial function
            Bx[j, i] = (a / 3.0) * grad_i[0]
            By[j, i] = (a / 3.0) * grad_i[1]

    return Ak, Bx, By


def build_stiffness(nodes, elements):
    n = len(nodes)
    K_global = np.zeros((3 * n, 3 * n))

    for e in elements:
        x1, x2, x3 = nodes[e[0]], nodes[e[1]], nodes[e[2]]
        Ak, Bx, By = compute_local_matrices(x1, x2, x3)

        for i in range(3):
            for j in range(3):
                row_node = e[i]
                col_node = e[j]

                # 1. Velocity Laplacian Blocks (A_x and A_y)
                K_global[row_node, col_node] += Ak[i, j]  # Top-left (ux)
                K_global[row_node + n, col_node + n] += Ak[i, j]  # Center (uy)

                # 2. Gradient Blocks (B_x^T and B_y^T)
                K_global[row_node, col_node + 2 * n] += Bx[j, i]  # ux row, p col
                K_global[row_node + n, col_node + 2 * n] += By[j, i]  # uy row, p col

                # 3. Divergence Blocks (B_x and B_y)
                K_global[row_node + 2 * n, col_node] += Bx[i, j]  # p row, ux col
                K_global[row_node + 2 * n, col_node + n] += By[i, j]  # p row, uy col

    return K_global


def build_load(nodes, elements, f):
    n = len(nodes)
    F = np.zeros(3 * n)

    for e in elements:
        x1, x2, x3 = nodes[e[0]], nodes[e[1]], nodes[e[2]]
        centroid = (x1 + x2 + x3) / 3
        forcing = f(centroid[0], centroid[1])
        a = area(x1, x2, x3)

        approx_integral_x = forcing[0] * a / 3
        approx_integral_y = forcing[1] * a / 3

        for k in range(3):
            F[e[k]] += approx_integral_x
            F[e[k] + n] += approx_integral_y
            # Pressure RHS (F[e[k] + 2*n]) remains 0 because div(u) = 0
    return F

def boundary_conditions(nodes, F, K):
    n = len(nodes)

    # 1. Apply Dirichlet boundary conditions to Velocity degrees of freedom
    for i in range(n):
        node = nodes[i]
        if np.abs(node[0]) < 1e-7 or np.abs(node[0] - 1) < 1e-7 or np.abs(node[1]) < 1e-7 or np.abs(node[1] - 1) < 1e-7:
            for dof in [i, i + n]:
                K[dof, 0:2*n] = 0 #note that clearing the whole way causes singularities, we don't want to clear the pressure columns
                K[dof, dof] = 1
                F[dof] = 0  # No-slip boundary condition (u = 0)

    # 2. Pin a single pressure node to eliminate the hydrostatic constant pressure nullspace
    p_pin = 2 * n
    K[p_pin, :] = 0
    K[:, p_pin] = 0
    K[p_pin, p_pin] = 1
    F[p_pin] = 0

    return F, K

if __name__ == "__main__":
    f = lambda x, y: [np.exp(-50 * ((x - 0.4) ** 2 + (y - 0.6) ** 2)), np.exp(-50 * ((x - 0.6) ** 2 + (y - 0.4) ** 2))]
    ux, uy, p = solve(30, 30, f)
    print(ux)
    print(uy)
    print(p)