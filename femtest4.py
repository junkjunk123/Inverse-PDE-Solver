
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def solve(nx, ny, f):
    mesh_data = taylor_hood_mesh(nx, ny)
    K, F = assemble_system(mesh_data, f)
    nv = mesh_data['num_vel_nodes']
    F, K = boundary_conditions(mesh_data['vel_nodes'], nv, F, K)

    # 4. Solve
    K_sparse = csr_matrix(K)
    U = spsolve(K_sparse, F)

    np_nodes = mesh_data['num_pres_nodes']

    ux = U[0:nv]
    uy = U[nv:2 * nv]
    p = U[2 * nv:2 * nv + np_nodes]

    return ux, uy, p

def boundary_conditions(nodes, nv, F, K):
    n = len(nodes)

    # 1. Apply Dirichlet boundary conditions to Velocity degrees of freedom
    for i in range(n):
        node = nodes[i]
        if np.abs(node[0]) < 1e-7 or np.abs(node[0] - 1) < 1e-7 or np.abs(node[1]) < 1e-7 or np.abs(node[1] - 1) < 1e-7:
            for dof in [i, i + n]:
                K[dof, :] = 0 #note that clearing the whole way causes singularities, we don't want to clear the pressure columns
                K[dof, dof] = 1
                F[dof] = 0  # No-slip boundary condition (u = 0)

    # 2. Pin a single pressure node to eliminate the hydrostatic constant pressure nullspace
    p_pin = 2 * nv
    K[p_pin, :] = 0
    K.T[p_pin, :] = 0
    K[p_pin, p_pin] = 1
    F[p_pin] = 0

    return F, K

def taylor_hood_mesh(nx, ny):
    #P2/P1 mesh; note that velocity uses midpoint nodes for quadratic basis but pressure only uses vertices
    nvx = 2 * nx - 1
    nvy = 2 * ny - 1
    x_vel = np.linspace(0, 1, nvx)
    y_vel = np.linspace(0, 1, nvy)

    vel_nodes = []
    for i in range(nvx):
        for j in range(nvy):
            vel_nodes.append([x_vel[i], y_vel[j]])
    vel_nodes = np.array(vel_nodes)

    def v_idx(i, j):
        return i * nvy + j

    def p_idx(i, j):
        return i * ny + j

    x_pres = np.linspace(0, 1, nx)
    y_pres = np.linspace(0, 1, ny)

    pres_nodes = []
    for i in range(nx):
        for j in range(ny):
            pres_nodes.append([x_pres[i], y_pres[j]])
    pres_nodes = np.array(pres_nodes)

    elements = []

    for i in range(nx - 1):
        for j in range(ny - 1):
            #pressure indices
            p0 = p_idx(i, j)
            p1 = p_idx(i + 1, j)
            p2 = p_idx(i, j + 1)
            p3 = p_idx(i + 1, j + 1)

            #velocity vertex nodes
            v00 = v_idx(2 * i, 2 * j)
            v20 = v_idx(2 * i + 2, 2 * j)
            v02 = v_idx(2 * i, 2 * j + 2)
            v22 = v_idx(2 * i + 2, 2 * j + 2)

            #velocity midpoint nodes
            v10 = v_idx(2 * i + 1, 2 * j)
            v01 = v_idx(2 * i, 2 * j + 1)
            v21 = v_idx(2 * i + 2, 2 * j + 1)
            v12 = v_idx(2 * i + 1, 2 * j + 2)
            v11 = v_idx(2 * i + 1, 2 * j + 1)

            elements.append({
                'p_nodes': [p0, p1, p3],
                'v_nodes': [v00, v20, v22, v10, v21, v11]
            })
            elements.append({
                'p_nodes': [p0, p3, p2],
                'v_nodes': [v00, v22, v02, v11, v12, v01]
            })

    return {
        'vel_nodes': vel_nodes,
        'pres_nodes': pres_nodes,
        'elements': elements,
        'num_vel_nodes': len(vel_nodes),
        'num_pres_nodes': len(pres_nodes)
    }

def quadratic_basis(xi, eta):
    bound = 1 - xi - eta #nodes 1, 2, and 4 lie on zero set of this expression
    N = np.zeros(6)
    N[0] = bound * (1 - 2 * xi - 2 * eta) #(0,0)
    N[1] = xi * (2 * xi - 1) #(1,0)
    N[2] = eta * (2 * eta - 1) #(0,1)
    N[3] = 4 * xi * bound #(1/2, 0)
    N[4] = 4 * xi * eta #(1/2, 1/2)
    N[5] = 4 * eta * bound #(0, 1/2)
    #Nodes 0-2 and 3-5 follow standard counter-clockwise orientation

    #\frac{\partial N / \partial \xi}
    partial_xi = np.array([-3 + 4 * xi + 4 * eta, 4 * xi - 1, 0, 4 - 8 * xi - 4 * eta, 4 * eta, -4 * eta])

    # \frac{\partial N / \partial \eta}
    partial_eta = np.array([-3 + 4 * xi + 4 * eta, 0, 4 * eta - 1, -4 * xi, 4 * xi, 4 - 4 * xi - 8 * eta])

    return N, partial_xi, partial_eta

def assemble_system(mesh, f):
    nv_nodes = mesh['num_vel_nodes']
    np_nodes = mesh['num_pres_nodes']
    dof = 2 * nv_nodes + np_nodes
    K_global = np.zeros((dof, dof))
    F_global = np.zeros(dof)

    #3-point symmetric Gauss Quadrature for triangles
    quad_points = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
    quad_weights = np.array([1 / 6, 1 / 6, 1 / 6])

    for element in mesh['elements']:
        v_idx = element['v_nodes']
        p_idx = element['p_nodes']
        v_coords = mesh['vel_nodes'][v_idx[0:3]]
        x1, x2, x3 = v_coords[0], v_coords[1], v_coords[2]

        # Since (xi, eta) are reference coordinates, we map them to spacial coordinates. Let g be the map (xi, eta) -> (x, y).
        # Compute J = Dg, the Jacobian map
        J = np.array([
            [x2[0] - x1[0], x3[0] - x1[0]],
            [x2[1] - x1[1], x3[1] - x1[1]]
        ])
        detJ = np.abs(np.linalg.det(J))
        invJ = np.linalg.inv(J)

        Ak = np.zeros((6, 6))
        Bx = np.zeros((3, 6))
        By = np.zeros((3, 6))
        Fx = np.zeros(6)
        Fy = np.zeros(6)

        #Numerical integration
        for q in range(3):
            xi, eta = quad_points[q][0], quad_points[q][1]
            w = quad_weights[q] * detJ

            # 1. Evaluate P2 Velocity Shapes
            N, partial_xi, partial_eta = quadratic_basis(xi, eta)
            gradN = np.zeros((6, 2))

            for k in range(6):
                gradN[k] = invJ.T @ np.array([partial_xi[k], partial_eta[k]])

            psi = np.array([1 - xi - eta, xi, eta])
            pos = (1 - xi - eta)*x1 + xi*x2 + eta*x3
            val = f(pos[0], pos[1])

            for i in range(6):
                Fx[i] += N[i] * val[0] * w
                Fy[i] += N[i] * val[1] * w
                for j in range(6):
                    Ak[i, j] += np.dot(gradN[i], gradN[j]) * w

            for i in range(3):
                for j in range(6):
                    Bx[i, j] += psi[i] * gradN[j][0] * w
                    By[i, j] += psi[i] * gradN[j][1] * w

        for i in range(6):
            row_v = v_idx[i]
            F_global[row_v] += Fx[i]
            F_global[row_v + nv_nodes] += Fy[i]

            for j in range(6):
                col_v = v_idx[j]
                K_global[row_v, col_v] += Ak[i, j]  #Ax
                K_global[row_v + nv_nodes, col_v + nv_nodes] += Ak[i, j]  #Ay

        for i in range(3):
            row_p = p_idx[i] + 2 * nv_nodes

            for j in range(6):
                col_v = v_idx[j]

                K_global[row_p, col_v] += Bx[i, j]
                K_global[row_p, col_v + nv_nodes] += By[i, j]

                K_global[col_v, row_p] += Bx[i, j]
                K_global[col_v + nv_nodes, row_p] += By[i, j]

    return K_global, F_global

if __name__ == "__main__":
    f = lambda x, y: [np.exp(-50 * ((x - 0.4) ** 2 + (y - 0.6) ** 2)), np.exp(-50 * ((x - 0.6) ** 2 + (y - 0.4) ** 2))]
    ux, uy, p = solve(30, 30, f)
    print(ux)
    print(uy)
    print(p)