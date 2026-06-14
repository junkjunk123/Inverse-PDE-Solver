#This test adds Jax-based autodifferentiation capabilities to the Stokes flow solver from femtest4.py. Although the code works functionality-wise, the solver takes ~21 minutes to solve the problem, which suggests that some optimizations would be helpful.
import time

import jax
import jax.numpy as jnp
from jaxopt.linear_solve import solve_cg

def solve(nx, ny, f):
    mesh_data = taylor_hood_mesh(nx, ny)
    K, F = assemble_system(mesh_data, f)
    nv = mesh_data['num_vel_nodes']
    F, K = boundary_conditions(mesh_data['vel_nodes'], nv, F, K)

    # 4. Solve
    U = solve_cg(K, F)

    np_nodes = mesh_data['num_pres_nodes']

    ux = U[0:nv]
    uy = U[nv:2 * nv]
    p = U[2 * nv:2 * nv + np_nodes]

    return ux, uy, p

def boundary_conditions(nodes, nv, F, K):
    n = K.shape[0]
    bc_idx = []

    for i, node in enumerate(nodes):
        if (jnp.abs(node[0]) < 1e-7 or
                jnp.abs(node[0] - 1) < 1e-7 or
                jnp.abs(node[1]) < 1e-7 or
                jnp.abs(node[1] - 1) < 1e-7):
            bc_idx.append(i)
            bc_idx.append(i + nv)
    bc_idx = jnp.array(bc_idx)
    p_pin = 2 * nv
    bc_idx = jnp.concatenate([bc_idx, jnp.array([p_pin])])
    mask = jnp.zeros(n).at[bc_idx].set(1.0)
    I = jnp.eye(n)
    K = (1 - mask)[:, None] * K + mask[:, None] * I
    F = F * (1 - mask)

    return F, K

def taylor_hood_mesh(nx, ny):
    #P2/P1 mesh; note that velocity uses midpoint nodes for quadratic basis but pressure only uses vertices
    nvx = 2 * nx - 1
    nvy = 2 * ny - 1
    x_vel = jnp.linspace(0, 1, nvx)
    y_vel = jnp.linspace(0, 1, nvy)

    vel_nodes = []
    for i in range(nvx):
        for j in range(nvy):
            vel_nodes.append([x_vel[i], y_vel[j]])
    vel_nodes = jnp.array(vel_nodes)

    def v_idx(i, j):
        return i * nvy + j

    def p_idx(i, j):
        return i * ny + j

    x_pres = jnp.linspace(0, 1, nx)
    y_pres = jnp.linspace(0, 1, ny)

    pres_nodes = []
    for i in range(nx):
        for j in range(ny):
            pres_nodes.append([x_pres[i], y_pres[j]])
    pres_nodes = jnp.array(pres_nodes)

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
                'p_nodes': jnp.array([p0, p1, p3]),
                'v_nodes': jnp.array([v00, v20, v22, v10, v21, v11])
            })
            elements.append({
                'p_nodes': jnp.array([p0, p3, p2]),
                'v_nodes': jnp.array([v00, v22, v02, v11, v12, v01])
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
    N = jnp.array([
        bound * (1 - 2 * xi - 2 * eta),
        xi * (2 * xi - 1),
        eta * (2 * eta - 1),
        4 * xi * bound,
        4 * xi * eta,
        4 * eta * bound
    ])
    #Nodes 0-2 and 3-5 follow standard counter-clockwise orientation

    #\frac{\partial N / \partial \xi}
    partial_xi = jnp.array([-3 + 4 * xi + 4 * eta, 4 * xi - 1, 0, 4 - 8 * xi - 4 * eta, 4 * eta, -4 * eta])

    # \frac{\partial N / \partial \eta}
    partial_eta = jnp.array([-3 + 4 * xi + 4 * eta, 0, 4 * eta - 1, -4 * xi, 4 * xi, 4 - 4 * xi - 8 * eta])

    return N, partial_xi, partial_eta

def assemble_system(mesh, f):
    nv_nodes = mesh['num_vel_nodes']
    np_nodes = mesh['num_pres_nodes']
    dof = 2 * nv_nodes + np_nodes
    K_global = jnp.zeros((dof, dof))
    F_global = jnp.zeros(dof)

    #3-point symmetric Gauss Quadrature for triangles
    quad_points = jnp.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
    quad_weights = jnp.array([1 / 6, 1 / 6, 1 / 6])

    for element in mesh['elements']:
        v_idx = jnp.array(element['v_nodes'])
        p_idx = jnp.array(element['p_nodes'])
        v_coords = mesh['vel_nodes'][v_idx[0:3]]
        x1, x2, x3 = v_coords[0], v_coords[1], v_coords[2]

        # Since (xi, eta) are reference coordinates, we map them to spacial coordinates. Let g be the map (xi, eta) -> (x, y).
        # Compute J = Dg, the Jacobian map
        J = jnp.array([
            [x2[0] - x1[0], x3[0] - x1[0]],
            [x2[1] - x1[1], x3[1] - x1[1]]
        ])
        detJ = jnp.abs(jnp.linalg.det(J))
        invJ = jnp.linalg.inv(J)

        A_k = jnp.zeros((6, 6))
        B = jnp.zeros((3, 6, 2))
        F_local = jnp.zeros((2, 6))

        #Numerical integration
        for q in range(3):
            xi, eta = quad_points[q]
            w = quad_weights[q] * detJ

            # 1. Evaluate P2 Velocity Shapes
            N, partial_xi, partial_eta = quadratic_basis(xi, eta)
            reference_grads = jnp.stack(
                [partial_xi, partial_eta],
                axis=1
            )

            gradN = reference_grads @ invJ.T

            psi = jnp.array([1 - xi - eta, xi, eta])
            pos = (1 - xi - eta)*x1 + xi*x2 + eta*x3
            val = jnp.asarray(f(pos[0], pos[1]))

            F_local += w * jnp.outer(val, N)
            A_k += w * (gradN @ gradN.T)

            B = B + w * (
                    psi[:, None, None] *
                    gradN[None, :, :]
            )

        B_x = B[:, :, 0]
        B_y = B[:, :, 1]
        F_global = F_global.at[v_idx].add(F_local[0])
        F_global = F_global.at[v_idx + nv_nodes].add(F_local[1])

        rows_v = v_idx[:, None]
        cols_v = v_idx[None, :]

        K_global = K_global.at[rows_v, cols_v].add(A_k)
        K_global = K_global.at[
            rows_v + nv_nodes,
            cols_v + nv_nodes
        ].add(A_k)

        p_rows = p_idx[:, None] + 2 * nv_nodes
        v_cols = v_idx[None, :]

        K_global = K_global.at[p_rows, v_cols].add(B_x)
        K_global = K_global.at[p_rows, v_cols + nv_nodes].add(B_y)

        K_global = K_global.at[v_cols.T, p_rows.T].add(B_x.T)
        K_global = K_global.at[
            (v_cols + nv_nodes).T,
            p_rows.T
        ].add(B_y.T)

    return K_global, F_global

if __name__ == "__main__":
    start_time = time.perf_counter()
    f = lambda x, y: [jnp.exp(-50 * ((x - 0.4) ** 2 + (y - 0.6) ** 2)), jnp.exp(-50 * ((x - 0.6) ** 2 + (y - 0.4) ** 2))]
    ux, uy, p = solve(30, 30, f)
    print(ux)
    print(uy)
    print(p)
    loss = lambda amp: jnp.sum(
        solve(
            30,
            30,
            lambda x, y: [
                amp * jnp.exp(-50 * ((x - .4) ** 2 + (y - .6) ** 2)),
                amp * jnp.exp(-50 * ((x - .6) ** 2 + (y - .4) ** 2))
            ]
        )[0]
    )
    end_time = time.perf_counter()
    print(jax.grad(loss)(1.0))
    print(end_time - start_time)