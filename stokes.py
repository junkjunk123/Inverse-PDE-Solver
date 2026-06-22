#This test adds Jax-based autodifferentiation capabilities to the Stokes flow solver from femtest4.py. The code works functionally and autodifferentiates in ~11 seconds.

import time
import jax
import jax.numpy as jnp
from jaxopt._src.linear_solve import solve_gmres

def solve_jitted(nx, ny, theta):
    def f(x, y):
        return jnp.array([
            jnp.exp(-50.0 * ((x - theta) ** 2 + (y - 1 + theta) ** 2)),
            jnp.exp(-50.0 * ((x - 1 + theta) ** 2 + (y - theta) ** 2))
        ])

    mesh_data = taylor_hood_mesh(nx, ny)
    nv = mesh_data['num_vel_nodes']
    nodes = mesh_data['num_pres_nodes']
    total_dof = 2 * nv + nodes

    F_global, A_k_all, B_x_all, B_y_all = element_matrices(mesh_data, f)
    mask = boundary_conditions(mesh_data['vel_nodes'], nv, total_dof)

    # 3. Matrix-Free Operator
    def stokes_operator(U):
        ux = U[0:nv]
        uy = U[nv:2 * nv]
        p = U[2 * nv:]

        # Gather
        ux_local = ux[mesh_data['v_idx']]
        uy_local = uy[mesh_data['v_idx']]
        p_local = p[mesh_data['p_idx']]

        # Vectorized Math via Einsum
        local_Ax = jnp.einsum('eij,ej->ei', A_k_all, ux_local)
        local_Ay = jnp.einsum('eij,ej->ei', A_k_all, uy_local)
        local_Bx = jnp.einsum('eij,ej->ei', B_x_all, ux_local)
        local_By = jnp.einsum('eij,ej->ei', B_y_all, uy_local)
        local_Gpx = jnp.einsum('eji,ej->ei', B_x_all, p_local)
        local_Gpy = jnp.einsum('eji,ej->ei', B_y_all, p_local)

        # Scatter
        Ax_u = jnp.zeros(nv).at[mesh_data['v_idx']].add(local_Ax)
        Ay_u = jnp.zeros(nv).at[mesh_data['v_idx']].add(local_Ay)
        Bx_u = jnp.zeros(nodes).at[mesh_data['p_idx']].add(local_Bx)
        By_u = jnp.zeros(nodes).at[mesh_data['p_idx']].add(local_By)
        GradPx = jnp.zeros(nv).at[mesh_data['v_idx']].add(local_Gpx)
        GradPy = jnp.zeros(nv).at[mesh_data['v_idx']].add(local_Gpy)

        KU_raw = jnp.concatenate([Ax_u + GradPx, Ay_u + GradPy, Bx_u + By_u])
        return (1.0 - mask) * KU_raw + mask * U

    F_bc = F_global * (1.0 - mask)

    U = solve_gmres(stokes_operator, F_bc, maxiter=500)

    return U[0:nv], U[nv:2 * nv], U[2 * nv:]

solve_jitted = jax.jit(solve_jitted, static_argnums=(0, 1))

def taylor_hood_mesh(nx, ny):
    nvx = 2 * nx - 1
    nvy = 2 * ny - 1

    v_x, v_y = jnp.meshgrid(jnp.linspace(0, 1, nvx), jnp.linspace(0, 1, nvy), indexing='ij')
    vel_nodes = jnp.stack([v_x.ravel(), v_y.ravel()], axis=1)

    p_x, p_y = jnp.meshgrid(jnp.linspace(0, 1, nx), jnp.linspace(0, 1, ny), indexing='ij')
    pres_nodes = jnp.stack([p_x.ravel(), p_y.ravel()], axis=1)

    i_idx, j_idx = jnp.meshgrid(jnp.arange(nx - 1), jnp.arange(ny - 1), indexing='ij')
    i_flat, j_flat = i_idx.ravel(), j_idx.ravel()

    # Pressure indexing mapping vectors
    p0 = i_flat * ny + j_flat
    p1 = (i_flat + 1) * ny + j_flat
    p2 = i_flat * ny + (j_flat + 1)
    p3 = (i_flat + 1) * ny + (j_flat + 1)

    # Velocity index mapping vectors
    v00 = (2 * i_flat) * nvy + (2 * j_flat)
    v20 = (2 * i_flat + 2) * nvy + (2 * j_flat)
    v02 = (2 * i_flat) * nvy + (2 * j_flat + 2)
    v22 = (2 * i_flat + 2) * nvy + (2 * j_flat + 2)
    v10 = (2 * i_flat + 1) * nvy + (2 * j_flat)
    v01 = (2 * i_flat) * nvy + (2 * j_flat + 1)
    v21 = (2 * i_flat + 2) * nvy + (2 * j_flat + 1)
    v12 = (2 * i_flat + 1) * nvy + (2 * j_flat + 2)
    v11 = (2 * i_flat + 1) * nvy + (2 * j_flat + 1)

    p_table1 = jnp.stack([p0, p1, p3], axis=1)
    v_table1 = jnp.stack([v00, v20, v22, v10, v21, v11], axis=1)

    p_table2 = jnp.stack([p0, p3, p2], axis=1)
    v_table2 = jnp.stack([v00, v22, v02, v11, v12, v01], axis=1)

    return {
        'vel_nodes': vel_nodes, 'pres_nodes': pres_nodes,
        'p_idx': jnp.concatenate([p_table1, p_table2], axis=0),
        'v_idx': jnp.concatenate([v_table1, v_table2], axis=0),
        'num_vel_nodes': len(vel_nodes), 'num_pres_nodes': len(pres_nodes)
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


def boundary_conditions(nodes, nv, total_dof):
    left = jnp.abs(nodes[:, 0]) < 1e-7
    right = jnp.abs(nodes[:, 0] - 1.0) < 1e-7
    bottom = jnp.abs(nodes[:, 1]) < 1e-7
    top = jnp.abs(nodes[:, 1] - 1.0) < 1e-7

    boundary = left | right | bottom | top

    mask = jnp.zeros(total_dof)
    mask = mask.at[0:nv].set(jnp.where(boundary, 1.0, 0.0))
    mask = mask.at[nv:2 * nv].set(jnp.where(boundary, 1.0, 0.0))
    mask = mask.at[2 * nv].set(1.0) #pin
    return mask


def element_matrices(mesh, f):
    v_idx = mesh['v_idx']
    v_coords = mesh['vel_nodes'][v_idx[:, 0:3]]

    def compute_single_element(v_coords):
        x1, x2, x3 = v_coords[0], v_coords[1], v_coords[2]
        J = jnp.array([[x2[0] - x1[0], x3[0] - x1[0]], [x2[1] - x1[1], x3[1] - x1[1]]])
        detJ = jnp.abs(jnp.linalg.det(J))
        invJ = jnp.linalg.inv(J)

        points = jnp.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
        weights = jnp.array([1 / 6, 1 / 6, 1 / 6])

        A_k = jnp.zeros((6, 6))
        B = jnp.zeros((3, 6, 2))
        F_local = jnp.zeros((2, 6))

        for q in range(3):
            xi, eta = points[q]
            w = weights[q] * detJ

            N, partial_xi, partial_eta = quadratic_basis(xi, eta)
            gradN = jnp.stack([partial_xi, partial_eta], axis=1) @ invJ.T

            psi = jnp.array([1 - xi - eta, xi, eta])
            pos = (1 - xi - eta) * x1 + xi * x2 + eta * x3
            val = f(pos[0], pos[1])

            F_local += w * jnp.outer(val, N)
            A_k += w * (gradN @ gradN.T)
            B += w * (psi[:, None, None] * gradN[None, :, :])

        return F_local, A_k, B[:, :, 0], B[:, :, 1]
    F_local_all, A_k_all, B_x_all, B_y_all = jax.vmap(compute_single_element)(v_coords)

    nv_nodes = mesh['num_vel_nodes']
    dof = 2 * nv_nodes + mesh['num_pres_nodes']
    F_global = jnp.zeros(dof)
    F_global = F_global.at[v_idx].add(F_local_all[:, 0, :])
    F_global = F_global.at[v_idx + nv_nodes].add(F_local_all[:, 1, :])

    return F_global, A_k_all, B_x_all, B_y_all

#gemini-generated test
if __name__ == "__main__":
    # Specify the parameters as static scalars
    NX, NY = 30, 30

    t0 = time.perf_counter()
    ux, uy, p = solve_jitted(NX, NY, 0.4)
    print(f"First run (with JIT compilation overhead): {time.perf_counter() - t0:.4f} seconds")
    print("Max X-Velocity:", jnp.max(jnp.abs(ux)))
    print("Max Y-Velocity:", jnp.max(jnp.abs(uy)))
    print("Max Pressure:", jnp.max(p))

    t0 = time.perf_counter()
    ux, uy, p = solve_jitted(NX, NY, 0.6)
    print(f"Second run (Pure cached execution): {time.perf_counter() - t0:.4f} seconds")


    # Define the jitted loss function path for differentiation
    def loss_function(amp):
        u_x, _, _ = solve_jitted(NX, NY, amp)
        return jnp.sum(u_x)


    print("Evaluating gradient via backward autodiff...")
    t0 = time.perf_counter()
    gradient_val = jax.grad(loss_function)(0.5)
    print(f"Gradient calculated: {gradient_val}")
    print(f"Autodiff time: {time.perf_counter() - t0:.4f} seconds")