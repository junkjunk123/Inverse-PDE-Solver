#This test applies Jax-based autodifferentiation capabilities to the Wave Equation. Since the equation is time-dependent, the FEM technique gives us an ordinary differential equation in U with respect to time, whose solutions can be approximated to find U. Furthermore, we use jax gradients on energy functions instead of explicitly building matrix operators for easier development. The code works functionally and autodifferentiates in ~0.6 seconds.

import time
import jax
import jax.numpy as jnp

g = 1.0 / jnp.sqrt(3.0)
N_quad_pts = 0.25 * jnp.array([
    [(1 - g) * (1 - g), (1 + g) * (1 - g), (1 + g) * (1 + g), (1 - g) * (1 + g)],
    [(1 + g) * (1 - g), (1 - g) * (1 - g), (1 - g) * (1 + g), (1 + g) * (1 + g)],
    [(1 + g) * (1 + g), (1 - g) * (1 + g), (1 - g) * (1 - g), (1 + g) * (1 - g)],
    [(1 - g) * (1 + g), (1 + g) * (1 + g), (1 + g) * (1 - g), (1 - g) * (1 - g)]
])

def local_potential_energy(c, u, B, det, quad_weights):
    grad = jax.vmap(jnp.dot)(B, jnp.tile(u, (4, 1)))
    energy_density = 0.5 * (c ** 2) * jnp.sum(grad ** 2, axis=-1)
    return jnp.sum(energy_density * det * quad_weights)

def local_kinetic_energy(rho, v, N, det, quad_weights):
    v_quad = jnp.dot(N, v)
    energy_density = 0.5 * rho * (v_quad ** 2)
    return jnp.sum(energy_density * det * quad_weights)

potential = jax.vmap(local_potential_energy, in_axes=(None, 0, 0, 0, None))
kinetic = jax.vmap(local_kinetic_energy, in_axes=(None, 0, None, 0, None))

def global_potential_energy(c, U, gathering_matrix, B, detJ, weights):
    u = U[gathering_matrix]
    element_energies = potential(c, u, B, detJ, weights)
    return jnp.sum(element_energies)

def global_kinetic_energy(rho, V, gathering_matrix, N, detJ, weights):
    v = V[gathering_matrix]
    element_energies = kinetic(rho, v, N, detJ, weights)
    return jnp.sum(element_energies)

def process_single_element(x, partial_xi):
    def compute_at_quad_point(partial_xi_q):
        J = jnp.dot(partial_xi_q, x)
        detJ = jnp.linalg.det(J)
        J_inv = jnp.linalg.inv(J)
        B_local = jnp.dot(J_inv, partial_xi_q)
        return B_local, detJ

    B_mats, detJs = jax.vmap(compute_at_quad_point)(partial_xi)
    return B_mats, detJs

vmap_mesh_geometry = jax.vmap(process_single_element, in_axes=(0, None))

def step(U, V, dt, M_diag, int_forces):
    A = -int_forces / M_diag
    V_new = V + dt * A
    U_new = U + dt * V_new
    return U_new, V_new

def initial_conditions(node_coordinates):
    X = node_coordinates[:, 0]
    Y = node_coordinates[:, 1]
    U0 = jnp.exp(-100.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
    V0 = jnp.zeros_like(U0)
    return U0, V0

def solve(c, rho, size, dt, steps):
    x_coord = jnp.linspace(0, 1, size)
    y_coord = jnp.linspace(0, 1, size)
    X, Y = jnp.meshgrid(x_coord, y_coord)
    node_coordinates = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    i, j = jnp.meshgrid(jnp.arange(size - 1), jnp.arange(size - 1))
    n0 = j * size + i
    n1 = n0 + 1
    n2 = n0 + size + 1
    n3 = n0 + size
    gathering_matrix = jnp.stack([n0.ravel(), n1.ravel(), n2.ravel(), n3.ravel()], axis=1)

    partial_xi = jnp.array([
        [[-0.25, 0.25, 0.25, -0.25], [-0.25, -0.25, 0.25, 0.25]],
        [[-0.25, 0.25, 0.25, -0.25], [-0.25, -0.25, 0.25, 0.25]],
        [[-0.25, 0.25, 0.25, -0.25], [-0.25, -0.25, 0.25, 0.25]],
        [[-0.25, 0.25, 0.25, -0.25], [-0.25, -0.25, 0.25, 0.25]]
    ])
    coords = node_coordinates[gathering_matrix]
    B, detJ = vmap_mesh_geometry(coords, partial_xi)
    quad_weights = jnp.array([1.0, 1.0, 1.0, 1.0])

    U0, V0 = initial_conditions(node_coordinates)

    mass_functional = lambda V_vec: global_kinetic_energy(rho, V_vec, gathering_matrix, N_quad_pts, detJ, quad_weights)
    M_diag = jax.grad(mass_functional)(jnp.ones_like(U0))

    potential_functional = lambda U_vec: global_potential_energy(c, U_vec, gathering_matrix, B, detJ, quad_weights)
    f = jax.grad(potential_functional)

    def step_fn(carry, _):
        U, V = carry
        U_new, V_new = step(U, V, dt, M_diag, f(U))
        return (U_new, V_new), U_new

    (_, _), U_hist = jax.lax.scan(step_fn, (U0, V0), None, length=steps)
    return U_hist

solve_jitted = jax.jit(solve, static_argnums=(2,3,4))

#gemini-generated test
if __name__ == "__main__":
    # Specify the parameters as static scalars
    # For a square mesh, NX equals your grid size variable
    SIZE = 30
    RHO = 1.0
    DT = 0.005
    STEPS = 40

    t0 = time.perf_counter()
    # Evaluating with wave speed c = 0.4
    history = solve_jitted(0.4, RHO, SIZE, DT, STEPS)
    history.block_until_ready()
    print(f"First run (with JIT compilation overhead): {time.perf_counter() - t0:.4f} seconds")
    print("Max Displacement overall:", jnp.max(jnp.abs(history)))
    print("Final State Max Displacement:", jnp.max(jnp.abs(history[-1])))

    t0 = time.perf_counter()
    # Evaluating with wave speed c = 0.6 (uses cached graph since NX is identical)
    history = solve_jitted(0.6, RHO, SIZE, DT, STEPS)
    history.block_until_ready()
    print(f"Second run (Pure cached execution): {time.perf_counter() - t0:.4f} seconds")

    # Define the jitted loss function path for differentiation with respect to wave speed 'c'
    def loss_function(amp):
        history_differentiated = solve_jitted(amp, RHO, SIZE, DT, STEPS)
        return jnp.sum(history_differentiated[-1])

    print("Evaluating gradient via backward autodiff...")
    t0 = time.perf_counter()
    gradient_val = jax.grad(loss_function)(0.5)
    print(f"Gradient calculated: {gradient_val}")
    print(f"Autodiff time: {time.perf_counter() - t0:.4f} seconds")