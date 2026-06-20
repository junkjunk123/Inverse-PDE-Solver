import time
import jax
import jax.numpy as jnp

def local_potential_energy(c, u, B, det, quad_weights):
    grad = jnp.dot(B, u)
    energy_density = 0.5 * (c ** 2) * jnp.sum(grad ** 2, axis=-1)
    return jnp.sum(energy_density * det * quad_weights)


def local_kinetic_energy(rho, v, N, det, quad_weights):
    v_quad = jnp.dot(N, v)
    energy_density = 0.5 * rho * (v_quad ** 2)
    return jnp.sum(energy_density * det * quad_weights)


potential = jax.vmap(local_potential_energy, in_axes=(0, 0, 0, None))
kinetic = jax.vmap(local_kinetic_energy, in_axes=(0, 0, 0, None))


def global_potential_energy(c, U, gathering_matrix, B, detJ, weights):
    u = U[gathering_matrix]
    element_energies = potential(c, u, B, detJ, weights)
    return jnp.sum(element_energies)


def global_kinetic_energy(rho, V, gathering_matrix, N, detJ, weights):
    v = V[gathering_matrix]
    element_energies = kinetic(rho, v, N, detJ, weights)
    return jnp.sum(element_energies)

internal_forces = jax.grad(global_potential_energy, argnums=1)

def step(U, V, dt, M_diag, int_forces):
    A = -int_forces / M_diag
    V_new = V + dt * A
    U_new = U + dt * V_new
    return U_new, V_new


def solve(U0, V0, dt, M_diag, steps, f):
    def step_fn(carry, _):
        U, V = carry
        F_int = f(U)
        U_new, V_new = step(U, V, dt, M_diag, F_int)
        return (U_new, V_new), U_new

    (_, _), U_hist = jax.lax.scan(step_fn,(U0, V0),None, length=steps)

    return U_hist