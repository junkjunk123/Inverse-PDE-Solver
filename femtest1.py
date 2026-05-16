#This test uses FEM-based techniques to solve the Poisson equation \Delta u = f
import numpy
import numpy as np

def mesh(nx, ny):
    x = numpy.linspace(0, 1, nx)
    y = numpy.linspace(0, 1, ny)

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
    return numpy.dot(grad_i, grad_j) * a