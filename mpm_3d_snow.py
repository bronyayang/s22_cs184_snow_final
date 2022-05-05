export_file = ''  # use '/tmp/mpm3d.ply' for exporting result to disk

import numpy as np

import taichi as ti

ti.init(arch=ti.gpu)

#dim, n_grid, steps, dt = 2, 128, 20, 2e-4
#dim, n_grid, steps, dt = 2, 256, 32, 1e-4
# dim, n_grid, steps, dt = 3, 32, 25, 4e-4
dim, n_grid, steps, dt = 3, 64, 25, 2e-4
# #dim, n_grid, steps, dt = 3, 128, 25, 8e-5

# n_particles = n_grid**dim // 2**(dim - 1)
# dx = 1 / n_grid

# p_rho = 1
# p_vol = (dx * 0.5)**2
# p_mass = p_vol * p_rho
# gravity = 9.8
# bound = 3
# E = 400

# x = ti.Vector.field(dim, float, n_particles)
# v = ti.Vector.field(dim, float, n_particles)
# C = ti.Matrix.field(dim, dim, float, n_particles)
# J = ti.field(float, n_particles)

# grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
# grid_m = ti.field(float, (n_grid, ) * dim)

# neighbour = (3, ) * dim


# @ti.kernel
# def substep():
#     for I in ti.grouped(grid_m):
#         grid_v[I] = ti.zero(grid_v[I])
#         grid_m[I] = 0
#     ti.block_dim(n_grid)
#     for p in x:
#         Xp = x[p] / dx
#         base = int(Xp - 0.5)
#         fx = Xp - base
#         w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
#         stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
#         affine = ti.Matrix.identity(float, dim) * stress + p_mass * C[p]
#         for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
#             dpos = (offset - fx) * dx
#             weight = 1.0
#             for i in ti.static(range(dim)):
#                 weight *= w[offset[i]][i]
#             grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
#             grid_m[base + offset] += weight * p_mass
#     for I in ti.grouped(grid_m):
#         if grid_m[I] > 0:
#             grid_v[I] /= grid_m[I]
#         grid_v[I][1] -= dt * gravity
#         cond = (I < bound) & (grid_v[I] < 0) | \
#                (I > n_grid - bound) & (grid_v[I] > 0)
#         grid_v[I] = 0 if cond else grid_v[I]
#     ti.block_dim(n_grid)
#     for p in x:
#         Xp = x[p] / dx
#         base = int(Xp - 0.5)
#         fx = Xp - base
#         w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
#         new_v = ti.zero(v[p])
#         new_C = ti.zero(C[p])
#         for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
#             dpos = (offset - fx) * dx
#             weight = 1.0
#             for i in ti.static(range(dim)):
#                 weight *= w[offset[i]][i]
#             g_v = grid_v[base + offset]
#             new_v += weight * g_v
#             new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
#         v[p] = new_v
#         x[p] += dt * v[p]
#         J[p] *= 1 + dt * new_C.trace()
#         C[p] = new_C

quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 9000 * quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters

x = ti.Vector.field(3, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(3, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(3, 3, dtype=float,
                    shape=n_particles)  # affine velocity field
F = ti.Matrix.field(3, 3, dtype=float,
                    shape=n_particles)  # deformation gradient
material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
grid_v = ti.Vector.field(3, dtype=float,
                         shape=(n_grid, n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid))  # grid node mass
gravity = ti.Vector.field(3, dtype=float, shape=())
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(3, dtype=float, shape=())

@ti.kernel
def substep():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0
    for p in x:  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        F[p] = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]  # deformation gradient update
        h = max(0.1, min(5, ti.exp(10 * (1.0 - Jp[p]))))  # Hardening coefficient: snow gets harder when compressed
        mu, la = mu_0 * h, lambda_0 * h # Lame parameters being functions of the plastic deformation gradients
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                                1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
            F[p] = U @ sig @ V.transpose(
            )  # Reconstruct elastic deformation gradient after plasticity
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose(
        ) + ti.Matrix.identity(float, 3) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j, k in ti.static(ti.ndrange(
                3, 3, 3)):  # Loop over 3x3x3 grid node neighborhood
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:  # No need for epsilon here
            grid_v[i,
                   j, k] = (1 / grid_m[i, j, k]) * grid_v[i,
                                                    j, k]  # Momentum to velocity
            grid_v[i, j, k] += dt * gravity[None] * 30  # gravity
            dist = attractor_pos[None] - dx * ti.Vector([i, j, k])
            grid_v[i, j, k] += dist / (
                0.01 + dist.norm()) * attractor_strength[None] * dt * 100
            if i < 3 and grid_v[i, j, k][0] < 0:
                grid_v[i, j, k][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j, k][0] > 0: grid_v[i, j, k][0] = 0
            if j < 3 and grid_v[i, j, k][1] < 0: grid_v[i, j, k][1] = 0
            if j > n_grid - 3 and grid_v[i, j, k][1] > 0: grid_v[i, j, k][1] = 0
            if k < 3 and grid_v[i, j, k][2] < 0: grid_v[i, j, k][2] = 0
            if k > n_grid - 3 and grid_v[i, j, k][2] > 0: grid_v[i, j, k][2] = 0
    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)
        for i, j, k in ti.static(ti.ndrange(
                3, 3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j, k]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j, k])]
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection


# @ti.kernel
# def reset():
#     group_size = n_particles // 3
#     for i in range(n_particles):
#         x[i] = [
#             ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size),
#             ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size),
#             ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size),
#         ]
#         v[i] = [0, 0, 0] # particle velocity
#         F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # deformation gradient
#         Jp[i] = 1 # det of F
#         C[i] = ti.Matrix.zero(float, 3, 3) # Delta v

@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = ti.Vector([ti.random() for i in range(dim)]) * 0.4 + 0.15
        v[i] = [0, 0, 0] # particle velocity
        F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # deformation gradient
        Jp[i] = 1 # det of F
        C[i] = ti.Matrix.zero(float, 3, 3) # Delta v


def T(a):
    if dim == 2:
        return a

    phi, theta = np.radians(28), np.radians(32)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    c, s = np.cos(phi), np.sin(phi)
    C, S = np.cos(theta), np.sin(theta)
    x, z = x * c + z * s, z * c - x * s
    u, v = x, y * C + z * S
    return np.array([u, v]).swapaxes(0, 1) + 0.5


init()
gravity[None] = [0, -10, 0]
gui = ti.GUI('MPM3D', background_color=0x112F41)
while gui.running and not gui.get_event(gui.ESCAPE):
    for s in range(steps):
        substep()
    pos = x.to_numpy()
    # if export_file:
    #     writer = ti.tools.PLYWriter(num_vertices=n_particles)
    #     writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
    #     writer.export_frame(gui.frame, export_file)
    gui.circles(T(pos), radius=1.5, color=0x66ccff)
    gui.show()