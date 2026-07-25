"""
2D Transient Heat Conduction | FEniCSx / DOLFINx
=================================================
Governing PDE (strong form):
rho*cp * dT/dt = div(k * grad(T)) + Q in Omega = [0,1] x [0,1]
Boundary conditions (all four walls distinct):
LEFT (x=0): Dirichlet T = T_cold = 300 K (cold sink)
RIGHT (x=1): Neumann k*dT/dn = +q_right (heat flux IN)
BOTTOM (y=0): Dirichlet T = T_hot = 500 K (hot base)
TOP (y=1): Neumann k*dT/dn = -q_top (heat flux OUT / convective loss)
Physics expectation:
- Strong gradient bottom (hot) -> top (cooler)
- Left wall anchored hot, right wall receiving flux
- A 2-D "saddle" or skewed temperature field | NOT uniform
"""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io, plot
from dolfinx.fem import functionspace, Function, Constant, dirichletbc
from dolfinx.fem.petsc import LinearProblem
import ufl
from ufl import TestFunction, TrialFunction, dx, ds, inner, grad
import pyvista
# 1. Material & source parameters (steel-like)
k_val = 0.60 # Thermal conductivity [W/m·K]
rho_val = 1000.0 # Density [kg/m³]
cp_val = 4184.0 # Specific heat [J/kg·K]
Q_val = 0.0 # Volumetric source [W/m³]
q_right = 10.0 # Inward flux, right BC [W/m²]
q_top = 0.0 # Outward flux, top BC [W/m²]
T_cold = 300.0 # Right wall and Top wall temp [K]
T_hot = 500.0 # Left Wall and Bottom wall temp [K]
# Thermal diffusivity & timescale
alpha = k_val / (rho_val * cp_val)
tau = 10**2 / alpha # diffusion timescale [s]
t_end = 2.5 * tau
dt = tau / 20
n_steps = int(t_end / dt)
# 2. Mesh (refined 128x128 for smooth 2-D field)
comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 128, 128, mesh.CellType.quadrilateral)
V = functionspace(domain, ("Lagrange", 1))
T = TrialFunction(V)
v = TestFunction(V)
T_n = Function(V) # previous step
T_h = Function(V) # current solution
# Initialise field: linear interpolation between T_cold and T_hot
T_n.interpolate(lambda x: T_cold + (T_hot - T_cold) * x[1] + (((T_hot-T_cold)**2)/2)*(x[1]**2))
T_h.x.array[:] = T_n.x.array
# UFL constants
k = Constant(domain, PETSc.ScalarType(k_val))
rho = Constant(domain, PETSc.ScalarType(rho_val))
cp = Constant(domain, PETSc.ScalarType(cp_val))
Q = Constant(domain, PETSc.ScalarType(Q_val))
qR = Constant(domain, PETSc.ScalarType(q_right))
qT = Constant(domain, PETSc.ScalarType(q_top))
Dt = Constant(domain, PETSc.ScalarType(dt))
# 3. Boundary facet tags
fdim = domain.topology.dim - 1
def mark_facets(domain, fdim):
left_f = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
right_f = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 1.0))
bottom_f = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
top_f = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
facets = np.concatenate([left_f, right_f, bottom_f, top_f])
markers = np.concatenate([
    np.full_like(left_f, 1, dtype=np.int32),
    np.full_like(right_f, 2, dtype=np.int32),
    np.full_like(bottom_f, 3, dtype=np.int32),
    np.full_like(top_f, 4, dtype=np.int32),
])
# sort by facet index (required by DOLFINx)
order = np.argsort(facets)
return mesh.meshtags(domain, fdim, facets[order], markers[order])
facet_tags = mark_facets(domain, fdim)
ds_tagged = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
# 4. Dirichlet BCs -- -- -- -- --
# LEFT T = T_hot
left_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
bc_left = dirichletbc(Constant(domain, PETSc.ScalarType(T_hot)), left_dofs, V)
# BOTTOM T=T_hot
bot_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0.0))
bc_bot = dirichletbc(Constant(domain, PETSc.ScalarType(T_hot)), bot_dofs, V)
# TOP -
top_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 1.0))
bc_top = dirichletbc(Constant(domain, PETSc.ScalarType(T_cold)), top_dofs, V)
bcs = [bc_left, bc_bot, bc_top]
# 5. Variational form (Backward Euler)
#
# rho*cp*(T - T_n)/dt * v dx
# + k * grad(T)·grad(v) dx
# = rho*cp*T_n/dt * v dx
# + Q * v dx
# + q_right * v ds(right) [+: flux INTO domain]
# - q_top * v ds(top) [-: flux OUT of domain]
a = (rho * cp / Dt) * inner(T, v) * dx + k * inner(grad(T), grad(v)) * dx
L = (rho * cp / Dt) * inner(T_n, v) * dx + inner(Q, v) * dx + inner(qR, v) * ds_tagged(2)
- inner(qT, v) * ds_tagged(4)
# 6. Linear solver
problem = LinearProblem(
    a, L,
    bcs=bcs,
    u=T_h,
    petsc_options_prefix="heat_",
    petsc_options={
    "ksp_type": "cg",
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    }
)
# 7. Time-stepping
vtx = io.VTXWriter(comm, "heat2d_solution.bp", [T_h], engine="BP4")
vtx.write(0.0)
t = 0.0
for step in range(n_steps):
    t += dt
    problem.solve()
    T_n.x.array[:] = T_h.x.array
    vtx.write(t)
    arr = T_h.x.array
vtx.close()
# Pyvista visualizaation -
topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
T_array = T_h.x.array.real
grid.point_data["Temperature [K]"] = T_array
plotter = pyvista.Plotter()
plotter.subplot(0, 0)
plotter.add_text("2D Temperature Field (top view)", font_size=10)
flat = grid.copy()
flat.set_active_scalars("Temperature [K]")
plotter.add_mesh(flat, show_edges=False, cmap="inferno",
scalar_bar_args={"title": "T [K]", "vertical": True})
plotter.view_xy()
plotter.add_axes()
plotter.show()