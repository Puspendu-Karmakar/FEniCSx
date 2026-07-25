from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import pyvista

from dolfinx.fem import (
    Constant,
    Function,
    extract_function_spaces,
    functionspace,
    dirichletbc,
    form,
    locate_dofs_geometrical,
)
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    create_vector,
    set_bc,
)
from dolfinx.mesh import create_unit_square
from dolfinx.plot import vtk_mesh
from basix.ufl import element
from ufl import (
    FacetNormal,
    Identity,
    TestFunction,
    TrialFunction,
    div,
    dot,
    dx,
    inner,
    lhs,
    nabla_grad,
    rhs,
    sym,
)
from dolfinx import mesh as mesh

# Define the time parameters
t = 0.0
T = 10.0
num_steps = 500
dt = T / num_steps

# Mesh (unit square, 10x10 quadrilaterals)
mesh = create_unit_square(MPI.COMM_WORLD, 50, 50, mesh.CellType.quadrilateral)

# Create function spaces
v_cg2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
s_cg1 = element("Lagrange", mesh.basix_cell(), 1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)

# Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# --- Corrected Boundary Conditions ---

# 1. No-slip boundaries: Left, Right, and Bottom walls
def walls(x):
    return np.logical_or(np.isclose(x[1], 0.0),  # Bottom
           np.logical_or(np.isclose(x[0], 0.0),  # Left
                         np.isclose(x[0], 1.0))) # Right

bndry_walls = locate_dofs_geometrical(V, walls)
bc_walls = dirichletbc(np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType), bndry_walls, V)

# 2. Moving Lid: Top wall (excluding the exact corners to prevent singularities)
def lid(x):
    return np.logical_and(np.isclose(x[1], 1.0),
           np.logical_and(x[0] > 1e-8, x[0] < 1.0 - 1e-8)) 

U_val = 1.0
u_top = np.array((U_val, 0.0), dtype=PETSc.ScalarType)
bndry_lid = locate_dofs_geometrical(V, lid)
bc_lid = dirichletbc(u_top, bndry_lid, V)

# Group velocity BCs
bcu = [bc_walls, bc_lid]

# 3. Strain-rate and Stress tensors
def epsilon(u):
    return sym(nabla_grad(u))

def sigma(u, p):
    return 2 * mu * epsilon(u) - p * Identity(mesh.geometry.dim)

# 4. Pin pressure at the absolute bottom-left corner to establish a reference point
def bottom_left(x):
    return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))

bndry_p0 = locate_dofs_geometrical(Q, bottom_left)
bc_p0 = dirichletbc(PETSc.ScalarType(0.0), bndry_p0, Q)
bcp = [bc_p0]
# --- Time-dependent constants and previous solutions ---
u_n = Function(V)
u_n.name = "u_n"
p_n = Function(Q)
p_n.name = "p_n"

k = Constant(mesh, PETSc.ScalarType(dt))
mu = Constant(mesh, PETSc.ScalarType(1.0))
rho = Constant(mesh, PETSc.ScalarType(1.0))
f = Constant(mesh, PETSc.ScalarType((0.0, 0.0)))
n = FacetNormal(mesh)

U = 0.5 * (u_n + u)

# --- Step 1: Tentative velocity ---

F1 = rho * dot((u - u_n) / k, v) * dx
F1 += rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx
F1 += inner(sigma(U, p_n), epsilon(v)) * dx
F1 -= dot(f, v) * dx

a1 = form(lhs(F1))
L1 = form(rhs(F1))

A1 = assemble_matrix(a1, bcs=bcu)
A1.assemble()
b1 = create_vector(extract_function_spaces(L1))

# --- Step 2: Pressure correction ---
u_ = Function(V)
u_.name = "u_"

a2 = form(dot(nabla_grad(p), nabla_grad(q)) * dx)
L2 = form(dot(nabla_grad(p_n), nabla_grad(q)) * dx - (rho / k) * div(u_) * q * dx)

A2 = assemble_matrix(a2, bcs=bcp)
A2.assemble()
b2 = create_vector(extract_function_spaces(L2))

# --- Step 3: Velocity correction ---
p_ = Function(Q)
p_.name = "p_"

a3 = form(rho * dot(u, v) * dx)
L3 = form(rho * dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx)

A3 = assemble_matrix(a3,bcs=bcu)
A3.assemble()
b3 = create_vector(extract_function_spaces(L3))

# --- Solvers ---
solver1 = PETSc.KSP().create(mesh.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.HYPRE)
pc1.setHYPREType("boomeramg")

solver2 = PETSc.KSP().create(mesh.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.CG)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)        # Upgraded to HYPRE
pc2.setHYPREType("boomeramg")

solver3 = PETSc.KSP().create(mesh.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)

# --- Time loop ---
for i in range(num_steps):
    t += dt

    # Step 1: Tentative velocity
    with b1.localForm() as loc:
        loc.set(0.0)
    assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [bcu])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    solver1.solve(b1, u_.x.petsc_vec)
    u_.x.scatter_forward()

    # Step 2: Pressure correction
    with b2.localForm() as loc:
        loc.set(0.0)
    assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [bcp])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    solver2.solve(b2, p_.x.petsc_vec)
    p_.x.scatter_forward()

    # Step 3: Velocity correction
    with b3.localForm() as loc:
        loc.set(0.0)
    assemble_vector(b3, L3)
    apply_lifting(b3, [a3], [bcu])
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b3, bcu)
    solver3.solve(b3, u_n.x.petsc_vec)
    u_n.x.scatter_forward()

    # Update intermediate step values securely
    p_n.x.array[:] = p_.x.array[:]

# --- Cleanup ---
b1.destroy()
b2.destroy()
b3.destroy()
solver1.destroy()
solver2.destroy()
solver3.destroy()


# --- Visualization Fix: Clean Parallel Node Mapping ---
u_n.x.scatter_forward()

# 1. Get the local block/node sizes
local_nodes = V.dofmap.index_map.size_local
bs = V.dofmap.index_map_bs  

# FEniCSx returns unique coordinates per node block.
dof_coords = V.tabulate_dof_coordinates()
local_dof_coords = dof_coords[:local_nodes]  

# 2. Extract the local solution values and reshape them to match the nodes
u_local_flat = u_n.x.array[:local_nodes * bs].real
u_local = u_local_flat.reshape((local_nodes, bs))

u_magnitude = np.sqrt(u_local[:, 0]**2 + u_local[:, 1]**2)

# Pad to 3D so PyVista can orient the glyph arrows properly
values = np.zeros((local_nodes, 3), dtype=np.float64)
values[:, :bs] = u_local

# 3. GATHER all pieces from all processors onto Rank 0
comm = mesh.comm  # <-- Defined right here so it's accessible below!
global_coords = comm.gather(local_dof_coords, root=0)
global_vectors = comm.gather(values, root=0)
global_mags = comm.gather(u_magnitude, root=0)

# 4. Only Rank 0 builds the full plot and displays it
if comm.rank == 0:
    all_coords = np.vstack(global_coords)
    all_vectors = np.vstack(global_vectors)
    all_mags = np.concatenate(global_mags)
    
    # --- Filter out the boundaries entirely ---
    x_coords = all_coords[:, 0]
    y_coords = all_coords[:, 1]
    
    # Exclude points exactly at x=0, x=1, y=0, y=1 (with a tiny tolerance)
    tol = 1e-5
    interior_mask = (
        (x_coords > tol) & (x_coords < 1.0 - tol) & 
        (y_coords > tol) & (y_coords < 1.0 - tol)
    )
    
    # Slice the arrays to keep only the interior data
    interior_coords = all_coords[interior_mask]
    interior_vectors = all_vectors[interior_mask]
    interior_mags = all_mags[interior_mask]
    # -----------------------------------------------

    # Create the point cloud grid using ONLY the interior points
    global_grid = pyvista.PolyData(interior_coords)
    global_grid.point_data["u_direction"] = interior_vectors
    global_grid.point_data["Velocity Magnitude"] = interior_mags

    # Generate glyphs from the filtered dataset
    glyphs = global_grid.glyph(
        orient="u_direction", 
        scale=False,  
        factor=0.03                  
    ) 

    # Initialize the single plotter window
    plotter = pyvista.Plotter()
    
    # Draw a perfect box outline based on the original limits so the frame stays intact
    full_box = pyvista.PolyData(all_coords)
    plotter.add_mesh(full_box.outline(), style="wireframe", color="k")
    
    # Add the pristine interior glyphs
    plotter.add_mesh(glyphs, scalars="Velocity Magnitude", cmap="viridis", show_scalar_bar=True)
    plotter.view_xy()

    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot("couette_uniform_glyphs.png")