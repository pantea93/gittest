import mshr
import time
from fenics import *
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import scipy.linalg as scl
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from mpi4py import MPI
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import interpolate
from mshr import *
from functions import *

plt.style.use('classic')

File_name = 'skew_mesh_2D_advdiff_1e-3_png'
# Adaptive data
TOL = 1e-30
REFINE_RATIO = 0.5
MAX_ITER = 100
LOCAL_CON = False
REF_TYPE = 1
Nini = 2
M = 500
Degree = 8

# Polynomial dregrees (FEM = Ptrial)

TrialType = "CG"
TestType = "DG"

Ptrial = 1
Ptest = Ptrial

uexact = Expression('1. + tanh(%s*(x[1]-x[0]/3-0.5))' %M,degree = Degree)

# Forcing data
f = Constant(0.)

TrialTypeDir = os.path.join(File_name, 'p%s'%Ptrial)
print(TrialTypeDir)

if not os.path.isdir(TrialTypeDir): os.makedirs(TrialTypeDir)

xmin, xmax = 0, 1
ymin, ymax = 0, 1
mesh = UnitSquareMesh(Nini, Nini)
fig = plt.figure()
plt.axis('off')
plt.grid(b=None)
plot(mesh, linewidth = 0.5)
data_filename = os.path.join(TrialTypeDir, 'mesh_0.png')
fig.savefig(data_filename, format='png', transparent=True)
plt.close()

# data for plots 2D
N = 500
X = np.linspace(0.,1.,N)
Y = np.linspace(0.,1.,N)
X,Y = np.meshgrid(X,Y)

# data for plots 1D
Ndiag = 600
X2 = np.linspace(0.,1.,Ndiag)

LevelStep = np.zeros(MAX_ITER)
L2vect = np.zeros(MAX_ITER)
Dofsvect = np.zeros(MAX_ITER)
Dofsevect = np.zeros(MAX_ITER)
Dofsuvect = np.zeros(MAX_ITER)
udiagvect = np.zeros(MAX_ITER,dtype=np.object)
TimeILU = np.zeros(MAX_ITER)
TimeMINRES = np.zeros(MAX_ITER)
TotalTime = np.zeros(MAX_ITER)

btol = 10*DOLFIN_EPS

# Boundaries definitions
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], xmin)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], xmax)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], ymin)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], ymax)

class all_weak(SubDomain):
    def inside(self,x, on_boundary):
        return on_boundary

# Initialize sub-domain instances
left = Left()
right = Right()
top = Top()
bottom = Bottom()

for level in range(MAX_ITER):
    start_total = time.time()
    level_step = level
    totaliter = 0
    print('Level %s' %level)
    n = FacetNormal(mesh)
    h = Circumradius(mesh)

    x, y = SpatialCoordinate(mesh)

    #Advection and diffusion values
    beta_vec = as_vector([1, pi/2])
    kappa = 1E-3

    boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    boundary_markers.set_all(0)
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

    V1 = FunctionSpace(mesh, TestType, Ptest)
    V2 = FunctionSpace(mesh, TrialType, Ptrial)

    e = TrialFunction(V1)
    u = TrialFunction(V2)

    e1 = TestFunction(V1)
    u1 = TestFunction(V2)

    dofs1 = V1.dofmap().dofs()
    dofs2 = V2.dofmap().dofs()

    Ndofs1 = np.size(dofs1)
    Ndofs2 = np.size(dofs2)
    Ndofs = Ndofs1 + Ndofs2

    print('dof1 = %s, dof2 = %s, dofs = %s' %(Ndofs1, Ndofs2, Ndofs))

    n = FacetNormal(mesh)
    h = Circumradius(mesh)
    hmin = mesh.hmin()
    hmax = mesh.hmax()

    print('h min = %s, h max = %s' %(hmin, hmax))

    #Weak imposition of boundary conditions
    uD = conditional(dot(beta_vec,n) < 0, conditional(y < pi/10, 1., 0.), 0.)

    #BC's in the bilinear form
    if kappa == 0:
        bd = 0.5*inner((abs(dot(beta_vec,n))-dot(beta_vec,n))*e1, u)*ds
        btd = 0.5*inner((abs(dot(beta_vec,n))-dot(beta_vec,n))*e, u1)*ds
        bdrhs = 0.5*inner((abs(dot(beta_vec,n))-dot(beta_vec,n))*e1, uD)*ds
    else:
        bd = 0.5*inner((abs(dot(beta_vec,n))-dot(beta_vec,n))*e1, u)*ds
        bd += -inner(dot(kappa*grad(e1),n), u)*ds + inner(kappa/h*e1, u)*ds
        btd = 0.5*inner((abs(dot(beta_vec,n))-dot(beta_vec,n))*e, u1)*ds
        btd += -inner(dot(kappa*grad(e),n), u1)*ds + inner(kappa/h*e, u1)*ds
        bdrhs = 0.5*inner((abs(dot(beta_vec,n))-dot(beta_vec,n))*e1, uD)*ds
        bdrhs += -inner(dot(kappa*grad(e1),n), uD)*ds + inner(kappa/h*e1, uD)*ds

    if kappa == 0:
        print("You are solving a pure-advection problem")
        g = inner(e1, e)*dx + inner(h*dot(beta_vec, grad(e)), dot(beta_vec, grad(e1)))*dx
        g += 0.5*inner(abs(dot(beta_vec('+'),n('+')))*jump(e), jump(e1))*dS
        g += 0.5*inner(abs(dot(beta_vec,n))*e, e1)*ds

        b = inner(div(beta_vec*u), e1)*dx + bd

        b_t = inner(div(beta_vec*u1), e)*dx + btd

        l = inner(e1, f)*dx + bdrhs

    else:
        print("You are solving an advection-diffusion problem")
        g = inner(e1, e)*dx + inner(h*dot(beta_vec, grad(e)), dot(beta_vec, grad(e1)))*dx
        g += 0.5*inner(abs(dot(beta_vec('+'),n('+')))*jump(e), jump(e1))*dS
        g += 0.5*inner(abs(dot(beta_vec,n))*e, e1)*ds
        g += inner(kappa*grad(e), grad(e1))*dx
        g += inner(kappa/(h('+')+h('-'))*jump(e), jump(e1))*dS

        b = inner(div(beta_vec*u), e1)*dx + bd
        b += -inner(div(kappa*grad(u)), e1)*dx + inner(jump(kappa*grad(u), n), avg(e1))*dS

        b_t = inner(div(beta_vec*u1), e)*dx + btd
        b_t += -inner(div(kappa*grad(u1)), e)*dx + inner(jump(kappa*grad(u1), n), avg(e))*dS

        l = inner(e1, f)*dx + bdrhs

    Ge = assemble(g)
    G_mat = as_backend_type(Ge).mat()
    G = sp.csc_matrix(sp.csr_matrix(G_mat.getValuesCSR()[::-1], shape = G_mat.size))

    Be = assemble(b)
    B1 = Be
    B1_mat = as_backend_type(B1).mat()
    B = sp.csc_matrix(sp.csr_matrix(B1_mat.getValuesCSR()[::-1], shape = B1_mat.size))

    Be_t = assemble(b_t)
    B2 = Be_t
    B2_mat = as_backend_type(B2).mat()
    B_t = sp.csc_matrix(sp.csr_matrix(B2_mat.getValuesCSR()[::-1], shape = B2_mat.size))

    L = assemble(l)
    print('assembly done')

    eps_cg = 1E-10
    tol_ave = 1E-5
    iter_cg = 0
    max_iter_cg = 200
    Minres_max = Ndofs2

    #initial guess
    if level == 0:
        Gtil_inv = spl.inv(sp.csc_matrix(G))
        u_k = spl.spsolve(B_t.dot(Gtil_inv.dot(B)), B_t.dot(Gtil_inv.dot(L)))
        e_k = Gtil_inv.dot(L - B.dot(u_k))
        du = np.zeros(Ndofs2)
        de = np.zeros(Ndofs1)
        res_e = np.zeros(Ndofs1)
        res_u = np.zeros(Ndofs2)
        sume = 0
        sumu = 0
        sume0 = 1
        sumu0 = 1
        ave = 0
        elapsed_time_iLU = 0

    else:

        e_proj = Function(V1)
        u_proj = Function(V2)

        e_k = disc_proj(e_past, mesh, V1, Ndofs1, 1E-12)

        e_proj.vector()[:] = e_k

        u_k = np.zeros(Ndofs2)

        dof_coords_V2 = V2.tabulate_dof_coordinates().reshape(-1, mesh.topology().dim())

        for ni in range(Ndofs2):
            u_k[ni] = u_past(dof_coords_V2[ni])

        u_proj.vector()[:] = u_k
        plt.figure(1)
        plot(e_past)

        plt.figure(2)
        plot(e_proj)

        res_e = L - G.dot(e_k) - B.dot(u_k)
        res_u = -B_t.dot(e_k)

        sume = sum(res_e*res_e)
        sumu = sum(res_u*res_u)

        sum0 = sume + sumu

        sume0 = sume
        sumu0 = sumu

        print('sume0 = %s' %sume0)
        print('sumu0 = %s' %sumu0)

        ave = sum0

        start_ilu = time.time()
        Gtil_inv = spl.spilu(G,drop_tol = 1e-4,fill_factor = 6)
        elapsed_time_iLU = time.time() - start_ilu

        S_lamb = lambda x: Schur(B,B_t,Gtil_inv,x)

        S_op = spl.LinearOperator((Ndofs2,Ndofs2), S_lamb)
        #print('Schur creation done')
        ave = 1

    start_minres = time.time()
    while (iter_cg <= max_iter_cg) and (ave > tol_ave):
        iter_cg += 1

        du,auu = spl.minres(S_op,B_t.dot(Gtil_inv.solve(res_e)) - res_u, tol = 1E-4)
        sumu = sum(res_u*res_u)

        totaliter += 1

        de = Gtil_inv.solve(res_e - B.dot(du))

        u_k += du
        e_k += de

        res_e = L - G.dot(e_k) - B.dot(u_k)
        res_u = -B_t.dot(e_k)

        sumu = sum(res_u*res_u)
        sume = sum(res_e*res_e)

        sumtot = sume + sumu
        ave = sumtot/sum0

    elapsed_time_minres = time.time() - start_minres
    total_elapsed_time = time.time() - start_total

    print('ave = %s' %ave)

    inner_iter = totaliter
    outer_iter = iter_cg
    avg_iter = 1#totaliter/iter_cg

    print('Accumulative iter %s' %totaliter)
    #print(delta_new/delta_0)

    u_disc = Function(V2)
    u_disc.vector()[:] = u_k
    e_disc = Function(V1)
    e_disc.vector()[:] = e_k

    V1past = V1
    V2past = V2
    mesh_past = mesh

    e_past = Function(V1past)
    u_past = Function(V2past)
    e_past = e_disc
    u_past = u_disc
    L2e = errornorm(uexact, u_disc, 'L2')

    # compute error indicators from the DPG mixed variable e
    PC = FunctionSpace(mesh,"DG", 0)
    c  = TestFunction(PC)             # p.w constant fn

    if kappa == 0:
        ge = e_disc*e_disc*c*dx + h*dot(beta_vec, grad(e_disc))*dot(beta_vec, grad(e_disc))*c*dx
        ge += 0.5*abs(dot(beta_vec,n))*e_disc*e_disc*c*ds
        ge += 0.5*abs(dot(beta_vec('+'),n('+')))*jump(e_disc)*jump(e_disc)*(c('+')+c('-'))*dS

    else:
        ge = e_disc*e_disc*c*dx + h*dot(beta_vec, grad(e_disc))*dot(beta_vec, grad(e_disc))*c*dx
        ge += 0.5*abs(dot(beta_vec,n))*e_disc*e_disc*c*ds
        ge += 0.5*abs(dot(beta_vec('+'),n('+')))*jump(e_disc)*jump(e_disc)*(c('+')+c('-'))*dS
        ge += kappa*dot(grad(e_disc), grad(e_disc))*c*dx
        ge += kappa/(h('+')+h('-'))*jump(e_disc)*jump(e_disc)*jump(c)*dS

    g = assemble(ge)

    PC2 = FunctionSpace(mesh,"DG", 0)

    # element-wise norms of e

    if kappa == 0:
        Ee = assemble(e_disc*e_disc*dx + h*dot(beta_vec, grad(e_disc))*dot(beta_vec, grad(e_disc))*dx)
        Ee += 0.5*assemble(abs(dot(beta_vec,n))*e_disc*e_disc*ds)
        Ee += 0.5*assemble(abs(dot(beta_vec('+'),n('+')))*jump(e_disc)*jump(e_disc)*dS)

    else:
        Ee = assemble(e_disc*e_disc*dx + h*dot(beta_vec, grad(e_disc))*dot(beta_vec, grad(e_disc))*dx)
        Ee += 0.5*assemble(abs(dot(beta_vec,n))*e_disc*e_disc*ds)
        Ee += 0.5*assemble(abs(dot(beta_vec('+'),n('+')))*jump(e_disc)*jump(e_disc)*dS)
        Ee += assemble(kappa*dot(grad(e_disc), grad(e_disc))*dx)
        Ee += assemble(kappa/(h('+')+h('-'))*jump(e_disc)*jump(e_disc)*dS)

    E = sqrt(Ee)

    e0r = u_disc - uexact
    e0_res = Function(V2)
    e0_res = e0r
    E1r = L2e
    E2r = 0 #0.5*assemble( (abs(dot(beta_vec,n)))*e0_res*e0_res*ds)
    E3r = 0 #assemble( abs(dot(beta_vec('+'),n('+')))*(e0_res('+')-e0_res('-'))*(e0_res('+')-e0_res('-'))*dS )
    E4r = 0 #assemble( h*div(beta_vec*u_disc)*div(beta_vec*u_disc)*dx )
    Er = 0 #sqrt(E1r**2 + E2r + E3r + E4r)
    Graph = 0 #sqrt(L2e**2 + assemble( div(beta_vec*u_disc)*div(beta_vec*u_disc)*dx ))


    LevelStep[level] = level_step
    L2vect[level] = L2e
    Dofsvect[level] = Ndofs
    Dofsevect[level] = Ndofs1
    Dofsuvect[level] = Ndofs2
    TimeILU[level] = elapsed_time_iLU
    TimeMINRES[level] = elapsed_time_minres
    TotalTime[level] = total_elapsed_time

    uu = np.zeros(X.shape)
    uuex = np.zeros(X.shape)
    udiag = np.zeros(Ndiag)
    ww = np.zeros(X.shape)

    cont = -1
    for ni in range(Ndiag):
        udiag[ni] = u_disc(np.array([X2[ni],1.-X2[ni]]))

    udiagvect[level] = udiag
    for ni in range(X.shape[0]):
        for nj in range(X.shape[1]):
            cont += 1

            uu[ni,nj] = u_disc(np.array([X[ni,nj],Y[ni,nj]]))
            ww[ni,nj] = np.abs(e_disc(np.array([X[ni,nj],Y[ni,nj]])))
            uuex[ni,nj] = uexact(np.array([X[ni,nj],Y[ni,nj]]))


    # Mark cells for refinement
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    if REF_TYPE == 1:
        rg = np.sort(g)
        rg = rg[::-1]
        rgind = np.argsort(g)
        rgind = rgind[::-1]
        scum = 0.
    #print(sum(g)-E**2)
        g0 = REFINE_RATIO**2*E**2
    #for cellm in cells(mesh):
    #     print(cellm)
        Ntot = mesh.num_cells()
        for cellm in cells(mesh):
            if cellm.index() == rgind[0]:
                break

        cell_markers[cellm] = True


        for nj in range(1,Ntot):
            scum += g[rgind[nj]]
            for cellm in cells(mesh):
                if cellm.index() == rgind[nj]:
                    break
            cell_markers[cellm] = scum < g0
            if scum > g0:
                break

    if REF_TYPE == 2:
        g0 = sorted(g, reverse=True)[int(len(g)*REFINE_RATIO)]
    #gutol = 1e-1
        for cellm in cells(mesh):
            cell_markers[cellm] = g[cellm.index()] > g0

    # Refine mesh
    mesh = refine(mesh, cell_markers)

#--------------------------------------------------------------------------------------------------#
# Plotting figures
#--------------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------------------#
# Plotting figures
#--------------------------------------------------------------------------------------------------#

    plt.style.use('classic')

    fig = plt.figure()
    plt.axis('off')
    plt.grid(b=None)
    plot(mesh, linewidth = 0.5)
    data_filename = os.path.join(TrialTypeDir, 'mesh_%s.png'%(level+1))
    fig.savefig(data_filename, format='png', transparent=True)
    plt.close()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #plt.tight_layout()
    surf = ax.plot_surface(X, Y, uu,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.grid(color='white', linestyle='--', linewidth=0.1, alpha=0.8)
    ax.view_init(40, 250)
    plt.axis('off')
    data_filename = os.path.join(TrialTypeDir, 'sol_3D_%s.png'%(level))
    fig.savefig(data_filename, format='png', transparent=True)
    plt.close()


    fig,ax = plt.subplots()
    plt.colorbar(surf,ax=ax, orientation='horizontal')
    surf.set_clim(min(u_disc.vector()), max(u_disc.vector()))
    ax.remove()
    data_filename = os.path.join(TrialTypeDir, 'bar_2D_%s.png'%(level))
    plt.savefig(data_filename, format='png', transparent=True)
    plt.close()

    fig, ax1 = plt.subplots()
    cf = plot(e_disc)
    fig.colorbar(cf,ax=ax1)
    #cf.set_clim(-0.25, 0.25)
    data_filename = os.path.join(TrialTypeDir, 'error_2D_%s.png'%(level))
    fig.savefig(data_filename, format='png', transparent=True)
    plt.close()

    fig, ax1 = plt.subplots()
    cf = plot(u_disc)
    fig.colorbar(cf,ax=ax1)
    #cf.set_clim(0., 2.)
    data_filename = os.path.join(TrialTypeDir, 'sol_2D_%s.png'%(level))
    fig.savefig(data_filename, format='png', transparent=True)
    plt.close()

    if level>2:
        fig = plt.figure()
        plt.loglog(Dofsvect[:level],L2vect[:level])
        data_filename = os.path.join(TrialTypeDir, 'l2err%s.png'%(level))
        fig.savefig(data_filename, format='png', transparent=True)
        plt.close()

        all_data = np.array(8,dtype=np.object)
        all_data = LevelStep, level, Dofsvect, Dofsevect, Dofsuvect, L2vect, TimeILU, TimeMINRES, TotalTime
        data_filename = os.path.join(TrialTypeDir, 'data')
        np.save(data_filename, all_data)

    # Check convergence
    if E < TOL:
        print("Success, solution converged after %d iterations" %level)
        #break
