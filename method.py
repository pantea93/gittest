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
from method import *
from sksparse.cholmod import cholesky
def solver_method(mesh,Ptrial,Ptest,TrialType,TestType,f,uD,beta_vec,kappa,uexact,bh,gh,lh,MAX_ITER,level):
	#parameters for iterative solver
	eps_cg = 1E-22
	tol_ave = 1E-6
	Tol_schur = 1e-5
	Tol_gram = 1e-4
	max_iter_cg = 50

	global h,n
	n = FacetNormal(mesh)
	h = Circumradius(mesh)
	# Preconditioner type: 'CHOLESKY', 'ILU'
	Ptype = 'CHOLESKY'
	# Iterative solver for Gram matrix: 'NONE', 'CG'
	Gtype = 'NONE'
	# Global variables only for iLU
	global ilu_drop_tol, ilu_fill_factor
	# drop tol, default 1E-4
	ilu_drop_tol = 1E-4
	# fill factor, default 10
	ilu_fill_factor = 10
	def Preconditioner(G,NdofsV, Type):
		global ilu_drop_tol, ilu_fill_factor
		if Type == 'CHOLESKY':
			start_time = time.time()
			P = cholesky(G,mode="auto")
			P_op = spl.LinearOperator((NdofsV,NdofsV), lambda x: P.solve_A(x))
			end_time = time.time() - start_time
			print(' Elapsed time Cholesky = %s'%end_time)

		if Type == 'ILU':
			start_time = time.time()
			Pmat = spl.spilu(G,drop_tol = ilu_drop_tol,fill_factor = ilu_fill_factor)
			P_op = spl.LinearOperator((NdofsV,NdofsV), lambda x: Pmat.solve(x))
			end_time = time.time() - start_time
			print(' Elapsed time iLU = %s'%end_time)
		return P_op
	def cg_solver(A,b,x0,P,Niter,tol):
		global call_sum
		call_sum += 1
		#print('----------- First cg inner loop -----------')
		bnorm = np.sqrt(np.sum(b*b))
		r = b-A.dot(x0)
		d = P(r)
		rho = np.abs(np.sum(r*d))
		rnorm = np.sqrt(np.sum(r*r))
		for k in range(Niter):
			if (np.sqrt(rho) > tol*bnorm) and (rnorm>tol*bnorm):
				omega = A.dot(d)
				alpha = rho/(np.sum(omega*d))
				#print('alpha %s' %alpha)
				x0 += alpha*d
				r -= alpha*omega
				rnorm = np.sqrt(np.sum(r*r))
				rhoold = rho
				dold = d
				d = P(r)
				rho = np.sum(d*r)
				d = d + rho/rhoold*dold
			else:
                #if k>0:
                #    print('Niter = %s'%k)
				break
			if call_sum%100==0:
				print('   call %s, Niter = %s'%(call_sum, k))
				#print('tol_cg = %s, niter = %s' %(tol_ac,cgiter))
				#print(resx)
			return x0
	def Schur_solver(A,b,tol_s):
		sol_type = 1
		if sol_type == 1:
			num_iters = 0
			def callback2(xk):
				nonlocal num_iters
				num_iters+=1
		du,auu = spl.cg(A,b, tol = tol_s, callback=callback2)
		print(" Second iterative iterations = %s"%num_iters)
		x0 = du
		k = num_iters
		return x0, k
	def disc_proj(u1, u2, n, h, V, P,type):
		proj = Function(V)
		u = TrialFunction(V)
		v = TestFunction(V)
		if type=='Uh':
			dof_coords = V.tabulate_dof_coordinates().reshape(-1, 2)
				#print(len(dof_coords))
				#u_sol = Function(V)
			for j in range(len(dof_coords)):
				proj.vector()[j] = u1(dof_coords[j])+u2(dof_coords[j])
		elif type=='L2':
			m = u*v*dx
			Me = assemble(m)
			M_mat = as_backend_type(Me).mat()
			M = sp.csc_matrix(sp.csr_matrix(M_mat.getValuesCSR()[::-1], shape = M_mat.size))
			rhs = assemble(u1*v*dx) + assemble(u2*v*dx)
			val,auuu = spl.minres(M,rhs,tol=1e-10)
			proj.vector()[:] = val
		elif type=='H1':
			m = u*v*dx + inner(grad(u),grad(v))*dx
			Me = assemble(m)
			M_mat = as_backend_type(Me).mat()
			M = sp.csc_matrix(sp.csr_matrix(M_mat.getValuesCSR()[::-1], shape = M_mat.size))
			rhs = assemble(u1*v*dx) + assemble(u2*v*dx) + assemble(inner(grad(u1),grad(v))*dx) + assemble(inner(grad(u2),grad(v))*dx)
			val,auuu = spl.cg(M,rhs,tol=1e-7)
			proj.vector()[:] = val
		elif type=='Vh':
			Me = gh(u,v,Constant(1.))
			M_mat = as_backend_type(Me).mat()
			M = sp.csc_matrix(sp.csr_matrix(M_mat.getValuesCSR()[::-1], shape = M_mat.size))
			rhs = gh(u1,v,Constant(1.)) + gh(u2,v,Constant(1.))
			val,auuu = spl.cg(M,rhs,M=P,tol=1e-7)
			proj.vector()[:] = val
		return proj
	boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
	boundary_markers.set_all(0)
	ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
	V = FunctionSpace(mesh, TestType, Ptest)
	U = FunctionSpace(mesh, TrialType, Ptrial)
	e = TrialFunction(V)
	u = TrialFunction(U)
	v = TestFunction(V)
	w = TestFunction(U)
	dofsV = V.dofmap().dofs()
	dofsU = U.dofmap().dofs()
	NdofsV = np.size(dofsV)
	NdofsU = np.size(dofsU)
	Ndofs = NdofsV + NdofsU
	print('dofV = %s, dofU = %s, dofs = %s' %(NdofsV, NdofsU, Ndofs))
	totaliter = 0


	hmin = mesh.hmin()
	hmax = mesh.hmax()
	print('h min = %s, h max = %s' %(hmin, hmax))
	Ge = gh(e,v,Constant(1.))
	G_mat = as_backend_type(Ge).mat()
	G = sp.csc_matrix(sp.csr_matrix(G_mat.getValuesCSR()[::-1], shape = G_mat.size))
	G.eliminate_zeros()
	Be = bh(u,v,Constant(1.))
	Be_mat = as_backend_type(Be).mat()
	B = sp.csc_matrix(sp.csr_matrix(Be_mat.getValuesCSR()[::-1], shape = Be_mat.size))
	B.eliminate_zeros()
	Be_t = bh(w,e,Constant(1.))
	Be_t_mat = as_backend_type(Be_t).mat()
	B_t = sp.csc_matrix(sp.csr_matrix(Be_t_mat.getValuesCSR()[::-1], shape = Be_t_mat.size))
	B_t.eliminate_zeros()
	L = lh(v)
	P_op = Preconditioner(G,NdofsV,Ptype)
	if Gtype == 'NONE':
		Gtil_inv = lambda x: P_op(x)
	if Gtype == 'CG':
		Gtil_inv = lambda x: cg_solver(G,x,P_op(x),P_op,NdofsU,Tol_gram)
	S_lamb = lambda x: B_t.dot(Gtil_inv(B.dot(x)))
	S_op = spl.LinearOperator((NdofsU,NdofsU), S_lamb)
	#initial guess
	e_ass = Function(V)
	u_ass = Function(U)
	deltae = Function(V)
	deltau = Function(U)
	e_past = Function(V)
	u_past = Function(U)
	e_disc = disc_proj(e_ass, deltae, n, h, V, P_op, "Vh")
	u_disc = disc_proj(u_ass, deltau, n, h, U, P_op, "H1")
	e_past.vector()[:] = e_disc.vector()[:]
	u_past.vector()[:] = u_disc.vector()[:]

	if level == 0:
		res_e = np.array(L)
		res_u = np.zeros(NdofsU)
	else:
		e_ass.assign(e_past)
		u_ass.assign(u_past)
		res_e = np.array(L - gh(e_ass,v,Constant(1.))-bh(u_ass,v,Constant(1.)))
		res_u = np.array(-bh(w,e_ass,Constant(1.)))
	sume = sum(res_e*res_e)
	sumu = sum(res_u*res_u)
	sum0 = sume + sumu
	sume0 = sume
	sumu0 = sumu
	print(' sume0 = %s' %sume0)
	print(' sumu0 = %s' %sumu0)
	ave = sum0
		#print('Schur creation done')
	ave = 1
	start_minres = time.time()
	iter_cg = 0
	tol_def = Tol_schur
	while (iter_cg < max_iter_cg) and (ave > tol_ave):
		ave_old = ave
		iter_cg += 1
		niter = 0
		while niter == 0:
			du,niter = Schur_solver(S_op,B_t.dot(Gtil_inv(res_e)) - res_u, tol_def)
			if niter == 0:
				tol_def/=5
				print(' ---------------------------- ')
				print(' new tol cg =%s' %tol_def)
				print(' ---------------------------- ')
		sumu = sum(res_u*res_u)
		totaliter += 1
		de = Gtil_inv(res_e - B.dot(du))
		deltau.vector()[:] += du
		deltae.vector()[:] += de
		#print(res_e)
		res_e += - G.dot(de) - B.dot(du)
		res_u += -B_t.dot(de)
		sumu = sum(res_u*res_u)
		sume = sum(res_e*res_e)
		sumtot = sume + sumu
		ave = sumtot/sum0
		ave_ref = np.abs(ave_old - ave)/ave_old
		print(' iter%s ave = %s' %(iter_cg,ave))
	#inner_iter = totaliter
	#outer_iter = iter_cg
	#avg_iter = 1#totaliter/iter_cg
	#print(' External iter %s' %iter_cg)
		e_disc = disc_proj(e_ass, deltae, n, h, V, P_op, "Vh")
		u_disc = disc_proj(u_ass, deltau, n, h, U, P_op, "H1")
		e_past = Function(V)
		u_past = Function(U)
		e_past.vector()[:] = e_disc.vector()[:]
		u_past.vector()[:] = u_disc.vector()[:]
	return u_disc,e_disc
def refinment(mesh,Ptrial,Ptest,TrialType,TestType,f,uD,beta_vec,kappa,uexact,bh,gh,lh,MAX_ITER,REF_TYPE,REFINE_RATIO):
	for level in range(MAX_ITER):

		u_disc,e_disc=solver_method(mesh,Ptrial,Ptest,TrialType,TestType,f,uD,beta_vec,kappa,uexact,bh,gh,lh,MAX_ITER,level)
		# compute error indicators from the DPG mixed variable e

		PC = FunctionSpace(mesh,"DG",0)
		c  = TestFunction(PC) # p.w constant fn
		g_plot = Function(PC)
		g = gh(e_disc,e_disc,c)
		Ee = np.sum(np.array(g))
		g_plot.vector()[:] = g
		PC2 = FunctionSpace(mesh,"DG", 0)
		# element-wise norms of e
		# E = sqrt(Ee)
		E=Ee
		cont = -1

			# Mark cells for refinement
		cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
		if REF_TYPE == 1:
			rg = np.sort(g)
		rg = rg[::-1]
		rgind = np.argsort(g)
		rgind = rgind[::-1]
		scum = 0.
			#print(sum(g)-E**2)
				# g0 = REFINE_RATIO**2*E**2
		g0 = REFINE_RATIO*E
		    #for cellm in cells(mesh):
		    #     print(cellm)
		Ntot = mesh.num_cells()
		for cellm in cells(mesh):
			if cellm.index() == rgind[0]:
				break
		cell_markers[cellm] = True
		for nj in range(1,Ntot):
			scum += g[rgind[nj]]
		            #cell_ratio = g[rgind[nj]]/g[rgind[0]]
		for cellm in cells(mesh):
			if cellm.index() == rgind[nj]:
				break
			cell_markers[cellm] = scum < g0
			if scum > g0:
		#or cell_ratio < 0.50:
		#print('cell_ratio = %s' %cell_ratio)
				break
		if REF_TYPE == 2:
			g0 = sorted(g, reverse=True)[int(len(g)*REFINE_RATIO)]
		    #gutol = 1e-1
		for cellm in cells(mesh):
			cell_markers[cellm] = g[cellm.index()] > g0

		# REfine the mesh0

		# mesh=refine(mesh,cell_markers)

	return u_disc,e_disc,mesh
