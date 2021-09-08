import numpy as np
from cvxopt import matrix
import picos

Ne= 128
H_id = np.matrix(H_id)
H_d = np.matrix(H_d)
A_ = H_id.H @ H_id
u_ = H_id.H @ H_d
u_abs2 = picos.Constant(np.multiply(u_,u_.conjugate()))
comp_eye = np.eye(Ne).astype(np.complex128)
comp_zero = np.zeros_like(A_)
M_u = np.concatenate([A_, np.eye(Ne)], axis=0)
M_l = np.concatenate([np.eye(Ne),np.zeros_like(A_)], axis=0)
M_ = picos.Constant(np.matrix(np.concatenate([M_u, M_l], axis=1)))
W_ = picos.HermitianVariable("W_", 2*Ne)
# Define and solve the CVXPY problem.
# The operator >> denotes matrix inequality.


prob = picos.Problem()
prob.set_objective("max",picos.trace(W_*M_))
constraints = [W_ >> 0]
constraints += [
    W_[i,i].real <= 1 for i in range(Ne)
]

constraints += [
    W_[dict(enumerate([range(Ne,2*Ne)]*2))] == u_abs2[:]
]
prob.add_list_of_constraints(constraints)
print(prob)
prob.solve(solver="cvxopt",verbose=True)
print("\nOptimal W_:", W_, sep="\n")
W_sub = W_.value[0:Ne,0:Ne]
w,v = np.linalg.eig(W_sub)
Urnk = len([s for s in w if abs(s) > 1e-6])
print("\nrank(W_sub) =", Urnk)
if np.linalg.matrix_rank(W_sub) ==1:
    theta_init = v @ np.sqrt(w).reshape(-1,1)
else:
    Q = 100
    phi = []
    prod = v @ np.diag(np.sqrt(w))
    P = []
    for q in range(Q):
        r = math.sqrt(0.5)*np.random.randn(Ne, 2).view(np.complex128).reshape(-1,1)
        phiq = np.exp(1j*np.angle(prod @ r))
        Pq = np.linalg.norm(H_d+ H_id@phiq)
        phi.append(phiq)
        P.append(Pq)
    theta_init = phi[[i for i,p in enumerate(P) if p ==max(P)][0]]
