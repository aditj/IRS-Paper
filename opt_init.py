import numpy as np
from cvxopt import matrix
import picos

Ne= 256
H_id = np.matrix(H_id)
H_d = np.matrix(H_d)
A_ = picos.Constant(H_id.H @ H_id)
u_ = H_id.H @ H_d
u_abs2 = picos.Constant(np.multiply(u_,u_.conjugate())[:])
w1 = picos.HermitianVariable("w1",Ne)
w2 = picos.HermitianVariable("w2",Ne)
w3 = picos.HermitianVariable("w3",Ne)
W_ = picos.block([[w1,w2],[w2.H,w3]], shapes = ((Ne,Ne), (Ne,Ne)))
M_ = picos.block([[A_,"I"],["I",0]], shapes = ((Ne,Ne), (Ne,Ne)))



# Define and solve the picos problem.
# The operator >> denotes matrix inequality.
obj = picos.trace(w1*A_+2*w2).refined

constraints = [W_ >> 0]
constraints += [
    w1[i,i].real <= 1 for i in range(Ne)
]

constraints += [
    picos.maindiag(w3) == u_abs2
]

P = picos.Problem()
P.set_objective("max", obj)
P.add_list_of_constraints(constraints)
print(P)

prob.solve(solver="cvxopt",verbose=True)
print("\nOptimal W_sub:", w1, sep="\n")

w,v = np.linalg.eig(w1.value)
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
