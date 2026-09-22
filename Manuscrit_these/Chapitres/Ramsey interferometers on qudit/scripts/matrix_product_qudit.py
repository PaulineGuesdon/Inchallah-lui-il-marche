"""
Build a total evolution operator U from an ORDERED list of pulses/phases,
without ever having to think about which side of the matrix product each
operator goes on.

The bug this fixes
-------------------
Physically, if you apply operation A first and then operation B, the
correct evolution operator is

    U = B @ A        (B is applied "on the left" because it acts last)

so U|psi> = B(A|psi>): A acts first on the ket. Writing U = A @ B instead
silently applies A *last*, which is the mistake (U = phi * X_pi/2 instead
of U = X_pi/2 * phi).

`compose(*ops)` below takes operators in PHYSICAL / chronological order
(first pulse first) and returns the correctly ordered matrix product, so
you never have to reverse the list by hand again.
"""

import sympy as sp


def phase_gate(N, index, phi):
    """diag(1, ..., 1, e^{-i phi}, 1, ..., 1), phase on basis state `index` (0-based)."""
    d = [sp.Integer(1)] * N
    d[index] = sp.exp(-sp.I * phi)
    return sp.diag(*d)


def x_pi2(N, pairs):
    """
    Block pi/2 beam-splitter, sqrt(2)-normalized, acting simultaneously on the
    2-level subspaces listed in `pairs` (list of 0-based index tuples),
    identity everywhere else. Matches the convention
        [[1, -i], [-i, 1]] / sqrt(2)
    on each pair.
    """
    M = sp.eye(N)
    for (i, j) in pairs:
        M[i, j] = -sp.I
        M[j, i] = -sp.I
    return M / sp.sqrt(2)


def compose(*ops_in_time_order):
    """
    Build U from operators listed in the order they are PHYSICALLY applied.

    compose(A, B, C) means "apply A, then B, then C" (time order) and
    returns U = C @ B @ A, so that U @ psi0 gives the correctly evolved state.
    This is the one place where left/right order matters: get it right here,
    once, and everything built from `compose` will have the correct order.
    """
    U = sp.eye(ops_in_time_order[0].shape[0])
    for step in ops_in_time_order:
        U = step * U          # `step` is applied AFTER everything already in U
    return U


def transform(E, U):
    """Heisenberg picture: E' = U^dagger E U (U.H = conjugate transpose in sympy)."""
    return sp.simplify(U.H * E * U)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Reproduces the 4-level example of SpinControl.tex (eq. 294/298/306):
    # basis order |-9/2>, |-7/2>, |-5/2>, |-3/2>  ->  indices 0, 1, 2, 3
    # ------------------------------------------------------------------
    phi = sp.symbols('phi', real=True)
    N = 4

    Phi = phase_gate(N, index=2, phi=phi)              # phase on |-5/2>
    Xpi2 = x_pi2(N, pairs=[(0, 1), (2, 3)])             # pi/2 pulses on {-9/2,-7/2} and {-5/2,-3/2}

    # Physically: phi accumulates FIRST, THEN the pi/2 pulse is applied.
    # => list them in that order and let compose() get the matrix product right:
    U = compose(Phi, Xpi2)          # == Xpi2 * Phi, i.e. hat U = hat X_pi/2 * hat Phi(phi)
    print("U = X_pi/2 . Phi(phi) :")
    sp.pprint(U)

    # detection operators E_i = |i><i|
    E = [sp.zeros(N) for _ in range(N)]
    for i in range(N):
        E[i][i, i] = 1

    E2p = transform(E[1], U)   # E_2' (|-7/2><-7/2|), should be phi-independent
    E4p = transform(E[3], U)   # E_4' (|-3/2><-3/2|), should carry the e^{+/-i phi} coherence

    print("\nE_2' = U^dagger E_2 U :")
    sp.pprint(E2p)
    print("\nE_4' = U^dagger E_4 U :")
    sp.pprint(E4p)

    # ------------------------------------------------------------------
    # Sanity check: the WRONG order (Phi applied last) kills the phi
    # dependence entirely, because Phi is diagonal and commutes through a
    # diagonal projector E_i -- this is exactly the symptom of the bug.
    # ------------------------------------------------------------------
    U_wrong = compose(Xpi2, Phi)   # == Phi * Xpi2  (the mistake: U = phi * X_pi/2)
    E4p_wrong = transform(E[3], U_wrong)
    print("\n[wrong order U = Phi . X_pi/2] E_4' (phi has vanished!) :")
    sp.pprint(E4p_wrong)
