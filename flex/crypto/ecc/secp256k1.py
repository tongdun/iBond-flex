import hashlib
import hmac
import sys

from typing import (
    Any,
    cast,
    Tuple,
)

from gmpy2 import mpz, f_mod, f_div, powmod, mul, sub
from gmpy2 import add as gmpy_add

if sys.version_info.major == 2:
    safe_ord = ord
else:
    def safe_ord(value: Any) -> int:    # type: ignore
        if isinstance(value, int):
            return value
        else:
            return ord(value)

PlainPoint2D = Tuple[int, int]
PlainPoint3D = Tuple[int, int, int]

# Elliptic curve parameters (secp256k1)
P = 2**256 - 2**32 - 977
N = 115792089237316195423570985008687907852837564279074904382605163141518161494337
A = 0
B = 7
Gx = 55066263022277343669578718895168534326250603453777594175500187360389116729240
Gy = 32670510020758816978083085130507043184471273380659243275938904335757337482424
G = cast("PlainPoint2D", (Gx, Gy))


def bytes_to_int(x: bytes) -> int:
    o = 0
    for b in x:
        o = (o << 8) + safe_ord(b)      # type: ignore
    return o


# Extended Euclidean Algorithm
def inv(a: int, n: int) -> int:
    if a == 0:
        return 0
    lm, hm = 1, 0
    low, high = f_mod(mpz(a), n), n
    while low > 1:
        r = f_div(high, low)
        nm, new = sub(hm, mul(lm, r)), sub(high, mul(low, r))
        lm, low, hm, high = nm, new, lm, low
    return f_mod(mpz(lm), n)


def to_jacobian(p: "PlainPoint2D") -> "PlainPoint3D":
    o = (p[0], p[1], 1)
    return cast("PlainPoint3D", o)


def jacobian_double(p: "PlainPoint3D") -> "PlainPoint3D":
    if not p[1]:
        return cast("PlainPoint3D", (0, 0, 0))
    ysq = powmod(p[1], 2, P)
    S = f_mod(mul(mul(4, p[0]), ysq), P)
    M = f_mod(gmpy_add(mul(3, powmod(p[0], 2, P)), mul(A, powmod(p[2], 4, P))), P)
    nx = f_mod(sub(powmod(M, 2, P), mul(2, S)), P)
    ny = f_mod(sub(mul(M, sub(S, nx)), mul(8, powmod(ysq, 2, P))), P)
    nz = f_mod(mul(mul(2, p[1]), p[2]), P)
    return cast("PlainPoint3D", (nx, ny, nz))


def jacobian_add(p: "PlainPoint3D", q: "PlainPoint3D") -> "PlainPoint3D":
    if not p[1]:
        return q
    if not q[1]:
        return p
    U1 = f_mod(mul(p[0], powmod(q[2], 2, P)), P)
    U2 = f_mod(mul(q[0], powmod(p[2], 2, P)), P)
    S1 = f_mod(mul(p[1], powmod(q[2], 3, P)), P)
    S2 = f_mod(mul(q[1], powmod(p[2], 3, P)), P)
    if U1 == U2:
        if S1 != S2:
            return cast("PlainPoint3D", (0, 0, 1))
        return jacobian_double(p)
    H = sub(U2, U1)
    R = sub(S2, S1)
    H2 = f_mod(mul(H, H), P)
    H3 = f_mod(mul(H, H2), P)
    U1H2 = f_mod(mul(U1, H2), P)
    nx = f_mod(sub(sub(powmod(R, 2, P), H3), mul(2, U1H2)), P)
    ny = f_mod(sub(mul(R, sub(U1H2, nx)), mul(S1, H3)), P)
    nz = f_mod(mul(mul(H, p[2]), q[2]), P)
    return cast("PlainPoint3D", (nx, ny, nz))


def from_jacobian(p: "PlainPoint3D") -> "PlainPoint2D":
    z = inv(p[2], P)
    return cast("PlainPoint2D", (f_mod(mul(p[0], powmod(z, 2, P)), P), f_mod(mul(p[1], powmod(z, 3, P)), P)))


def jacobian_multiply(a: "PlainPoint3D", n: int) -> "PlainPoint3D":   # type: ignore
    if a[1] == 0 or n == 0:
        return cast("PlainPoint3D", (0, 0, 1))
    if n == 1:
        return a
    if n < 0 or n >= N:
        return jacobian_multiply(a, f_mod(mpz(n), N))
    if (n % 2) == 0:
        return jacobian_double(jacobian_multiply(a, f_div(mpz(n), 2)))
    if (n % 2) == 1:
        return jacobian_add(jacobian_double(jacobian_multiply(a, f_div(mpz(n), 2))), a)


def multiply(a: "PlainPoint2D", n: int) -> "PlainPoint2D":
    return from_jacobian(jacobian_multiply(to_jacobian(a), n))


def add(a: "PlainPoint2D", b: "PlainPoint2D") -> "PlainPoint2D":
    return from_jacobian(jacobian_add(to_jacobian(a), to_jacobian(b)))


# bytes32
def privtopub(privkey: bytes) -> "PlainPoint2D":
    return multiply(G, bytes_to_int(privkey))


def deterministic_generate_k(msghash: bytes, priv: bytes) -> int:
    v = b'\x01' * 32
    k = b'\x00' * 32
    k = hmac.new(k, v + b'\x00' + priv + msghash, hashlib.sha256).digest()
    v = hmac.new(k, v, hashlib.sha256).digest()
    k = hmac.new(k, v + b'\x01' + priv + msghash, hashlib.sha256).digest()
    v = hmac.new(k, v, hashlib.sha256).digest()
    return bytes_to_int(hmac.new(k, v, hashlib.sha256).digest())


# bytes32, bytes32 -> v, r, s (as numbers)
def ecdsa_raw_sign(msghash: bytes, priv: bytes) -> Tuple[int, int, int]:

    z = bytes_to_int(msghash)
    k = deterministic_generate_k(msghash, priv)

    r, y = multiply(G, k)
    s = f_mod(mul(inv(k, N), gmpy_add(z, mul(r, bytes_to_int(priv)))), N)

    v, r, s = gmpy_add(27, pow(f_mod(y, 2), 0 if s * 2 < N else 1)), r, s if mul(s, 2) < N else sub(N, s)
    return v, r, s


def ecdsa_raw_recover(msghash: bytes, vrs: Tuple[int, int, int]) -> "PlainPoint2D":
    v, r, s = vrs
    if not (27 <= v <= 34):
        raise ValueError("%d must in range 27-31" % v)
    x = r
    xcubedaxb = f_mod(gmpy_add(gmpy_add(pow(mpz(x), 3), mul(A, x)), B), P)
    beta = powmod(xcubedaxb, f_div(gmpy_add(P, 1), 4), P)
    y = beta if v % 2 ^ beta % 2 else (P - beta)

    # If xcubedaxb is not a quadratic residue, then r cannot be the x coord
    # for a point on the curve, and so the sig is invalid
    if f_mod(sub(xcubedaxb, mul(y, y)), P) != 0 or not f_mod(r, N) or not f_mod(s, N):
        raise ValueError("sig is invalid, %d cannot be the x coord for point on curve" % r)
    z = bytes_to_int(msghash)
    Gz = jacobian_multiply(cast("PlainPoint3D", (Gx, Gy, 1)), (N - z) % N)
    XY = jacobian_multiply(cast("PlainPoint3D", (x, y, 1)), s)
    Qr = jacobian_add(Gz, XY)
    Q = jacobian_multiply(Qr, inv(r, N))
    Q_jacobian = from_jacobian(Q)

    return Q_jacobian