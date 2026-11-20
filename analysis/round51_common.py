#!/usr/bin/env python3
"""Putaran 51: fungsi bersama (penyesuaian dengan penyusutan empirical Bayes, efek kovariat sintetis yang diketahui, indeks keterpulihan kovariat).
Lihat PRAREGISTRASI_putaran51.md. Tidak dijalankan langsung."""
import numpy as np


def ols_coef(Dt, M):
    return np.linalg.lstsq(Dt, M, rcond=None)[0]


def eb_coef(Dt, M):
    """Koefisien penyesuaian dengan penyusutan normal-normal (rata-rata prior 0) per kolom kovariat, lintas fitur, dalam skala terbakukan.
    Dt: desain pasien latih (n x k, kolom 0 = intersep); M: fitur pasien latih (n x p). Untuk kolom k >= 1:
    beta*_kj = B_kj / sigma_j (sigma_j = SD residu OLS fitur j, derajat bebas n - rank), Var(beta*_kj) ~= V_kk dengan V = (Dt'Dt)^-1;
    tau2_k = max(0, rata-rata_j beta*_kj^2 - V_kk); lambda_k = tau2_k / (tau2_k + V_kk); B_kj <- lambda_k B_kj.
    Intersep dipasang ulang sehingga rata-rata residu pasien latih = 0. Mengembalikan (B, lambda)."""
    B = ols_coef(Dt, M); n, k = Dt.shape; R = M - Dt @ B; sig = np.sqrt((R ** 2).sum(0) / (n - np.linalg.matrix_rank(Dt))) + 1e-300
    V = np.linalg.pinv(Dt.T @ Dt); lam = np.ones(k)
    for c in range(1, k):
        if V[c, c] <= 1e-12:                                               # kolom tak teridentifikasi pada pasien latih (mis. level era tidak hadir): koefisien 0
            lam[c] = 0.0; B[c] = 0.0; continue
        bs = B[c] / sig; tau2 = max(0.0, float(np.mean(bs ** 2)) - V[c, c]); lam[c] = tau2 / (tau2 + V[c, c]); B[c] = lam[c] * B[c]
    B[0] = (M - Dt[:, 1:] @ B[1:]).mean(0); return B, lam


def ash_normal(bh, se, nullweight=10.0, gridmult=np.sqrt(2), tol=1e-10, maxit=10000):
    """Adaptive shrinkage (Stephens 2017; default ashr: pointmass=TRUE, prior "nullbiased" nullweight 10, gridmult sqrt(2)) dengan prior campuran normal
    berpusat 0 dan galat baku sama untuk semua efek. Grid: sigmaamin = se/10; sigmaamax = 2 sqrt(max(bh^2 - se^2)) (atau 8 sigmaamin bila semua bh^2 <= se^2);
    sd = gridmult^(-npoint..0) * sigmaamax, didahului 0. Bobot campuran: EM dengan penalti Dirichlet (fixpoint ashr). Mengembalikan rataan posterior."""
    smin = se / 10.0; d = bh ** 2 - se ** 2; smax = 8 * smin if np.all(d <= 0) else 2 * np.sqrt(d.max())
    npt = int(np.ceil(np.log2(smax / smin) / np.log2(gridmult))); sd = np.r_[0.0, gridmult ** np.arange(-npt, 1) * smax]
    var = sd ** 2 + se ** 2; L = np.exp(-0.5 * bh[:, None] ** 2 / var[None, :]) / np.sqrt(2 * np.pi * var[None, :])
    prior = np.ones(len(sd)); prior[0] = nullweight; pi = np.full(len(sd), 1.0 / len(sd))
    for _ in range(maxit):
        m = L * pi; cp = m / m.sum(1, keepdims=True); new = np.maximum(cp.sum(0) + prior - 1, 0); new /= new.sum()
        if np.abs(new - pi).max() < tol:
            pi = new; break
        pi = new
    m = L * pi; w = m / m.sum(1, keepdims=True); return (w * (sd ** 2 / var)[None, :]).sum(1) * bh


def ash_coef(Dt, M):
    """Seperti eb_coef, tetapi tiap kolom kovariat disusutkan per fitur dengan ash_normal atas beta*_kj = B_kj / sigma_j (galat baku sqrt(V_kk)).
    Mengembalikan (B, faktor penyusutan per kolom x fitur)."""
    B = ols_coef(Dt, M); n, k = Dt.shape; R = M - Dt @ B; sig = np.sqrt((R ** 2).sum(0) / (n - np.linalg.matrix_rank(Dt))) + 1e-300
    V = np.linalg.pinv(Dt.T @ Dt); fac = np.ones_like(B)
    for c in range(1, k):
        if V[c, c] <= 1e-12:
            fac[c] = 0.0; B[c] = 0.0; continue
        bs = B[c] / sig; pm = ash_normal(bs, np.sqrt(V[c, c])); fac[c] = np.where(np.abs(bs) > 0, pm / np.where(bs == 0, 1, bs), 0.0); B[c] = pm * sig
    B[0] = (M - Dt[:, 1:] @ B[1:]).mean(0); return B, fac


def patient_means(Z, owner, n):
    S = np.zeros((n, Z.shape[1])); np.add.at(S, owner, Z); cnt = np.bincount(owner, minlength=n); return S / np.maximum(cnt, 1)[:, None]


def adjust_units(Z, owner, tr, kind, D):
    """none / pat (OLS atas rata-rata per pasien latih; identik dengan putaran 42/50) / eb (koefisien eb_coef) / ash (koefisien ash_coef)."""
    if kind == "none":
        return Z
    M = patient_means(Z, owner, len(D))[tr]
    if kind == "pat":
        B = ols_coef(D[tr], M)
    elif kind == "eb":
        B = eb_coef(D[tr], M)[0]
    elif kind == "ash":
        B = ash_coef(D[tr], M)[0]
    else:
        raise ValueError(kind)
    return Z - D[owner] @ B


def true_effect(XR, owner, D):
    """Efek kovariat 'sebenarnya' untuk kondisi E (tanpa label): OLS log rata-rata-per-pasien fitur jendela NYATA (tidak dipertukarkan) pada D, semua pasien.
    Mengembalikan matriks (k-1) x p (tanpa intersep)."""
    M = patient_means(np.log(np.maximum(XR, 1e-12)), owner, len(D)); return ols_coef(D, M)[1:]


def apply_effect(X, owner, D, Bt):
    """Kondisi E: jendela pasien target p dikalikan exp((D_p - rata-rata D) Bt) (aditif dalam skala log)."""
    Dc = D[:, 1:] - D[:, 1:].mean(0); return X * np.exp(Dc[owner] @ Bt)
