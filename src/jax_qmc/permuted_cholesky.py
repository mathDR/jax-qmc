import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import ndtr as phi

from scipy.stats._qmvnt import _permuted_cholesky

@jax.jit
def permuted_cholesky_jax(covar, low, high, tol=1e-10):
    n = covar.shape[0]
    sqtp = jnp.sqrt(2 * jnp.pi)
    indices = jnp.arange(n)

    # Initial Scaling
    dc = jnp.sqrt(jnp.maximum(jnp.diag(covar), 0.0))
    dc = jnp.where(dc == 0.0, 1.0, dc)
    new_lo = low / dc
    new_hi = high / dc
    cho = (covar / dc) / dc[:, jnp.newaxis]
    y = jnp.zeros(n)

    def body_fun(k, state):
        cho, new_lo, new_hi, y = state
        
        # 1. Pivot Search (Match NumPy tie-breaking: de <= dem)
        mask_ge_k = indices >= k
        y_masked = jnp.where(indices < k, y, 0.0)
        s_all = jnp.dot(cho, y_masked)
        
        diag_cho = jnp.diag(cho)
        ci_all = jnp.sqrt(jnp.maximum(diag_cho, 0.0))
        lo_all = (new_lo - s_all) / jnp.where(ci_all > 0, ci_all, 1.0)
        hi_all = (new_hi - s_all) / jnp.where(ci_all > 0, ci_all, 1.0)
        de_all = phi(hi_all) - phi(lo_all)
        
        score = jnp.where(mask_ge_k & (diag_cho > tol), de_all, 2.0)
        score = score - (indices * 1e-15) 
        im = jnp.argmin(score)
        
        dem = de_all[im]
        ck = ci_all[im]
        lo_m = lo_all[im]
        hi_m = hi_all[im]

        # 2. Replicate non-standard _swap_slices logic using Masks
        def perform_complex_swaps(c, n_l, n_h):
            # Swap diag elements
            val_k_k = c[k, k]
            val_im_im = c[im, im]
            c = c.at[im, im].set(val_k_k).at[k, k].set(val_im_im)
            
            # _swap_slices(cho, np.s_[im, :k], np.s_[k, :k])
            row_k = c[k, :]
            row_im = c[im, :]
            mask_pre_k = indices < k
            c = c.at[im, :].set(jnp.where(mask_pre_k, row_k, c[im, :]))
            c = c.at[k, :].set(jnp.where(mask_pre_k, row_im, c[k, :]))
            
            # _swap_slices(cho, np.s_[im + 1:, im], np.s_[im + 1:, k])
            col_k = c[:, k]
            col_im = c[:, im]
            mask_post_im = indices > im
            c = c.at[:, im].set(jnp.where(mask_post_im, col_k, c[:, im]))
            c = c.at[:, k].set(jnp.where(mask_post_im, col_im, c[:, k]))
            
            # _swap_slices(cho, np.s_[k + 1:im, k], np.s_[im, k + 1:im])
            mask_mid = (indices > k) & (indices < im)
            seg_col = c[:, k] # Column k, between k and im
            seg_row = c[im, :] # Row im, between k and im
            c = c.at[:, k].set(jnp.where(mask_mid, seg_row, c[:, k]))
            c = c.at[im, :].set(jnp.where(mask_mid, seg_col, c[im, :]))
            
            # Swap bounds
            lo_k, lo_im = n_l[k], n_l[im]
            hi_k, hi_im = n_h[k], n_h[im]
            n_l = n_l.at[k].set(lo_im).at[im].set(lo_k)
            n_h = n_h.at[k].set(hi_im).at[im].set(hi_k)
            
            return c, n_l, n_h

        cho, new_lo, new_hi = jax.lax.cond(
            im > k, 
            perform_complex_swaps, 
            lambda c, l, h: (c, l, h), 
            cho, new_lo, new_hi
        )

        # 3. Update Step
        def update_step(c, nl, nh, y_val):
            yk = jnp.where(
                jnp.abs(dem) > tol,
                (jnp.exp(-lo_m**2 / 2) - jnp.exp(-hi_m**2 / 2)) / (sqtp * dem),
                jnp.where(lo_m < -10, hi_m, jnp.where(hi_m > 10, lo_m, (lo_m + hi_m) / 2))
            )
            y_val = y_val.at[k].set(yk)
            
            ck_safe = jnp.where(ck > 0, ck, 1.0)
            
            # Scale column k
            col_k_raw = c[:, k]
            col_k_scaled = jnp.where(indices > k, col_k_raw / ck_safe, col_k_raw)
            
            # Submatrix rank-1 update (lower triangle only)
            mask_sub = (indices[:, None] > k) & (indices[None, :] > k) & (indices[:, None] >= indices[None, :])
            diff = jnp.outer(col_k_scaled, col_k_raw)
            c = jnp.where(mask_sub, c - diff, c)
            
            # Finalize row k
            c = c.at[:, k].set(col_k_scaled)
            c = c.at[k, k].set(ck)
            row_k = jnp.where(indices <= k, c[k, :] / ck_safe, 0.0)
            c = c.at[k, :].set(row_k)
            
            nl = nl.at[k].divide(ck_safe)
            nh = nh.at[k].divide(ck_safe)
            
            return c, nl, nh, y_val

        def skip_step(c, nl, nh, y_val):
            c = c.at[:, k].set(jnp.where(indices >= k, 0.0, c[:, k]))
            y_val = y_val.at[k].set((nl[k] + nh[k]) / 2)
            return c, nl, nh, y_val

        return jax.lax.cond(ck > (k + 1) * tol, update_step, skip_step, cho, new_lo, new_hi, y)

    final_cho, final_lo, final_hi, _ = jax.lax.fori_loop(0, n, body_fun, (cho, new_lo, new_hi, y))
    return jnp.tril(final_cho), final_lo, final_hi


if __name__ == "__main__":
    # mean = jnp.zeros(2)
    # rho = 0.5
    # cov  = jnp.array([[1.0, rho],[rho, 1.0]])
    # lower = jnp.array([-0.5, -1.5])
    # upper = jnp.array([0.5, 1.5])
    # lo = lower - mean
    # hi = upper - mean
    # L_s, lo_s, hi_s = _permuted_cholesky(cov,lo,hi)
    # L, low, high = permuted_cholesky_jax(cov, lo, hi)
    # print(L_s, lo_s, hi_s)
    # print(L, low, high)

    dim = 100
    mean = jax.random.normal(jax.random.key(seed=42), (dim,))
    cov = jax.random.normal(jax.random.key(seed=43), (dim, dim))
    cov = jnp.dot(cov.T,cov)
    x = jnp.zeros(dim)
    lower = -jnp.inf*jnp.ones_like(x)
    upper = jnp.inf*jnp.ones_like(x)
    L_s,lo_s,hi_s = _permuted_cholesky(cov,lower,upper)
    L,lo,hi = permuted_cholesky_jax(cov,lower,upper)

    # for row in range(L.shape[0]):
    #     print(row, (jnp.allclose(L[row,:],L_s[row,:])))
    row = 3
    print(L_s[row,:row+2])
    print(L[row,:row+2])
    # print(L_s)
    # print(L)