
from scipy.optimize import nnls
import numpy as np
import tqdm
import warnings


def fit_dats(bvalues, dw_data, arg, method, IR=True, rel_times=None):
    """
    Wrapper fit function.
    Returns:
        (Dpar, Fmv, Dmv, Dint, Fint, S0)
    """
    arg = checkarg_lsq(arg)
    if not arg.do_fit:
        return None

    if method != "NNLS":
        raise ValueError("Only method='NNLS' is supported in this aligned build.")

    return fit_NNLS(
        bvalues=np.asarray(bvalues, dtype=float).ravel(),
        dw_data=np.asarray(dw_data, dtype=float),
        IR=IR,
        bounds=arg.bounds,
        rel_times=rel_times,
    )


def tri_expN_noS0(bvalues, Dpar, Fint, Dint, Fmv, Dmv):
    
    return (
        Fmv * np.exp(-bvalues * Dmv)
        + Fint * np.exp(-bvalues * Dint)
        + (1.0 - Fmv - Fint) * np.exp(-bvalues * Dpar)
    )


def tri_expN(bvalues, S0, Dpar, Fint, Dint, Fmv, Dmv):
    """Tri-exponential IVIM."""
    return S0 * tri_expN_noS0(bvalues, Dpar, Fint, Dint, Fmv, Dmv)


def _require_rel_times(rel_times):
    if rel_times is None:
        raise ValueError("rel_times is required when IR=True.")
    req = [
        "echotime", "repetitiontime", "inversiontime",
        "bloodT1", "bloodT2", "tissueT1", "tissueT2", "isfT1", "isfT2"
    ]
    for k in req:
        if not hasattr(rel_times, k):
            raise ValueError(f"rel_times missing required attribute: {k}")


def tri_expN_noS0_IR(bvalues, Dpar, Fint, Dint, Fmv, Dmv, rel_times):
    
    _require_rel_times(rel_times)

    num = (
        (1 - Fmv - Fint)
        * (1 - 2*np.exp(-rel_times.inversiontime/rel_times.tissueT1) + np.exp(-rel_times.repetitiontime/rel_times.tissueT1))
        * np.exp(-rel_times.echotime/rel_times.tissueT2 - bvalues * Dpar)
        + Fint
        * (1 - 2*np.exp(-rel_times.inversiontime/rel_times.isfT1) + np.exp(-rel_times.repetitiontime/rel_times.isfT1))
        * np.exp(-rel_times.echotime/rel_times.isfT2 - bvalues * Dint)
        + Fmv
        * (1 - np.exp(-rel_times.repetitiontime/rel_times.bloodT1))
        * np.exp(-rel_times.echotime/rel_times.bloodT2 - bvalues * Dmv)
    )

    den = (
        (1 - Fmv - Fint)
        * (1 - 2*np.exp(-rel_times.inversiontime/rel_times.tissueT1) + np.exp(-rel_times.repetitiontime/rel_times.tissueT1))
        * np.exp(-rel_times.echotime/rel_times.tissueT2)
        + Fint
        * (1 - 2*np.exp(-rel_times.inversiontime/rel_times.isfT1) + np.exp(-rel_times.repetitiontime/rel_times.isfT1))
        * np.exp(-rel_times.echotime/rel_times.isfT2)
        + Fmv
        * (1 - np.exp(-rel_times.repetitiontime/rel_times.bloodT1))
        * np.exp(-rel_times.echotime/rel_times.bloodT2)
    )

    return num / np.maximum(den, 1e-12)


def tri_expN_IR(bvalues, S0, Dpar, Fint, Dint, Fmv, Dmv, rel_times):
    
    return S0 * tri_expN_noS0_IR(bvalues, Dpar, Fint, Dint, Fmv, Dmv, rel_times)


def correct_for_IR(ampl_Dpar, ampl_Dint, ampl_Dmv, rel_times):
    
    _require_rel_times(rel_times)

    TtLt = np.exp(-rel_times.echotime / rel_times.tissueT2) * (
        1 - 2*np.exp(-rel_times.inversiontime / rel_times.tissueT1)
        + np.exp(-rel_times.repetitiontime / rel_times.tissueT1)
    )
    TbLb = np.exp(-rel_times.echotime / rel_times.bloodT2) * (
        1 - np.exp(-rel_times.repetitiontime / rel_times.bloodT1)
    )
    TpLp = np.exp(-rel_times.echotime / rel_times.isfT2) * (
        1 - 2*np.exp(-rel_times.inversiontime / rel_times.isfT1)
        + np.exp(-rel_times.repetitiontime / rel_times.isfT1)
    )

    eps = 1e-12
    a_par = max(float(ampl_Dpar), 0.0)
    a_int = max(float(ampl_Dint), 0.0)
    a_mv = max(float(ampl_Dmv), 0.0)

    if a_par > 0 and a_int > 0 and a_mv > 0:
        n1 = ((TbLb * a_par) / (max(a_mv, eps) * TtLt)) + 1.0
        n2 = (TbLb * a_int) / (max(a_mv, eps) * TpLp)
        z = 1.0 / max(n1 + n2, eps)  # Fmv
        x = ((TbLb * a_par) / (max(a_mv, eps) * TtLt)) * z  # Fpar
        y = 1.0 - x - z  # Fint
        f_par, f_int, f_mv = x, y, z
    elif a_par > 0 and a_int > 0 and a_mv == 0:
        f_int = 1.0 / (((a_par / max(a_int, eps)) * (TpLp / TtLt)) + 1.0)
        f_par, f_mv = 1.0 - f_int, 0.0
    elif a_par > 0 and a_int == 0 and a_mv > 0:
        f_mv = 1.0 / (((a_par / max(a_mv, eps)) * (TbLb / TtLt)) + 1.0)
        f_par, f_int = 1.0 - f_mv, 0.0
    elif a_par == 0 and a_int > 0 and a_mv > 0:
        f_mv = 1.0 / (((a_int / max(a_mv, eps)) * (TbLb / TpLp)) + 1.0)
        f_int, f_par = 1.0 - f_mv, 0.0
    else:
        s = a_par + a_int + a_mv
        if s <= eps:
            return 0.0, 0.0, 0.0
        f_par, f_int, f_mv = a_par/s, a_int/s, a_mv/s

    f_par = float(np.clip(f_par, 0.0, 1.0))
    f_int = float(np.clip(f_int, 0.0, 1.0))
    f_mv = float(np.clip(f_mv, 0.0, 1.0))
    ss = f_par + f_int + f_mv
    if ss > 0:
        f_par, f_int, f_mv = f_par/ss, f_int/ss, f_mv/ss
    return f_par, f_int, f_mv


def fit_NNLS(
    bvalues,
    dw_data,
    IR=True,
    bounds=([0.9, 0.0001, 0.0, 0.0015, 0.0, 0.004],
            [1.1, 0.0015, 0.4, 0.004, 0.2, 0.2]),
    rel_times=None,
):
    
    try:
        bvalues = np.asarray(bvalues, dtype=float).ravel()
        dw_data = np.asarray(dw_data, dtype=float)

        if dw_data.ndim != 2:
            raise ValueError(f"dw_data must be 2D [vox, bvals], got {dw_data.shape}")
        if dw_data.shape[1] != bvalues.size:
            raise ValueError(f"Mismatch: dw_data has {dw_data.shape[1]} bvals, but bvalues has {bvalues.size}")

        Dspace = np.logspace(np.log10(bounds[0][1]), np.log10(bounds[1][5]), num=200)
        Dbasis = np.exp(-np.outer(bvalues, Dspace))  # [nb, nd]

        nvox = dw_data.shape[0]
        Dpar = np.full(nvox, np.nan, dtype=float)
        Dint = np.full(nvox, np.nan, dtype=float)
        Dmv  = np.full(nvox, np.nan, dtype=float)
        Fint = np.zeros(nvox, dtype=float)
        Fmv  = np.zeros(nvox, dtype=float)
        S0   = np.zeros(nvox, dtype=float)

        idx_parint = int(np.abs(Dspace - bounds[1][1]).argmin())  # Dpar upper
        idx_intmv  = int(np.abs(Dspace - bounds[1][3]).argmin())  # Dint upper

        def _wmean(d, w):
            s = np.sum(w)
            if s <= 0:
                return np.nan
            return float(np.sum(d * w) / s)

        for i in tqdm.tqdm(range(nvox), position=0, leave=True):
            y = dw_data[i, :]
            if not np.isfinite(y).all():
                continue

            x, _ = nnls(Dbasis, y)

            a_par = float(np.sum(x[:idx_parint]))
            a_int = float(np.sum(x[idx_parint:idx_intmv]))
            a_mv  = float(np.sum(x[idx_intmv:]))

            Dpar[i] = _wmean(Dspace[:idx_parint], x[:idx_parint])
            Dint[i] = _wmean(Dspace[idx_parint:idx_intmv], x[idx_parint:idx_intmv])
            Dmv[i]  = _wmean(Dspace[idx_intmv:], x[idx_intmv:])

            if IR:
                _require_rel_times(rel_times)
                _, cFint, cFmv = correct_for_IR(a_par, a_int, a_mv, rel_times=rel_times)
                Fint[i], Fmv[i] = cFint, cFmv
            else:
                s = a_par + a_int + a_mv
                if s > 0:
                    Fint[i], Fmv[i] = a_int / s, a_mv / s

            S0[i] = a_par + a_int + a_mv

        return Dpar, Fmv, Dmv, Dint, Fint, S0

    except Exception as e:
        warnings.warn(f"NNLS failed: {e}")
        n = len(dw_data) if hasattr(dw_data, "__len__") else 0
        z = np.zeros(n, dtype=float)
        return z, z, z, z, z, z


def checkarg_lsq(arg):
    if arg is None:
        class _Tmp:
            pass
        arg = _Tmp()

    if not hasattr(arg, "do_fit"):
        warnings.warn("arg.fit.do_fit not defined. Using default True")
        arg.do_fit = True
    if not hasattr(arg, "fitS0"):
        warnings.warn("arg.fit.fitS0 not defined. Using default True")
        arg.fitS0 = True
    if not hasattr(arg, "jobs"):
        warnings.warn("arg.fit.jobs not defined. Using default 1")
        arg.jobs = 1
    if not hasattr(arg, "bounds"):
        warnings.warn("arg.fit.bounds not defined. Using default 7T-aligned bounds")
        arg.bounds = (
            [0.9, 0.0001, 0.0, 0.0015, 0.0, 0.004],
            [1.1, 0.0015, 0.4, 0.004, 0.2, 0.2],
        )
    return arg