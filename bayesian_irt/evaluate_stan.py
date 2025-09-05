import matplotlib.pyplot as plt
import re
import numpy as np
import arviz as az
from cmdstanpy import CmdStanMCMC, CmdStanVB
from pathlib import Path

def expand_param_names(all_names, wanted):
    """
    Expand vector/matrix parameters: if 'mu_org' is in wanted,
    match 'mu_org' and all 'mu_org[<i>]' from all_names.
    """
    keep = []
    for w in wanted:
        regex = re.compile(f"^{re.escape(w)}(\\[.*\\])?$")  # matches w and w[...]
        keep.extend([n for n in all_names if regex.match(n)])
    return keep

_name_re = re.compile(r'^(?P<base>[^\[]+)(?:\[(?P<idxs>[^\]]+)\])?$')

def _parse_colname(colname):
    """
    'theta[1,3]' -> ('theta', (0,2))
    'mu_org[4]'  -> ('mu_org', (0,))
    'sigma'      -> ('sigma', None)
    """
    m = _name_re.match(colname)
    base = m.group('base')
    idxs = m.group('idxs')
    if idxs is None:
        return base, None
    return base, tuple(int(s) - 1 for s in idxs.split(','))  # 1-based -> 0-based

def _pack_draws(draws, colnames, keep_bases=None):
    """
    Pack column-wise draws into dict of arrays with sample dims first.
    Accepts draws of shape (chains, draws, cols) or (draws, cols).
    Returns dict: name -> array of shape (chains, draws, *param_shape).
    """
    arr = np.asarray(draws)
    if arr.ndim == 2:                 # (draws, cols) -> add size-1 chain
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"draws must be (C,S,K) or (S,K); got {arr.shape}")
    C, S, K = arr.shape

    # group columns
    groups = {}  # base -> list[(col_idx, idx_tuple_or_None)]
    for j, name in enumerate(colnames):
        if name == 'lp__':            # skip by default
            continue
        base, idx = _parse_colname(name)
        if keep_bases is not None and base not in keep_bases:
            continue
        groups.setdefault(base, []).append((j, idx))

    packed = {}
    for base, entries in groups.items():
        # scalar?
        if all(idx is None for _, idx in entries):
            if len(entries) != 1:
                raise ValueError(f"Multiple scalar columns found for '{base}'.")
            jcol, _ = entries[0]
            packed[base] = arr[:, :, jcol]          # (C, S)
            continue

        # non-scalar: infer shape from max index per axis
        idx_list = [idx for _, idx in entries if idx is not None]
        k = len(idx_list[0])
        if any(len(t) != k for t in idx_list):
            raise ValueError(f"Inconsistent index arity for '{base}'.")
        max_idx = [0] * k
        for t in idx_list:
            for a, v in enumerate(t):
                if v > max_idx[a]:
                    max_idx[a] = v
        shape = tuple(v + 1 for v in max_idx)       # sizes for param axes

        out = np.empty((C, S, *shape), dtype=np.float64)
        for jcol, idx in entries:
            if idx is None:
                raise ValueError(f"Mixed indexed and non-indexed columns for '{base}'.")
            out[(slice(None), slice(None)) + idx] = arr[:, :, jcol]
        packed[base] = out

    return packed

def _make_dims_coords_two_sample_dims(posterior):
    """
    Build per-variable dims/coords for arrays shaped (chain, draw, ...).
    Avoid shared generic dim names across variables.
    """
    dims, coords = {}, {}
    for name, v in posterior.items():
        shp = np.asarray(v).shape
        if len(shp) <= 2:
            continue  # scalar: only (chain, draw)
        nd_extra = len(shp) - 2
        var_dims = []
        for i in range(nd_extra):
            dname = f"{name}_dim_{i}"
            var_dims.append(dname)
            coords[dname] = np.arange(shp[2 + i])
        dims[name] = var_dims
    return dims, coords

def _normalize_sample_axes(draws, n_chains, n_draws):
    """
    Ensure draws have shape (chains, draws, cols).

    Accepts:
      - (chains, draws, cols)  -> pass-through
      - (draws, chains, cols)  -> swap first two axes
      - (draws, cols)          -> add chain axis -> (1, draws, cols)
    """
    arr = np.asarray(draws)
    if arr.ndim == 2:  # (S, K)
        return arr[None, ...]  # (1, S, K)
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D draws, got {arr.shape}")

    c, s, k = arr.shape
    if c == n_chains and s == n_draws:
        return arr  # already (C, S, K)
    if c == n_draws and s == n_chains:
        return np.transpose(arr, (1, 0, 2))  # (S, C, K) -> (C, S, K)

    raise ValueError(
        f"Cannot normalize draws with shape {arr.shape} "
        f"given n_chains={n_chains}, n_draws={n_draws}."
    )


class StanModel:
    def __init__(self, source, stan_file=None, param_names=None):
        """
        Wrapper around Stan outputs for convenient ArviZ integration.

        Parameters
        ----------
        source : CmdStanMCMC | CmdStanVB | str | Path
            CmdStanPy fit or path to saved results (.zarr dir, .nc file, or CSV dir).
        stan_file : str | Path, optional
            Path to the Stan source file (attached as metadata).
        param_names : list[str], optional
            Subset of parameters to keep (ONLY when source is a fit object).
            For VB, names like 'mu_org' expand to 'mu_org[1]', 'mu_org[2]', ...
        """
        self.param_names = param_names

        # ----- Fit objects -----
        if isinstance(source, CmdStanMCMC):
            print("[StanModel] Loading from CmdStanPy MCMC (column-pack path)")
            raw = source.draws(concat_chains=False)  # could be (C,S,K) or (S,C,K)
            draws = _normalize_sample_axes(raw, source.chains, source.num_draws_sampling)  # -> (C,S,K)

            all_names = source.column_names
            if param_names is not None:
                keep_names = expand_param_names(all_names, param_names)
                keep_bases = set(param_names)
            else:
                keep_names = all_names
                keep_bases = None

            idx = [all_names.index(k) for k in keep_names]
            draws_sel = draws[..., idx]
            colnames_sel = [all_names[i] for i in idx]

            posterior = _pack_draws(draws_sel, colnames_sel, keep_bases=keep_bases)
            dims, coords = _make_dims_coords_two_sample_dims(posterior)
            self.idata = az.from_dict(posterior=posterior, dims=dims, coords=coords)




        elif isinstance(source, CmdStanVB):
            print("[StanModel] Loading from CmdStanPy VB")
            draws = source.variational_sample            # (draws, cols)
            all_names = source.column_names

            if param_names is not None:
                keep_names = expand_param_names(all_names, param_names)
                keep_bases = set(param_names)
            else:
                keep_names = all_names
                keep_bases = None

            idx = [all_names.index(k) for k in keep_names]
            draws_sel = draws[:, idx]                   # (S, M)
            colnames_sel = [all_names[i] for i in idx]

            posterior = _pack_draws(draws_sel, colnames_sel, keep_bases=keep_bases)  # -> (1, S, …)
            dims, coords = _make_dims_coords_two_sample_dims(posterior)
            self.idata = az.from_dict(posterior=posterior, dims=dims, coords=coords)




        # ----- File-based sources (always full InferenceData) -----
        else:
            src = Path(source)
            if src.is_file() and src.suffix == ".nc":
                print(f"[StanModel] Loading from NetCDF: {src}")
                self.idata = az.from_netcdf(src)
            elif src.is_dir() and src.suffix == ".zarr":
                print(f"[StanModel] Loading from Zarr: {src}")
                self.idata = az.from_zarr(src)
            elif src.is_dir():
                print(f"[StanModel] Parsing CmdStan CSVs in: {src}")
                csvs = glob.glob(str(src / "*.csv"))
                if not csvs:
                    raise RuntimeError(f"No Stan CSVs found in {src}")
                self.idata = az.from_cmdstan(csvs)
            else:
                raise ValueError(f"Unsupported source: {source}")

        # ----- Attach Stan code (if provided) -----
        if stan_file is not None:
            self.stan_code = Path(stan_file).read_text()
            self.idata.attrs["stan_code"] = self.stan_code
        else:
            self.stan_code = self.idata.attrs.get("stan_code", None)

        # Convenience handle to posterior variables
        self.parameters = {name: var for name, var in self.idata.posterior.items()}

    # ----- Save -----
    def save(self, filepath, format="netcdf"):
        """
        Save the current InferenceData (exactly what's in self.idata) to disk.

        Parameters
        ----------
        filepath : str | Path
            Base path (".zarr" or ".nc" will be added if missing).
        format : {"zarr", "netcdf"}, default="zarr"
            Storage format.
        """
        from pathlib import Path
        filepath = Path(filepath)

        if format == "netcdf":
            if filepath.suffix != ".nc":
                filepath = filepath.with_suffix(".nc")
            print(f"[StanModel] Saving InferenceData to NetCDF: {filepath}")
            self.idata.to_netcdf(filepath)

        elif format == "zarr":
            if filepath.suffix != ".zarr":
                filepath = filepath.with_suffix(".zarr")
            print(f"[StanModel] Saving InferenceData to Zarr: {filepath}")
            self.idata.to_zarr(filepath)

        else:
            raise ValueError("format must be 'netcdf' or 'zarr'")

    def show_model(self, n=20):
        """Print first n lines of Stan code (if available)."""
        if self.stan_code:
            for i, line in enumerate(self.stan_code.splitlines()[:n], start=1):
                print(f"{i:02d}: {line}")
        else:
            print("No Stan code stored in this object.")

    # -------------------------------
    # Convenience methods
    # -------------------------------

    def list_parameters(self):
        return list(self.parameters.keys())

    def list_sampler_stats(self):
        """Return list of all sampler stats available."""
        return list(self.idata.sample_stats.keys())

    def parameter_shapes(self):
        return {k: v.shape for k, v in self.parameters.items()}

    def summary(self, var_names=None, round_to=2):
        return az.summary(self.idata, var_names=var_names, round_to=round_to)

    def plot_trace(self, var_names=None):
        return az.plot_trace(self.idata, var_names=var_names)

    def plot_posterior(self, var_names=None, hdi_prob=0.95):
        return az.plot_posterior(self.idata, var_names=var_names, hdi_prob=hdi_prob)

    def plot_pair(self, var_names=None, kind="kde"):
        return az.plot_pair(self.idata, var_names=var_names, kind=kind)

    def extract(self, var_names=None):
        """Extract draws as pandas DataFrame."""
        return az.extract(self.idata, var_names=var_names).to_pandas()


    def _family_members(self, base):
        """
        Find all scalarized members like 'base[1]' / 'base[2,3]' in self.parameters.
        Returns sorted list of (full_name, idx_tuple_zero_based, dataarray).
        """
        pat = re.compile(rf"^{re.escape(base)}\[(.+)\]$")
        members = []
        for name, arr in self.parameters.items():
            m = pat.match(name)
            if m:
                idx = tuple(int(s.strip()) - 1 for s in m.group(1).split(","))
                members.append((name, idx, arr))
        members.sort(key=lambda x: x[1])  # sort by index tuple
        return members

    def _parse_param_name(self, name):
        """
        Parse 'theta[10,2]' -> ('theta', (9,1)); 'theta' -> ('theta', None).
        (No dependency on arr dims; works even if base isn't present as a single var.)
        """
        m = re.match(r"^([A-Za-z0-9_]+)(?:\[(.*)\])?$", name)
        if not m:
            raise ValueError(f"Invalid parameter name: {name}")
        base = m.group(1)
        if not m.group(2):
            return base, None
        idx = tuple(int(x.strip()) - 1 for x in m.group(2).split(","))
        return base, idx

    def plot_histogram(self, param, hdi_prob=0.95, bins=30, max_plots=16, figsize=(12, 8), true_values=None):
        """
        Plot histogram(s) of a parameter with mean + HDI, and optional true value(s).

        - If param is indexed (e.g. 'theta[10,2]'), plot that element.
        - If param is a vector/matrix with <= max_plots elements, plot all elements
          arranged in a grid.
        - If param has more than max_plots elements, ask user to specify indices.
        - true_values can be:
            * scalar: draw same vertical line in all subplots
            * array/list of length equal to number of plotted elements: per-subplot lines
        """
        import numpy as np

        base, idx = self._parse_param_name(param)

        def _truth_for(k, total):
            if true_values is None:
                return None
            if np.isscalar(true_values):
                return float(true_values)
            tv = np.asarray(true_values, dtype=float)
            if tv.shape[0] != total:
                raise ValueError(f"'true_values' length {tv.shape[0]} != number of plotted elements {total}.")
            return float(tv[k])

        # Case A: grouped param (MCMC fast path)
        if base in self.parameters:
            arr = self.parameters[base]
            extra_dims = tuple(d for d in arr.dims if d not in ("chain", "draw"))

            if idx is not None:  # specific element
                sel = {extra_dims[i]: idx[i] for i in range(len(idx))} if extra_dims else {}
                data = arr.isel(**sel).values.ravel()
                self._plot_single_hist(data, param, hdi_prob, bins, figsize, true_value=_truth_for(0, 1))
                return

            if not extra_dims:  # scalar
                self._plot_single_hist(arr.values.ravel(), base, hdi_prob, bins, figsize, true_value=_truth_for(0, 1))
                return

            shape = [arr.sizes[d] for d in extra_dims]
            n_elem = int(np.prod(shape))
            if n_elem > max_plots:
                print(f"[StanModel] {base} has {n_elem} elements. Please specify indices (e.g. {base}[i,...]).")
                return

            ncols = int(np.ceil(np.sqrt(n_elem)))
            nrows = int(np.ceil(n_elem / ncols))

            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
            axes = axes.ravel()
            for k, idx_tuple in enumerate(np.ndindex(*shape)):
                sel = {extra_dims[i]: idx_tuple[i] for i in range(len(idx_tuple))}
                da = arr.isel(**sel)
                label = f"{base}[{','.join(str(i + 1) for i in idx_tuple)}]"
                self._plot_single_hist(
                    da.values.ravel(),
                    label, hdi_prob, bins,
                    ax=axes[k], show=False,
                    true_value=_truth_for(k, n_elem)
                )
            for ax in axes[n_elem:]:  # empty subplots
                ax.axis("off")
            plt.tight_layout()
            plt.show()
            return

        # Case B: scalarized family
        members = self._family_members(base)
        if not members:
            raise ValueError(f"Parameter {base} not found. Available: {list(self.parameters.keys())[:20]}...")

        if idx is not None:  # specific element
            full_name = f"{base}[{','.join(str(i + 1) for i in idx)}]"
            arr = self.parameters[full_name]
            self._plot_single_hist(arr.values.ravel(), full_name, hdi_prob, bins, figsize, true_value=_truth_for(0, 1))
            return

        n_elem = len(members)
        if n_elem > max_plots:
            print(f"[StanModel] {base} has {n_elem} elements. Please specify indices (e.g. {base}[i] or {base}[i,j]).")
            return

        ncols = int(np.ceil(np.sqrt(n_elem)))
        nrows = int(np.ceil(n_elem / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.ravel()
        for k, (name, _idx_tuple, da) in enumerate(members):
            self._plot_single_hist(da.values.ravel(), name, hdi_prob, bins, ax=axes[k], show=False,
                                   true_value=_truth_for(k, n_elem))
        for ax in axes[n_elem:]:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    def _plot_single_hist(self, values, label, hdi_prob, bins, figsize=(6, 4), ax=None, show=True, true_value=None):
        import numpy as np
        mean_val = float(np.mean(values))
        hdi = az.hdi(values, hdi_prob=hdi_prob)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        ax.hist(values, bins=bins, density=True, alpha=0.6, color="steelblue")
        ax.axvline(mean_val, color="black", linestyle="--", label=f"mean={mean_val:.2f}")
        ax.axvline(hdi[0], color="black", linestyle=":", label=f"{int(hdi_prob * 100)}% HDI")
        ax.axvline(hdi[1], color="black", linestyle=":")

        if true_value is not None:
            ax.axvline(float(true_value), color="red", linestyle="-.", linewidth=1.5, label="true")

        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend()

        if show:
            plt.show()

    def plot_forest(self, base, hdi_prob=0.95, max_items=200, order="none",
                    true_values=None, figsize=(6, 8), ref_line=None):
        """
        Forest plot for a parameter family (vector/matrix), showing mean and HDI.

        Parameters
        ----------
        base : str
            Family name, e.g. "mu_org" or "mu_team".
        hdi_prob : float
            HDI probability (e.g., 0.95).
        max_items : int
            Max number of items to plot (avoid gigantic figures).
        order : {"none","mean","abs"}
            Sort items by mean or absolute mean; "none" keeps natural order.
        true_values : float | sequence[float] | None
            Ground truth(s) to mark.
              - scalar: vertical line at that value
              - array-like: per-item markers (must match number of items)
        figsize : tuple
            Figure size.
        ref_line : float | None
            Draw a global reference vertical line (e.g., 0.0). Set None to disable.
        """
        # ----- collect stats -----
        if base in self.parameters:  # grouped param
            arr = self.parameters[base]
            sample_dims = tuple(d for d in arr.dims if d in ("chain", "draw"))
            extra_dims = tuple(d for d in arr.dims if d not in ("chain", "draw"))
            items = []
            if not extra_dims:
                vals = arr.values.ravel()
                mean = float(np.mean(vals))
                hdi = az.hdi(vals, hdi_prob=hdi_prob)
                items.append((base, mean, float(hdi[0]), float(hdi[1])))
            else:
                shape = [arr.sizes[d] for d in extra_dims]
                for idx_tuple in np.ndindex(*shape):
                    sel = {extra_dims[i]: idx_tuple[i] for i in range(len(idx_tuple))}
                    vals = arr.isel(**sel).values.ravel()
                    mean = float(np.mean(vals))
                    hdi = az.hdi(vals, hdi_prob=hdi_prob)
                    label = f"{base}[{','.join(str(i + 1) for i in idx_tuple)}]"
                    items.append((label, mean, float(hdi[0]), float(hdi[1])))
        else:  # scalarized family
            members = self._family_members(base)
            if not members:
                raise ValueError(f"Parameter family '{base}' not found.")
            items = []
            for name, idx_tuple, da in members:
                vals = da.values.ravel()
                mean = float(np.mean(vals))
                hdi = az.hdi(vals, hdi_prob=hdi_prob)
                items.append((name, mean, float(hdi[0]), float(hdi[1])))

        n = len(items)
        if n > max_items:
            raise ValueError(f"{base} has {n} elements; set max_items higher or subset indices.")

        # ----- ordering -----
        if order == "mean":
            items.sort(key=lambda x: x[1])
        elif order == "abs":
            items.sort(key=lambda x: abs(x[1]))

        labels = [it[0] for it in items]
        means = np.array([it[1] for it in items])
        lowers = np.array([it[2] for it in items])
        uppers = np.array([it[3] for it in items])

        y = np.arange(n)[::-1]

        fig, ax = plt.subplots(figsize=figsize)

        # HDIs
        ax.hlines(y, lowers, uppers, linewidth=2)
        # Means
        ax.plot(means, y, "o", label="mean", color="C0")

        # ----- global reference line -----
        if ref_line is not None:
            ax.axvline(ref_line, linestyle="--", linewidth=1, color="gray", label="ref")

        # ----- true values -----
        # ----- true values -----
        if true_values is not None:
            if np.isscalar(true_values):
                ax.axvline(float(true_values), linestyle="-.", linewidth=1,
                           color="red", label="true (scalar)")
            else:
                tv = np.asarray(true_values, dtype=float)
                if tv.shape[0] != n:
                    raise ValueError(f"true_values has length {tv.shape[0]} but {n} items were plotted.")
                # Plot ticks instead of blobs
                ax.plot(tv, y, marker="|", markersize=12, color="red",
                        linestyle="None", label="true")

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(base)
        ax.set_title(f"{base} — mean and {int(hdi_prob * 100)}% HDI")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def get_draws(self, params, n=None, random=True):
        """
        Extract posterior draws for a list of parameters.

        params : list of str
            Parameter names to extract (must exist in self.parameters).
        n : int or None
            Number of draws to return. If None, return all draws.
        random : bool
            If True, sample n draws randomly from the posterior.
            If False, take the first n draws in order.

        Returns
        -------
        dict : {param_name: np.ndarray}
            Each entry is an array with shape (n_draws, ...) where ... is the
            parameter's shape in the Stan model.
        """
        out = {}
        for name in params:
            if name not in self.parameters:
                raise KeyError(f"Parameter {name} not found in posterior.")

            arr = self.parameters[name].values  # shape (chains, draws, ...)
            flat = arr.reshape(-1, *arr.shape[2:])  # flatten chains

            if n is None or n >= flat.shape[0]:
                sel = flat
            else:
                idx = (np.random.choice(flat.shape[0], size=n, replace=False)
                       if random else np.arange(n))
                sel = flat[idx]

            out[name] = sel
        return out

    def generate_quantities(self, *args, **kwargs):
        """Forward generate_quantities to the underlying CmdStanModel."""
        if self._stan_model is None:
            raise RuntimeError("Stan file was not provided; cannot call generate_quantities.")
        return self._stan_model.generate_quantities(*args, **kwargs)

if __name__ == "__main__":
    from pathlib import Path
    import glob
    from cmdstanpy import CmdStanModel
    # your own helpers


    here = Path(__file__).resolve().parent
    nc_path = here.parent / "saved_nc_models" / "nested_orgs.nc"
    assert nc_path.exists(), "nc path doesn't exists"
    model = StanModel(source=nc_path)
    model.list_parameters()



