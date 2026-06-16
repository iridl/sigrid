import numpy as np
import xarray as xr

import sigrid.harmonize as harmonize


def test_S_L_to_target():
    s_vals = xr.date_range(
        start='1980-01-01',
        periods=534,
        freq='MS'
    )
    s = xr.DataArray(s_vals, coords={'S': s_vals})
    l_vals = range(12)

    target, target_bnds = harmonize.S_L_to_target_monthly(s, l_vals)
    print(target_bnds)
    assert target_bnds.shape == (534, 12, 2)
    assert target_bnds.isel(S=0, L=0, nbound=0) == np.datetime64('1980-01-01')
    assert target_bnds.isel(S=533, L=11, nbound=1) == np.datetime64('2025-06-01')
    assert target.equals(target_bnds.isel(nbound=0))
