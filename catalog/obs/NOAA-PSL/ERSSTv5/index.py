import pydap_icechunk
    
def open(varname):
    ds = (
        pydap_icechunk.open_icechunk('obs/NOAA-PSL/ERSSTv5/sst', decode_times=False)
        [[varname]]  # The icechunk store has two variables in it: sst and ssta. Only keep the requested one.
        .rename({'IRIDL_time': 'T', 'lat': 'Y', 'lon': 'X'})
        .squeeze('lev', drop=True)
    )
    return ds
