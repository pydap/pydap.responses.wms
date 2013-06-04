import sys

import numpy 

from pydap.handlers.netcdf import nc, var_attrs
from pydap.responses.wms import fix_data


def prepare_netcdf():
    filename = sys.argv[1]

    if nc.__module__ == 'pupynere':
        raise Exception, "Pupynere cannot open netcdf files in append mode. Please install either PyNIO, netCDF4, Scientific.IO.NetCDF or pynetcdf."

    f = nc(filename, 'a')

    # set actual range
    for name, var in f.variables.items():
        if name in f.dimensions or hasattr(var, 'actual_range'): continue
        data = fix_data(numpy.asarray(var[:]), var_attrs(var)) 
        var.actual_range = numpy.amin(data), numpy.amax(data)

    f.close()
