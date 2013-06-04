from __future__ import division

from StringIO import StringIO
import re
import operator
import bisect

from paste.request import construct_url, parse_dict_querystring
from paste.httpexceptions import HTTPBadRequest
from paste.util.converters import asbool
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.cm import get_cmap
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib import rcParams
rcParams['xtick.labelsize'] = 'small'
rcParams['ytick.labelsize'] = 'small'
import iso8601
import coards

from pydap.model import *
from pydap.responses.lib import BaseResponse
from pydap.util.template import GenshiRenderer, StringLoader, TemplateNotFound
from pydap.util.safeeval import expr_eval
from pydap.lib import walk, encode_atom


WMS_ARGUMENTS = ['request', 'bbox', 'cmap', 'layers', 'width', 'height', 'transparent', 'time']


DEFAULT_TEMPLATE = """<?xml version='1.0' encoding="UTF-8" standalone="no" ?>
<!DOCTYPE WMT_MS_Capabilities SYSTEM "http://schemas.opengis.net/wms/1.1.1/WMS_MS_Capabilities.dtd"
 [
 <!ELEMENT VendorSpecificCapabilities EMPTY>
 ]>

<WMT_MS_Capabilities version="1.1.1"
        xmlns="http://www.opengis.net/wms" 
        xmlns:py="http://genshi.edgewall.org/"
        xmlns:xlink="http://www.w3.org/1999/xlink">

<Service>
  <Name>${dataset.name}</Name>
  <Title>WMS server for ${dataset.attributes.get('long_name', dataset.name)}</Title>
  <OnlineResource xlink:href="$location"></OnlineResource>
</Service>

<Capability>
  <Request>
    <GetCapabilities>
      <Format>application/vnd.ogc.wms_xml</Format>
      <DCPType>
        <HTTP>
          <Get><OnlineResource xlink:href="$location"></OnlineResource></Get>
        </HTTP>
      </DCPType>
    </GetCapabilities>
    <GetMap>
      <Format>image/png</Format>
      <DCPType>
        <HTTP>
          <Get><OnlineResource xlink:href="$location"></OnlineResource></Get>
        </HTTP>
      </DCPType>
    </GetMap>
  </Request>
  <Exception>
    <Format>application/vnd.ogc.se_blank</Format>
  </Exception>
  <VendorSpecificCapabilities></VendorSpecificCapabilities>
  <UserDefinedSymbolization SupportSLD="1" UserLayer="0" UserStyle="1" RemoteWFS="0"/>
  <Layer>
    <Title>WMS server for ${dataset.attributes.get('long_name', dataset.name)}</Title>
    <SRS>EPSG:4326</SRS>
    <LatLonBoundingBox minx="${lon_range[0]}" miny="${lat_range[0]}" maxx="${lon_range[1]}" maxy="${lat_range[1]}"></LatLonBoundingBox>
    <BoundingBox CRS="EPSG:4326" minx="${lon_range[0]}" miny="${lat_range[0]}" maxx="${lon_range[1]}" maxy="${lat_range[1]}"/>
    <Layer py:for="grid in layers">
      <Name>${grid.name}</Name>
      <Title>${grid.attributes.get('long_name', grid.name)}</Title>
      <Abstract>${grid.attributes.get('history', '')}</Abstract>
      <?python
          import numpy as np
          from pydap.responses.wms import get_lon, get_lat, get_time
          lon = get_lon(grid, dataset)
          lat = get_lat(grid, dataset)
          time = get_time(grid, dataset)
          minx, maxx = np.min(lon), np.max(lon)
          miny, maxy = np.min(lat), np.max(lat)
      ?>
      <LatLonBoundingBox minx="${minx}" miny="${miny}" maxx="${maxx}" maxy="${maxy}"></LatLonBoundingBox>
      <BoundingBox CRS="EPSG:4326" minx="${minx}" miny="${miny}" maxx="${maxx}" maxy="${maxy}"/>
      <Dimension py:if="time is not None" name="time" units="ISO8601"/>
      <Extent py:if="time is not None" name="time" default="${time[0].isoformat()}/${time[-1].isoformat()}" nearestValue="0">${time[0].isoformat()}/${time[-1].isoformat()}</Extent>
    </Layer>
  </Layer>
</Capability>
</WMT_MS_Capabilities>"""


class WMSResponse(BaseResponse):

    __description__ = "Web Map Service image"

    renderer = GenshiRenderer(
            options={}, loader=StringLoader( {'capabilities.xml': DEFAULT_TEMPLATE} ))

    def __init__(self, dataset):
        BaseResponse.__init__(self, dataset)
        self.headers.append( ('Content-description', 'dods_wms') )

    def __call__(self, environ, start_response):
        # Create a Beaker cache dependent on the query string, since
        # most (all?) pre-computed values will depend on the specific
        # dataset. We strip all WMS related arguments since they don't
        # affect the dataset.
        query = parse_dict_querystring(environ)
        try:
            dap_query = ['%s=%s' % (k, query[k]) for k in query
                    if k.lower() not in WMS_ARGUMENTS]
            dap_query = [pair.rstrip('=') for pair in dap_query]
            dap_query.sort()  # sort for uniqueness
            dap_query = '&'.join(dap_query)
            location = construct_url(environ,
                    with_query_string=True,
                    querystring=dap_query)
            self.cache = environ['beaker.cache'].get_cache(
                    'pydap.responses.wms+' + location)
        except KeyError:
            self.cache = None

        # Handle GetMap and GetCapabilities requests
        type_ = query.get('REQUEST', 'GetMap')
        if type_ == 'GetCapabilities':
            self.serialize = self._get_capabilities(environ)
            self.headers.append( ('Content-type', 'text/xml') )
        elif type_ == 'GetMap':
            self.serialize = self._get_map(environ)
            self.headers.append( ('Content-type', 'image/png') )
        elif type_ == 'GetColorbar':
            self.serialize = self._get_colorbar(environ)
            self.headers.append( ('Content-type', 'image/png') )
        else:
            raise HTTPBadRequest('Invalid REQUEST "%s"' % type_)

        return BaseResponse.__call__(self, environ, start_response)

    def _get_colorbar(self, environ):
        w, h = 90, 300
        query = parse_dict_querystring(environ)
        dpi = float(environ.get('pydap.responses.wms.dpi', 80))
        figsize = w/dpi, h/dpi
        cmap = query.get('cmap', environ.get('pydap.responses.wms.cmap', 'jet'))

        def serialize(dataset):
            fix_map_attributes(dataset)
            fig = Figure(figsize=figsize, dpi=dpi)
            fig.figurePatch.set_alpha(0.0)
            ax = fig.add_axes([0.05, 0.05, 0.45, 0.85])
            ax.axesPatch.set_alpha(0.5)

            # Plot requested grids.
            layers = [layer for layer in query.get('LAYERS', '').split(',')
                    if layer] or [var.id for var in walk(dataset, GridType)]
            layer = layers[0]
            names = [dataset] + layer.split('.')
            grid = reduce(operator.getitem, names)

            actual_range = self._get_actual_range(grid)
            norm = Normalize(vmin=actual_range[0], vmax=actual_range[1])
            cb = ColorbarBase(ax, cmap=get_cmap(cmap), norm=norm,
                    orientation='vertical')
            for tick in cb.ax.get_yticklabels():
                tick.set_fontsize(14)
                tick.set_color('white')
                #tick.set_fontweight('bold')

            # Save to buffer.
            canvas = FigureCanvas(fig)
            output = StringIO() 
            canvas.print_png(output)
            if hasattr(dataset, 'close'): dataset.close()
            return [ output.getvalue() ]
        return serialize

    def _get_actual_range(self, grid):
        try:
            actual_range = self.cache.get_value((grid.id, 'actual_range'))
        except (KeyError, AttributeError):
            try:
                actual_range = grid.attributes['actual_range']
            except KeyError:
                data = fix_data(np.asarray(grid.array[:]), grid.attributes)
                actual_range = np.min(data), np.max(data)
            if self.cache:
                self.cache.set_value((grid.id, 'actual_range'), actual_range)
        return actual_range

    def _get_map(self, environ):
        # Calculate appropriate figure size.
        query = parse_dict_querystring(environ)
        dpi = float(environ.get('pydap.responses.wms.dpi', 80))
        w = float(query.get('WIDTH', 256))
        h = float(query.get('HEIGHT', 256))
        time = query.get('time')
        figsize = w/dpi, h/dpi
        bbox = [float(v) for v in query.get('BBOX', '-180,-90,180,90').split(',')]
        cmap = query.get('cmap', environ.get('pydap.responses.wms.cmap', 'jet'))

        def serialize(dataset):
            fix_map_attributes(dataset)
            fig = Figure(figsize=figsize, dpi=dpi)
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

            # Set transparent background; found through http://sparkplot.org/browser/sparkplot.py.
            if asbool(query.get('TRANSPARENT', 'true')):
                fig.figurePatch.set_alpha(0.0)
                ax.axesPatch.set_alpha(0.0)
            
            # Plot requested grids (or all if none requested).
            layers = [layer for layer in query.get('LAYERS', '').split(',')
                    if layer] or [var.id for var in walk(dataset, GridType)]
            for layer in layers:
                names = [dataset] + layer.split('.')
                grid = reduce(operator.getitem, names)
                if is_valid(grid, dataset):
                    self._plot_grid(dataset, grid, time, bbox, (w, h), ax, cmap)

            # Save to buffer.
            ax.axis( [bbox[0], bbox[2], bbox[1], bbox[3]] )
            ax.axis('off')
            canvas = FigureCanvas(fig)
            output = StringIO() 
            canvas.print_png(output)
            if hasattr(dataset, 'close'): dataset.close()
            return [ output.getvalue() ]
        return serialize

    def _plot_grid(self, dataset, grid, time, bbox, size, ax, cmap='jet'):
        # Get actual data range for levels.
        actual_range = self._get_actual_range(grid)
        V = np.linspace(actual_range[0], actual_range[1], 10)

        # Slice according to time request (WMS-T).
        if time is not None:
            values = np.array(get_time(grid, dataset))
            l = np.zeros(len(values), bool)  # get no data by default

            tokens = time.split(',')
            for token in tokens:
                if '/' in token: # range
                    start, end = token.strip().split('/')
                    start = iso8601.parse_date(start, default_timezone=None)
                    end = iso8601.parse_date(end, default_timezone=None)
                    l[(values >= start) & (values <= end)] = True
                else:
                    instant = iso8601.parse_date(token.strip(), default_timezone=None)
                    l[values == instant] = True
        else:
            l = None

        # Plot the data over all the extension of the bbox.
        # First we "rewind" the data window to the begining of the bbox:
        lon = get_lon(grid, dataset)
        cyclic = hasattr(lon, 'modulo')
        lon = np.asarray(lon[:])
        lat = np.asarray(get_lat(grid, dataset)[:])
        while np.min(lon) > bbox[0]:
            lon -= 360.0
        # Now we plot the data window until the end of the bbox:
        w, h = size
        while np.min(lon) < bbox[2]:
            # Retrieve only the data for the request bbox, and at the 
            # optimal resolution (avoiding oversampling).
            if len(lon.shape) == 1:
                i0, i1 = find_containing_bounds(lon, bbox[0], bbox[2])
                j0, j1 = find_containing_bounds(lat, bbox[1], bbox[3])
                istep = max(1, np.floor( (len(lon) * (bbox[2]-bbox[0])) / (w * abs(lon[-1]-lon[0])) ))
                jstep = max(1, np.floor( (len(lat) * (bbox[3]-bbox[1])) / (h * abs(lat[-1]-lat[0])) ))
                lons = lon[i0:i1:istep]
                lats = lat[j0:j1:jstep]
                data = np.asarray(grid.array[...,j0:j1:jstep,i0:i1:istep])

                # Fix cyclic data.
                if cyclic:
                    lons = np.ma.concatenate((lons, lon[0:1] + 360.0), 0)
                    data = np.ma.concatenate((
                        data, grid.array[...,j0:j1:jstep,0:1]), -1)

                X, Y = np.meshgrid(lons, lats)

            elif len(lon.shape) == 2:
                i, j = np.arange(lon.shape[1]), np.arange(lon.shape[0])
                I, J = np.meshgrid(i, j)

                xcond = (lon >= bbox[0]) & (lon <= bbox[2])
                ycond = (lat >= bbox[1]) & (lat <= bbox[3])
                if not xcond.any() or not ycond.any():
                    lon += 360.0
                    continue

                i0, i1 = np.min(I[xcond]), np.max(I[xcond])
                j0, j1 = np.min(J[ycond]), np.max(J[ycond])
                istep = max(1, int(np.floor( (lon.shape[1] * (bbox[2]-bbox[0])) / (w * abs(np.max(lon)-np.amin(lon))) )))
                jstep = max(1, int(np.floor( (lon.shape[0] * (bbox[3]-bbox[1])) / (h * abs(np.max(lat)-np.amin(lat))) )))

                X = lon[j0:j1:jstep,i0:i1:istep]
                Y = lat[j0:j1:jstep,i0:i1:istep]
                data = grid.array[...,j0:j1:jstep,i0:i1:istep]

            # Plot data.
            if data.shape: 
                # apply time slices
                if l is not None:
                    data = np.asarray(data)[l]

                # reduce dimensions and mask missing_values
                data = fix_data(data, grid.attributes)

                # plot
                if data.any():
                    ax.contourf(X, Y, data, V, cmap=get_cmap(cmap))
            lon += 360.0

    def _get_capabilities(self, environ):
        def serialize(dataset):
            fix_map_attributes(dataset)
            grids = [grid for grid in walk(dataset, GridType) if is_valid(grid, dataset)]

            # Set global lon/lat ranges.
            try:
                lon_range = self.cache.get_value('lon_range')
            except (KeyError, AttributeError):
                try:
                    lon_range = dataset.attributes['NC_GLOBAL']['lon_range']
                except KeyError:
                    lon_range = [np.inf, -np.inf]
                    for grid in grids:
                        lon = np.asarray(get_lon(grid, dataset)[:])
                        lon_range[0] = min(lon_range[0], np.min(lon))
                        lon_range[1] = max(lon_range[1], np.max(lon))
                if self.cache:
                    self.cache.set_value('lon_range', lon_range)
            try:
                lat_range = self.cache.get_value('lat_range')
            except (KeyError, AttributeError):
                try:
                    lat_range = dataset.attributes['NC_GLOBAL']['lat_range']
                except KeyError:
                    lat_range = [np.inf, -np.inf]
                    for grid in grids:
                        lat = np.asarray(get_lat(grid, dataset)[:])
                        lat_range[0] = min(lat_range[0], np.min(lat))
                        lat_range[1] = max(lat_range[1], np.max(lat))
                if self.cache:
                    self.cache.set_value('lat_range', lat_range)

            # Remove ``REQUEST=GetCapabilites`` from query string.
            location = construct_url(environ, with_query_string=True)
            base = location.split('REQUEST=')[0].rstrip('?&')

            context = {
                    'dataset': dataset,
                    'location': base,
                    'layers': grids,
                    'lon_range': lon_range,
                    'lat_range': lat_range,
                    }
            # Load the template using the specified renderer, or fallback to the 
            # default template since most of the people won't bother installing
            # and/or creating a capabilities template -- this guarantees that the
            # response will work out of the box.
            try:
                renderer = environ['pydap.renderer']
                template = renderer.loader('capabilities.xml')
            except (KeyError, TemplateNotFound):
                renderer = self.renderer
                template = renderer.loader('capabilities.xml')

            output = renderer.render(template, context, output_format='text/xml')
            if hasattr(dataset, 'close'): dataset.close()
            return [output.encode('utf-8')]
        return serialize


def is_valid(grid, dataset):
    return (get_lon(grid, dataset) is not None and 
            get_lat(grid, dataset) is not None)


def get_lon(grid, dataset):
    def check_attrs(var):
        if (re.match('degrees?_e', var.attributes.get('units', ''), re.IGNORECASE) or
            var.attributes.get('axis', '').lower() == 'x' or
            var.attributes.get('standard_name', '') == 'longitude'):
            return var

    # check maps first
    for dim in grid.maps.values():
        if check_attrs(dim) is not None:
            return dim

    # check curvilinear grids
    if hasattr(grid, 'coordinates'):
        coords = grid.coordinates.split()
        for coord in coords:
            if coord in dataset and check_attrs(dataset[coord].array) is not None:
                return dataset[coord].array

    return None


def get_lat(grid, dataset):
    def check_attrs(var):
        if (re.match('degrees?_n', var.attributes.get('units', ''), re.IGNORECASE) or
            var.attributes.get('axis', '').lower() == 'y' or
            var.attributes.get('standard_name', '') == 'latitude'):
            return var

    # check maps first
    for dim in grid.maps.values():
        if check_attrs(dim) is not None:
            return dim

    # check curvilinear grids
    if hasattr(grid, 'coordinates'):
        coords = grid.coordinates.split()
        for coord in coords:
            if coord in dataset and check_attrs(dataset[coord].array) is not None:
                return dataset[coord].array

    return None


def get_time(grid, dataset):
    for dim in grid.maps.values():
        if ' since ' in dim.attributes.get('units', ''):
            try:
                return [coards.parse(value, dim.units) for value in np.asarray(dim)]
            except:
                pass

    return None


def fix_data(data, attrs):
    if 'missing_value' in attrs:
        data = np.ma.masked_equal(data, attrs['missing_value'])
    elif '_FillValue' in attrs:
        data = np.ma.masked_equal(data, attrs['_FillValue'])

    if attrs.get('scale_factor'): data *= attrs['scale_factor']
    if attrs.get('add_offset'): data += attrs['add_offset']

    while len(data.shape) > 2:
        ##data = data[0]
        data = np.ma.mean(data, 0)
    return data


def fix_map_attributes(dataset):
    for grid in walk(dataset, GridType):
        for map_ in grid.maps.values():
            if not map_.attributes and map_.name in dataset:
                map_.attributes = dataset[map_.name].attributes.copy()


def find_containing_bounds(axis, v0, v1):
    """
    Find i0, i1 such that axis[i0:i1] is the minimal array with v0 and v1.

    For example::

        >>> from numpy import *
        >>> a = arange(10)
        >>> i0, i1 = find_containing_bounds(a, 1.5, 6.5)
        >>> print a[i0:i1]
        [1 2 3 4 5 6 7]
        >>> i0, i1 = find_containing_bounds(a, 1, 6)
        >>> print a[i0:i1]
        [1 2 3 4 5 6]
        >>> i0, i1 = find_containing_bounds(a, 4, 12)
        >>> print a[i0:i1]
        [4 5 6 7 8 9]
        >>> i0, i1 = find_containing_bounds(a, 4.5, 12)
        >>> print a[i0:i1]
        [4 5 6 7 8 9]
        >>> i0, i1 = find_containing_bounds(a, -4, 7)
        >>> print a[i0:i1]
        [0 1 2 3 4 5 6 7]
        >>> i0, i1 = find_containing_bounds(a, -4, 12)
        >>> print a[i0:i1]
        [0 1 2 3 4 5 6 7 8 9]
        >>> i0, i1 = find_containing_bounds(a, 12, 19)
        >>> print a[i0:i1]
        []

    It also works with decreasing axes::

        >>> b = a[::-1]
        >>> i0, i1 = find_containing_bounds(b, 1.5, 6.5)
        >>> print b[i0:i1]
        [7 6 5 4 3 2 1]
        >>> i0, i1 = find_containing_bounds(b, 1, 6)
        >>> print b[i0:i1]
        [6 5 4 3 2 1]
        >>> i0, i1 = find_containing_bounds(b, 4, 12)
        >>> print b[i0:i1]
        [9 8 7 6 5 4]
        >>> i0, i1 = find_containing_bounds(b, 4.5, 12)
        >>> print b[i0:i1]
        [9 8 7 6 5 4]
        >>> i0, i1 = find_containing_bounds(b, -4, 7)
        >>> print b[i0:i1]
        [7 6 5 4 3 2 1 0]
        >>> i0, i1 = find_containing_bounds(b, -4, 12)
        >>> print b[i0:i1]
        [9 8 7 6 5 4 3 2 1 0]
        >>> i0, i1 = find_containing_bounds(b, 12, 19)
        >>> print b[i0:i1]
        []
    """
    ascending = axis[1] > axis[0]
    if not ascending: axis = axis[::-1]
    i0 = i1 = len(axis)
    for i, value in enumerate(axis):
        if value > v0 and i0 == len(axis):
            i0 = i-1
        if not v1 > value and i1 == len(axis):
            i1 = i+1
    if not ascending: i0, i1 = len(axis)-i1, len(axis)-i0
    return max(0, i0), min(len(axis), i1)
