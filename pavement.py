import string

from paver.easy import *
from paver.setuputils import *
import paver.doctools
from paver.release import setup_meta

__version__ = (0,5,4)

options = environment.options
setup(**setup_meta)

options(
    setup=Bunch(
        name='pydap.responses.wms',
        version='.'.join(str(d) for d in __version__),
        description='WMS response for Pydap',
        long_description='''
Pydap is an implementation of the Opendap/DODS protocol, written from
scratch. This response enables Pydap to serve data as a WMS server.
        ''',
        keywords='wms opendap dods dap data science climate oceanography meteorology',
        classifiers=filter(None, map(string.strip, '''
            Development Status :: 5 - Production/Stable
            Environment :: Console
            Environment :: Web Environment
            Framework :: Paste
            Intended Audience :: Developers
            Intended Audience :: Science/Research
            License :: OSI Approved :: MIT License
            Operating System :: OS Independent
            Programming Language :: Python
            Topic :: Internet
            Topic :: Internet :: WWW/HTTP :: WSGI
            Topic :: Scientific/Engineering
            Topic :: Software Development :: Libraries :: Python Modules
        '''.split('\n'))),
        author='Roberto De Almeida',
        author_email='rob@pydap.org',
        url='http://pydap.org/responses.html#wms',
        license='MIT',

        packages=find_packages(),
        package_data=find_package_data("pydap", package="pydap",
                only_in_packages=False),
        include_package_data=True,
        zip_safe=False,
        namespace_packages=['pydap', 'pydap.responses'],

        test_suite='nose.collector',

        dependency_links=[],
        install_requires=[
            'Pydap==3.1',
            'Paste',
            'matplotlib',
            'coards',
            'iso8601',
        ],
        entry_points="""
            [pydap.response]
            wms = pydap.responses.wms:WMSResponse

            [console_scripts]
            prepare_netcdf = pydap.responses.wms.prepare:prepare_netcdf
        """,
    ),
    minilib=Bunch(
        extra_files=['doctools', 'virtual']
    ),
)


@task
@needs(['generate_setup', 'setuptools.command.sdist'])
def sdist():
    """Overrides sdist to make sure that our setup.py is generated."""
    pass
