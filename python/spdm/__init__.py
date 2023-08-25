__path__ = __import__('pkgutil').extend_path(__path__, __name__)

try:
    from .__version__ import __version__
except:
    __version__ = "0.0.0"
