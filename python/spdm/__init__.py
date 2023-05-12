__path__ = __import__('pkgutil').extend_path(__path__, __name__)

try:
    from .__version__ import __version__
except:
    try:
        import subprocess
        __version__ = subprocess.check_output(['git', 'describe', '--always', '--dirty']).strip().decode('utf-8')
    except:
        __version__ = "0.0.0"
