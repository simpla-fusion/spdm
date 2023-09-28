# setup.py
#
# Author: Zhi YU
# Created Time: 2015-12-28 21:58:47
#

import pathlib
import subprocess

from setuptools import find_namespace_packages, setup
from setuptools.command.build_py import build_py

# Get the long description from the README file
with open('README.md') as f:
    long_description = f.read()

version = subprocess.check_output(['git', 'describe', '--always', '--dirty']).strip().decode('utf-8')

# # Get the version from git or the VERSION file
# with open('VERSION') as f:
#     version = f.read().strip()

# Get the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('LICENSE.txt') as f:
    license = f.read()


git_describe = subprocess.check_output(['git', 'describe', '--always', '--dirty']).strip().decode('utf-8')

source_dir = pathlib.Path(__file__).parent


class BuildPyCommand(build_py):
    description = 'Install __doc__,__version, and IMAS Wrapper'

    def run(self):
        build_dir = pathlib.Path(self.build_lib)/f"{self.distribution.get_name()}"

        super().run()

        with open(build_dir/'__version__.py', 'w') as f:
            f.write(f"__version__ = \"{self.distribution.get_version()}\"")

        if not (build_dir/"__doc__.py").exists():
            with open(build_dir/'__doc__.py', 'w') as f:
                f.write(f'"""\n{self.distribution.get_long_description()}\n"""')


# Setup the package
setup(
    name='spdm',
    version=version,
    description=f'Scientific Plasma Data Model {git_describe}',
    long_description=long_description,
    url='http://github.com/simpla/spdm',
    author='Zhi YU',
    author_email='yuzhi@ipp.ac.cn',
    license=license,

    packages=find_namespace_packages(
        "python", exclude=["*._*", "*.obsolete", "*.obsolete.*", "*.todo", "*.todo.*", "*.tests"]),  # 指定需要安装的包

    package_dir={"": "python"},  # 指定包的root目录

    requires=requirements,                  # 项目运行依赖的第三方包
    # extras_require={},                   # 项目运行依赖的额外包
    # package_data={},                     # 需要安装的数据文件，如图片、配置文件等 例如：package_data={'sample': ['package_data.dat']}
    # data_files=[],                       # 需要安装的静态文件，如配置文件等。例如：data_files=[('/etc/spdm.conf', ['data/spdm.conf'])]

    # cmdclass={'build_py': BuildPyCommand, },


    classifiers=[     # 项目的分类信息列表
        'Development Status :: 1 - Alpha',
        'Intended Audience :: Plasma Physicists',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],  # Optional
    keywords='plasma physics',  # 关键字列表
    python_requires='>=3.10, <4',  # Python版本要求
    # py_modules=[],  # 单文件模块列表
    # scripts=['bin/spdm'],  # 可执行脚本列表
    # package_dir={'': 'src'},  # 项目源码目录结构
    # package_data={'sample': ['package_data.dat']},  # 项目数据文件列表
    # data_files=[('my_data', ['data/data_file'])],  # 项目静态文件列表
    # include_package_data=True,  # 是否包含静态文件
    # zip_safe=False,  # 是否安全压缩
    # ext_modules=[Extension('spdm', ['src/spdm.c'])],  # 项目扩展模块列表
    # cmdclass={'build_ext': build_ext},  # 扩展模块构建命令
    # extras_require={  # Optional
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    # },
    # entry_points={  # Optional
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
)
