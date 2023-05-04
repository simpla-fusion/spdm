# setup.py
#
# Author: Zhi YU
# Created Time: 2015-12-28 21:58:47
#

from setuptools import setup, find_namespace_packages

# Get the long description from the README file
with open('../README.md') as f:
    long_description = f.read()
# Get the version from the VERSION file
with open('../VERSION') as f:
    version = f.read().strip()

# Get the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()


# Setup the package
setup(
    name='spdm',
    version=version,
    description='Scientific Plasma Data Model',
    long_description=long_description,
    url='http://github.com/simpla/spdm',
    author='Zhi YU',
    author_email='yuzhi@ipp.ac.cn',
    license='MIT',
    packages=find_namespace_packages(),  # 指定需要安装的包
    requires=requirements,               # 项目运行依赖的第三方包
    extras_require={},                   # 项目运行依赖的额外包
    package_data={},                     # 需要安装的数据文件，如图片、配置文件等 例如：package_data={'sample': ['package_data.dat']}
    data_files=[],                       # 需要安装的静态文件，如配置文件等。例如：data_files=[('/etc/spdm.conf', ['data/spdm.conf'])]

    entry_points={},  # 项目的入口模块，即用户使用命令行安装后可调用的模块。
                      # 例如：entry_points={'console_scripts': ['spdm = spdm:main']}

    project_urls={},  # 项目相关的额外链接信息。例如：project_urls={'Bug Reports': '...'}
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
    py_modules=['spdm'],  # 单文件模块列表
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
