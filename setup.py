import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="WSFpolytope",
    version="0.1",
    author="Robin Schneider",
    author_email="robin.schneider@physics.uu.se",
    description="A sage package to study fibrations of reflexive polytopes and their F-theroy compactifications.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robin-schneider/WSFpolytope",
    packages=setuptools.find_packages(),
    data_files=[('sage', ['WSFpolytope/WSFpolytope.sage']
    )],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=[
        "numpy",
        "pandas",
    ],
)