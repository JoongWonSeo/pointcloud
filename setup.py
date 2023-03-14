from setuptools import setup

setup(
    name="pointcloud_vision", # just the name of the package for install/uninstall
    version="0.0.1",
    packages=["pointcloud_vision", "robosuite_envs"], # the name of the packages (folders) that are importable
    # install_requires=["gymnasium==0.26.0"], #TODO: add other dependencies
)