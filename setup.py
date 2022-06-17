from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

class CustomDependencyInstallCommand(install):
    """Customized setuptools install command - install all dependencies."""

    @staticmethod
    def _define_dependencies():
        """
        Returns a list of dependency repo urls for pip to download
        Any version updates or additional dependencies should be added here
        """
        dependency_links = [

        ]

        return dependency_links

    def run(self):
        """
        Override original run function of install command,
        with actions to install dependencies
        """
        subprocess.call('echo "Hello, developer, how are you? :)"', shell=True)

        # go through all dependencies defined by user above
        for dep in self._define_dependencies():
            # install them with pip
            subprocess.call('pip install ' + dep, shell=True)

        install.run(self)

setup(
    name='equivariant_models',
    description='',
    version='1.0.0',
    author='Christopher Silkworth',
    author_email='csilkworth@gmail.com',
    packages=find_packages(exclude=['unit_test', 'unitTest', '*.unitTest',
                                    'unitTest.*', '*.unitTest.*', 'projects', 'projects.*']),
    install_requires=[
        'numpy==1.21.0',
        'tensorflow-gpu>=1.5.0',
        'scipy>= 0.19.0',
        'scikit-learn>=0.19.1',
        'scikit-optimize==0.5',
        'opencv-python',
        'matplotlib',
        'Pillow>=5.0.0'
    ],
    cmdclass={
        'install': CustomDependencyInstallCommand
    }
)
