from distutils.core import setup

setup(
    name='ImageNetwork',
    version='0.1',
    packages=['ImageNetwork'],
    url='https://github.com/ashkanpakzad/ImageNetwork',
    license='GNU General Public License v3.0',
    author='Ashkan Pakzad',
    author_email='ashkan.pakzad.13@ucl.ac.uk',
    description='Create networks of N dimensional binary images for analysis.',
    install_requires=['networkx>=2.4', 'numpy>=1.18.2']
)
