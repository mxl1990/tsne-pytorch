from setuptools import setup, find_packages

setup(
    name='tsne-torch',
    description='t-SNE accelerated with PyTorch',
    version='1.0.0',
    install_requires=[
        'torch', 'numpy', 'tqdm'
    ],
    python_requires='>=3.7.0',
    author='Xiao Li, Palle Klewitz',
    packages=find_packages(),
)
