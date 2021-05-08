from setuptools import setup, find_packages

setup(
    name='tsne-torch',
    description='t-SNE accelerated with PyTorch',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    version='1.0.0',
    license="MIT",
    install_requires=[
        'torch', 'numpy', 'tqdm'
    ],
    python_requires='>=3.7.0',
    author='Xiao Li, Palle Klewitz',
    packages=find_packages(),
    classifiers=[
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
)
