import os
import setuptools

setuptools.setup(
    name = 'KnowledgeGraph',
    version='1.0.0',
    description='knowledge graph',
    long_description=open(
        os.path.join(
            os.path.dirname(__file__),
            'README.md'
        )
    ).read(),
    author='Yu-Hsiang Lin, Huang-Ting Shieh',
    author_email='wl01154336@gmail.com, huangtingshieh@gmail.com',
    packages=setuptools.find_packages(),
    license='MIT',
)