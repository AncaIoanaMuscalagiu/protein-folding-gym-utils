from setuptools import setup, find_packages

VERSION = '0.0.2'
DESCRIPTION = 'Package with Reinforcement Learning Environments for Protein Folding'
LONG_DESCRIPTION = 'Python Library implementing various Gym Environments for solving the Protein Folding Problem using Reinforcement Learning'

setup(
    name="protein-folding-gym-utilities",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Muscalagiu Anca Ioana",
    author_email="ancamuscalagiu1@gmail.com",
    license='MIT',
    packages=find_packages(),
    install_requires=['bio==1.5.9',
'biopython==1.81',
'biothings-client==0.3.0',
'certifi==2022.12.7',
'charset-normalizer==3.1.0',
'cloudpickle==2.2.1',
'contourpy==1.0.7',
'cycler==0.11.0',
'fonttools==4.39.3',
'gprofiler-official==1.0.0',
'gym==0.26.2',
'gym-notices==0.0.8',
'importlib-metadata==6.6.0',
'importlib-resources==5.12.0',
'joblib==1.2.0',
'kiwisolver==1.4.4',
'matplotlib==3.7.1',
'mygene==3.2.2',
'numpy==1.23.5',
'packaging==23.1',
'pandas==2.0.1',
'Pillow==9.5.0',
'platformdirs==3.3.0',
'pooch==1.7.0',
'pyparsing==3.0.9',
'python-dateutil==2.8.2',
'pytz==2023.3',
'requests==2.28.2',
'scikit-learn==1.2.2',
'scipy==1.10.1',
'six==1.16.0',
'threadpoolctl==3.1.0',
'tqdm==4.65.0',
'tzdata==2023.3',
'urllib3==1.26.15',
'zipp==3.15.0',
],
    keywords=['protein', 'folding', 'reinforcement'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ]
)
