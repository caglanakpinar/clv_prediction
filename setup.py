import setuptools
from setuptools import find_packages

install_requires_list = [
        "psutil",
        "numpy >= 1.18.1",
        "pandas >= 0.25.3",
        "scipy >= 1.4.1 ",
        "PyYAML",
        "schedule >= 0.6.0",
        "multiprocess >= 0.70.9",
        "google-cloud-bigquery",
        "mysql-connector-python",
        "plotly >=  4.5.0",
        "dash-html-components >= 1.0.2",
        "dash-core-components >=  1.8.0",
        "dash >= 1.9.0",
        "threaded >= 4.0.8",
        "pytest-shutil >= 1.7.0",
        "python-dateutil >= 2.8.1",
        "random2 >= 1.0.1",
        "psycopg2-binary",
        "argparse",
        "python-math",
        "statsmodels >= 0.12.1",
        "keras-tuner >= 1.0.2",
        "multiprocess >= 0.70.9",
        "tensorflow >= 2.2.0",
        "Keras >= 2.3.1"
    ]


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="clv_prediction",
    version="0.1.8",
    author="Caglan Akpinar",
    author_email="cakpinar23@gmail.com",
    description="clv prediction applying with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='CLV, Customer Lifetime Value, Lifetime Prediction',
    packages= find_packages(exclude='__pycache__'),
    py_modules=['clv', 'clv/docs'],
    install_requires=install_requires_list,
    url="https://github.com/caglanakpinar/clv_prediction",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
)
