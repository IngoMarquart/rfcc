import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="rfcc",
    version="1.0.0",
    description="RFCC: Random Forest Consensus Clustering for Regression and Classification",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/IngoMarquart/RFCC",
    author="Ingo Marquart",
    author_email="ingo.marquart@esmt.org",
    license="MIT",
    packages=["rfcc"],
    include_package_data=False,
    install_requires=["pandas", "scipy"],
)
