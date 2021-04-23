# utilities
## Repository containing utilities for data science program

## How to use this repo

### Choose whether you want a stable release or the most current features:
* Stable
    1. Download the release you wish to use.
    2. Unzip to the project directory where you will be using `utilities.py`
* Current (Recommended)
    1. Clone this repository
    2. If you do not have your own `env.py`:
        1. Remove ".template" from the `env.py.template` file
        2. Set up your `env.py` with the correct settings for your environment
    3. If you have an `env.py`, add the following to your file:
        * `import sys`
        * `sys.path.append("/path/to/utilities")`
            * Replace "/path/to/utilities" with the absolute path to the repository
* Import to your notebook or module using:  `import utilities` or `from utilities import...`
