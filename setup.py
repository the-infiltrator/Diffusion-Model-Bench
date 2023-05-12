def install_dependencies():
    import os
    import subprocess
    import sys
    import importlib
    from subprocess import getoutput

    def is_installed(package):
        spec = importlib.util.find_spec(package)
        return spec is not None

    def install_with_url(url):
        subprocess.check_call([sys.executable, "-m", "pip", "install", url])

    # Read the packages from requirements.txt
    with open('requirements.txt', 'r') as f:
        packages = f.read().splitlines()

    # Install dependencies from requirements.txt
    for package in packages:
        if not is_installed(package):
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package])

    print("Dependency check passed!")
