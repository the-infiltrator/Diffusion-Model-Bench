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

    # Additional installation for xformers based on GPU
    gpu_info = getoutput('nvidia-smi')
    if 'T4' in gpu_info and not is_installed('xformers'):
        install_with_url(
            "https://github.com/daswer123/stable-diffusion-colab/raw/main/xformers%20prebuild/T4/python39/xformers-0.0.14.dev0-cp39-cp39-linux_x86_64.whl")
    elif 'P100' in gpu_info and not is_installed('xformers'):
        install_with_url(
            "https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/P100/xformers-0.0.13.dev0-py3-none-any.whl")
    elif 'V100' in gpu_info and not is_installed('xformers'):
        install_with_url(
            "https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/V100/xformers-0.0.13.dev0-py3-none-any.whl")
    elif 'A100' in gpu_info and not is_installed('xformers'):
        install_with_url(
            "https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/A100/xformers-0.0.13.dev0-py3-none-any.whl")

    print("Dependency check passed!")
