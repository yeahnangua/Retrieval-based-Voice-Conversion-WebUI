#!/bin/sh

if [ "$(uname)" = "Darwin" ]; then
  # macOS specific env:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
elif [ "$(uname)" != "Linux" ]; then
  echo "Unsupported operating system."
  exit 1
fi

if [ -d ".venv" ]; then
  echo "Activate venv..."
  . .venv/bin/activate
else
  echo "Create venv..."
  requirements_file="requirements.txt"

  # Check if Python 3.10 is installed
  if ! command -v python3.10 >/dev/null 2>&1; then
    echo "Python 3.10 not found. Attempting to install..."
    if [ "$(uname)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
      brew install python@3.10
    elif [ "$(uname)" = "Linux" ] && command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install python3.10 python3.10-venv
    else
      echo "Please install Python 3.10 manually."
      exit 1
    fi
  fi

  python3.10 -m venv .venv
  . .venv/bin/activate

  # Check if required packages are installed and install them if not
  if [ -f "${requirements_file}" ]; then
    installed_packages=$(python3.10 -m pip freeze)
    while IFS= read -r package; do
      expr "${package}" : "^#.*" > /dev/null && continue
      package_name=$(echo "${package}" | sed 's/[<>=!].*//')
      if ! echo "${installed_packages}" | grep -q "${package_name}"; then
        echo "${package_name} not found. Attempting to install..."
        python3.10 -m pip install --upgrade "${package}"
      fi
    done < "${requirements_file}"
  else
    echo "${requirements_file} not found. Please ensure the requirements file with required packages exists."
    exit 1
  fi
fi

# Download models
chmod +x tools/dlmodels.sh
./tools/dlmodels.sh

if [ $? -ne 0 ]; then
  exit 1
fi

# Run the main script
python3.10 infer-web.py --pycmd python3.10
