{ }:

let
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-26.05.tar.gz") {
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  };

  python = pkgs.python3.withPackages (ps: with ps; [
    jupyterlab
    ipykernel
    optuna
    matplotlib
    numpy
    pandas
    torch-bin
    gymnasium
    pip
    ruff
    scikit-learn
  ]);

in
pkgs.mkShell {
  buildInputs = [
    python
    pkgs.geos
    pkgs.gdal
    pkgs.proj
  ];

  packages = [
    python
  ];

  shellHook = ''
    # Make project imports resolve correctly.
    export PYTHONPATH="$PWD/src:$PYTHONPATH"

    # Expose the host NVIDIA driver to PyTorch.
    # Do NOT add /usr/lib/x86_64-linux-gnu wholesale to LD_LIBRARY_PATH,
    # since that can override Nix's glibc.
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libcuda.so.1''${LD_PRELOAD:+:$LD_PRELOAD}"

    python -c "import torch; print('CUDA WORKS: ', torch.cuda.is_available())"

    echo "Run: python -m next_state_predictor.main --help"
    echo "Run: jupyter notebook"
  '';
}
