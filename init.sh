#!/usr/bin/env bash
nix-shell --option extra-substituters 'https://cache.nixos-cuda.org' --option extra-trusted-public-keys 'cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M=' hub-shell.nix
