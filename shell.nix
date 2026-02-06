{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = [
    # pkgs.python310
    # pkgs.python313
    # pkgs.python313Packages.llvmlite
    pkgs.pkg-config
    pkgs.ffmpeg_6
    pkgs.libsndfile
    pkgs.gcc
    pkgs.llvmPackages_20.llvm
  ];

  shellHook = ''
    export PKG_CONFIG_PATH="${pkgs.ffmpeg_6.dev}/lib/pkgconfig:${pkgs.libsndfile.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"
    export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
    export LLVM_CONFIG="${pkgs.llvmPackages_20.llvm}/bin/llvm-config"
  '';
}
