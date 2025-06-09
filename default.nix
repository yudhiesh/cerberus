{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    just
    act
    uv
  ];

  shellHook = ''
    echo "🚀 Development environment loaded!"
    echo "Available tools:"
    echo "  - just: $(just --version)"
    echo "  - act: $(act --version)"
    echo "  - uv: $(uv --version)"
    echo ""
  '';
}