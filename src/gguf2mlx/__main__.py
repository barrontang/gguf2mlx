import sys

try:
    from .gguf2mlx import main
except ImportError as error:  # missing runtime dependency: report, don't traceback
    print(f"❌ {error}", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
