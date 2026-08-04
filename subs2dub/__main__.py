import sys

from .cli import main


def run() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        print("\nstopped.", file=sys.stderr)
        return 130
    except EOFError:
        print(file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(run())
