import sys

from .cli import main

try:
    sys.exit(main())
except KeyboardInterrupt:
    print("\nstopped.", file=sys.stderr)
    sys.exit(130)
except EOFError:
    print(file=sys.stderr)
    sys.exit(130)
