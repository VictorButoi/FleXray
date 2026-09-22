"""Allow ``python -m fxr.protocols`` to run the ``fxr-protocol`` command."""

from .cli import main

raise SystemExit(main())
