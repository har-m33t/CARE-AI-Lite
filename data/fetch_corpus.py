#!/usr/bin/env python3
"""Thin shim — kept so the documented command still works:

    python data/fetch_corpus.py --email you@example.com

The real, tested implementation now lives in `carelite.corpus.fetch` (the DOI
manifest, the Unpaywall -> NCBI -> PMC resolution chain, the %PDF guard,
dedup, idempotency, and the `_manual_needed.csv` report). Prefer running it
directly going forward:

    python -m carelite.corpus.fetch --email you@example.com
"""

from __future__ import annotations

import sys

from carelite.corpus.fetch import main

if __name__ == "__main__":
    sys.exit(main())
