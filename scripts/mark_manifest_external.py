from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Mark every row in a manifest as an external evaluation split.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    manifest["split"] = "external"
    output = args.output or args.manifest
    manifest.to_csv(output, index=False)
    print(f"Wrote external manifest to {output}")


if __name__ == "__main__":
    main()
