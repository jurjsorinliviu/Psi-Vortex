# Final result record

This directory contains the verified numerical and deployment evidence produced by Ψ-Vortex.

- `raw_results/` contains the complete per-run metrics for all 28 experiment groups.
- `checkpoints/` contains every checkpoint referenced by the active result rows.
- `artifacts/` contains the guarded recurrent exports, Verilog-A source, compiled OSDI module, and ngspice evidence.
- `figures/` and `tables/` contain the generated numerical artifacts.
- `configs/final.json` is the full reproduction configuration.
- `environment/reference_environment.json` records the reference software environment.
- `manifests/file_inventory.csv` provides the byte size and SHA-256 digest of every file in this record.

Validate the record from the repository root with:

```bash
python tools/verify_public_record.py
```

The verifier checks group coverage, file hashes, checkpoint hashes, source-disjoint splits, recurrence-control acceptance, guarded exports, circuit-validation tolerances, and the declared aggregate counts.
