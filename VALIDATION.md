# Release validation

- Default split: 500 training, 50 validation and 100 test scenes; 200 TX samples per scene. Output/cache paths are isolated under 500train_tx200.
- Core model, training, dataset, loss and preprocessing modules match the selected workspace sources apart from normalized line endings and final newlines; hashes are in SOURCE_PROVENANCE.json.
- Python 3.8 syntax parsing passed. Python 3.9+ is recommended for the dependencies.
- 16 synthetic checks passed locally with PyTorch 2.8.0+cpu, including completeness/disjointness, trajectory continuity, clean targets under noisy inputs, pseudo-label isolation, structural gradients, cache, metrics, compatibility dispatch, and training/validation/checkpoint-resume/final-test integration.
- A dedicated release test checks requested counts and retained sampling/loss defaults.

Command on 2026-10-30:

    python -B -m pytest -q --assert=plain -p no:cacheprovider --basetemp <workspace-test-directory>
    16 passed in 16.29s

Pytest third-party plugins were disabled and OMP/MKL/OpenBLAS threads set to 1. The sandbox default temporary folder initially denied atomic file renames; rerunning with an explicit workspace test directory passed without modifying source algorithms.

No real-data training, full 500-map experiment, GPU throughput or published accuracy is claimed for this new configuration.
