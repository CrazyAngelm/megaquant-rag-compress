# Data

Benchmarks expect SQuAD v1.1 JSON files at repository root:

```text
squad_train_v1.1.json
squad_dev_v1.1.json
```

They are intentionally not committed to this repository. Download from the official SQuAD v1.1 release:

- https://rajpurkar.github.io/SQuAD-explorer/

Then place the files at the repo root before running the benchmark scripts.


## Download commands

From the repository root:

```bash
curl -L -o squad_train_v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
curl -L -o squad_dev_v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

Expected filenames exactly:

```text
squad_train_v1.1.json
squad_dev_v1.1.json
```

Some quick or selected scripts may use only one split, but keeping both files at repo root is the safest reproducible setup.
