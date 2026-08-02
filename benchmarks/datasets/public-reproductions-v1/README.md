# Jacobian public-reproductions-v1

This Harbor dataset contains independently verifiable public mathematical cases.
Each case is a self-contained task directly under this dataset directory;
public provenance does not
make Oracle solutions or verifier code agent-visible.

`suite.toml` owns stable dataset policy; authoritative `members/*.toml` records
own membership and task contract metadata. Immutable snapshot locks freeze
Harbor task digests for intentional evaluations. The Oracle job runs every
migrated case:

```sh
make harbor-oracle DATASET=public-reproductions-v1
```

These tasks establish deterministic public-case correctness. They are not a
held-out performance split and their rewards must not be compared with the
workflow, diagnostic, performance, provider, or example datasets.
