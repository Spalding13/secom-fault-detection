# SECOM Data Files

The raw SECOM dataset files are not committed to this repository. They are kept out of Git because they are external raw data files and can be downloaded from the official UCI Machine Learning Repository.

The empty `data/raw/.gitkeep` file is intentional. It preserves the expected folder structure while allowing the raw data files to remain ignored.

## Source

Official dataset page:

https://archive.ics.uci.edu/dataset/179/secom

Direct official download used by this project:

https://archive.ics.uci.edu/static/public/179/secom.zip

## Required Files

After downloading, `data/raw/` should contain:

- `secom.data`
- `secom_labels.data`
- `secom.names`

## Download Command

From the repository root:

```powershell
python scripts/download_data.py
```

Use `--force` only if the local raw files should be replaced:

```powershell
python scripts/download_data.py --force
```

## Expected Validation

The download script validates:

- feature matrix shape: `(1567, 590)`
- label count: `1567`
- target labels: only `-1` and `1`

In SECOM, `1` means Fail and `-1` means Pass.
