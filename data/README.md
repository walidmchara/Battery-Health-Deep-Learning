# Battery benchmark data

NASA PCoE and CALCE CS2 measurements are included as cycle-level CSV tables. The source provenance and conversion metadata are maintained in [processed/manifest.json](processed/manifest.json).

## Dataset inventory

- **NASA:** 12 cells (B0005–B0018 benchmark plus B0029–B0032 and B0053–B0056 groups), in [processed/nasa/](processed/nasa/).
- **CALCE:** CS2_35, CS2_36, CS2_37 and CS2_38, in [processed/calce/](processed/calce/).

## Research protocol

Battery-level separation is essential: B0005/B0006/B0007 form the standard development group and B0018 is held out for evaluation. SOH is based on a capacity ratio; same-cycle capacity and capacity-derived target columns must not enter the model's predictors.

The original training pipeline `src/train.py` and `configs/nasa.yaml` was designed for NASA MATLAB inputs. The committed CSVs support cycle-level analysis, but they are not a drop-in replacement for that pipeline without adapting the data loader. The required original `.mat` traces can be extracted locally using the documented converter; see [../docs/REAL_DATA_SETUP.md](../docs/REAL_DATA_SETUP.md).

Sources: [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and [CALCE/UMD](https://calce.umd.edu/data).
