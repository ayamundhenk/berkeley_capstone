# RF-Based Drone Signal Detection Capstone

## Module 20: Initial Report and Exploratory Data Analysis

This capstone asks whether a machine-learning model can distinguish drone-associated radio-frequency (RF) signals from background noise while limiting false alarms.

The original Module 16 proposal named CardRF as the planned dataset. Because CardRF was not reliably accessible, the project changed to the public [Noisy Drone RF Signal Classification dataset](https://www.kaggle.com/datasets/sgluege/noisy-drone-rf-signal-classification). The research question remains the same.

## Notebook

The [Module 20 Jupyter notebook](notebooks/rf_drone_detection_capstone.executed.ipynb) contains:

- The problem statement and dataset description
- Data-quality checks for missing and duplicate values
- Exploratory data analysis (EDA)
- Categorical and continuous-variable visualizations
- SNR outlier analysis
- Feature engineering
- Train, validation, and test split descriptions
- A logistic regression baseline model
- Initial results and limitations

## Data

The full source contains 98,705 labeled RF examples across seven original classes. A manageable subset of 7,000 examples was used for this initial analysis:

- 1,000 background/noise examples
- 6,000 drone-associated examples
- SNR values from -20 dB to 30 dB

The original `Noise` class was mapped to binary label `0`. Six drone/controller classes were mapped to binary label `1`.

Large raw and processed signal files are not stored in GitHub. The repository includes the notebook, code, charts, and compact result files needed to review the Module 20 analysis.

## Data cleaning and EDA

The initial analysis checks required fields for missing values, checks sample IDs for duplicates, and removes invalid rows from an analysis copy if necessary. It also uses the interquartile range (IQR) method to identify unusual SNR values.

Extreme SNR observations are not automatically removed because the dataset intentionally includes weak signals. These difficult observations are relevant to the research question.

The EDA examines:

- Original RF class counts
- Binary target balance
- Class balance across data splits
- SNR distributions
- SNR differences by class
- Prediction performance at different signal-quality levels

## Feature engineering

Raw I/Q signal measurements were converted into normalized log-spectrograms. Ten numerical spectral summary features were extracted for the baseline model. For interpretation, the EDA also creates signal-quality groups and a variable indicating whether each prediction was correct.

## Baseline model

Module 20 uses logistic regression as the baseline classification model. Recall is the main evaluation metric because it measures the proportion of actual drone-associated signals detected. False-positive rate is also monitored because a useful detector should not repeatedly flag background noise.

The probability threshold was selected with validation data to maximize recall while keeping validation false-positive rate at or below 5%. The held-out test set was used only for the initial final check of this baseline.

## Initial results

The logistic regression baseline produced the following held-out test results:

- Recall: 64.5%
- Precision: 99.1%
- F1 score: 78.1%
- False-positive rate: 3.5%
- PR-AUC: 98.4%

The main EDA finding is that performance depends strongly on signal-to-noise ratio. Recall is low for weak signals at negative SNR values and improves as signal quality increases.

These are initial Module 20 findings, not a production-ready result. The model still misses many weak drone-associated signals.

## Repository organization

```text
.
├── notebooks/
│   ├── rf_drone_detection_capstone.ipynb
│   └── rf_drone_detection_capstone.executed.ipynb
├── reports/
│   └── artifacts/noisy_drone_rf/
├── rf_drone_detection/
├── scripts/
├── tests/
├── README.md
└── requirements.txt
```

## Work remaining after Module 20

Later capstone work may improve low-SNR performance, test additional modeling approaches, and prepare the final non-technical report. Those later steps are outside the scope of this Module 20 submission.
