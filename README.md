# Near-Earth-Objects-EDA-and-Risk-Assessment

### Overview

This project analyzes Near-Earth Objects (NEOs) to distinguish Potentially Hazardous Objects (PHOs) from non-hazardous ones. Using exploratory data analysis (EDA), statistical testing, and machine learning, I was able assess key factors influencing asteroid hazard classification.

---
### Key Findings
* Class Imbalance: 88% of recorded NEOs are non-hazardous, highlighting a highly skewed dataset.
* Size and Brightness: Larger NEOs (low absolute magnitude) are more likely to be hazardous (power-law decay pattern).
* Velocity Matters: Higher-velocity hazardous NEOs have greater impact threats due to their speed.
* Proximity (Miss Distance) Effect: While closer NEOs pose a potential impact risk, statistical tests show miss distance alone is not a strong classifier for hazard assessment.
* Statistical Validation: Findings are supported by Mann-Whitney U tests and Cliff’s Delta effect size analysis.

### Methods & Tools
  - EDA & Visualization: Matplotlib, Seaborn, Plotly
  - Statistical Analysis: Mann-Whitney U Test, Cliff’s Delta Effect Size
  - Feature Importance: SHAP (SHapley Additive Explanations)
  - Machine Learning: Classification models (Optional for future improvements)

