# Near-Earth Objects Classification and Risk Assessment (NEOCLARA)

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
  - Machine Learning: Data preprocessing for classification models 

    
---
### Reference

  - [Near-Earth-Object | Wikipedia ](https://en.wikipedia.org/wiki/Near-Earth_object)

  - [NEOs Basics | NASA ](https://cneos.jpl.nasa.gov/about/neo_groups.html)

  - [Asteroid 2024 YR4 no longer poses significant impact risk | European Space Agency ](https://www.esa.int/Space_Safety/Planetary_Defence/Asteroid_2024_YR4_no_longer_poses_significant_impact_risk)

  - [ Cliff's delta calculator | University of California San Francisco library](  https://search.library.ucsf.edu/discovery/fulldisplay?docid=cdi_scielo_journals_S1657_92672011000200018&context=PC&vid=01UCS_SAF:UCSF&lang=en&search_scope=DN_and_CI&adaptor=Primo%20Central&query=null,,1,AND&facet=citing,exact,cdi_FETCH-LOGICAL-c446t-50082f28956bdaaf4a39983b6260dc118caa28f2cb93d6e6be5b64416d46a5003&offset=20)

 - [Mann-Whitney U test | Data Lab](https://datatab.net/tutorial/mann-whitney-u-test)

  - [ Effect size for Mann-Whitney U test | Dr. Todd Grande](https://www.youtube.com/watch?v=ILD5Jvmokig)


  - Introduction to Modern Statistics | Mine Çetinkaya-Rundel and Johanna Hardin
