# Data Description

This project utilizes historical energy demand data structured as a
time series, along with optional exogenous data sources for analysis.

---

## Core Dataset (Energy Demand)

### Description
The primary dataset contains energy consumption values indexed by time.

### Key Columns
- `timestamp` : Datetime of observation
- `value` : Energy demand (target variable)
- `series_id` : Identifier for time series segments

### Usage
This dataset is used across:
- Training
- Validation
- Final test inference

---

## Exogenous Dataset (ERA5 Weather Data)

### Description
ERA5 is a global atmospheric reanalysis dataset produced by ECMWF,
providing hourly weather variables derived from numerical weather models
and observations.

### Candidate Variables
- Air temperature
- Wind speed
- Humidity
- Surface pressure

### Intended Purpose
ERA5 was considered to:
- Enhance model sensitivity to weather-driven demand fluctuations
- Improve forecasting during extreme climate conditions

---

## Exclusion from Final Pipeline

After evaluation, ERA5 features were not included in the final model due to:
- Limited incremental performance gains
- Increased feature engineering complexity
- Alignment constraints between weather data and demand timestamps

This decision was made to prioritize:
- Simplicity
- Consistency across datasets
- Production feasibility

---

## Data Integrity Notes
- No target leakage is introduced
- Temporal order is preserved
- Feature engineering is performed in a dedicated notebook

