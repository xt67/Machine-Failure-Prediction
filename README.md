# Machine Failure Prediction

## Overview
This repository contains a machine learning model that predicts equipment failures using sensor data. The model analyzes readings from temperature, air quality, pressure, and other sensors to identify high-risk conditions before breakdowns occur.

## Dataset
The dataset includes 174 machine observations with the following features:
- `footfall`: Number of people/objects passing by the machine
- `tempMode`: Temperature mode/setting
- `AQ`: Air quality index
- `USS`: Ultrasonic sensor data
- `CS`: Current sensor readings
- `VOC`: Volatile organic compounds level
- `RP`: Rotational position/RPM
- `IP`: Input pressure
- `Temperature`: Operating temperature
- `fail`: Binary indicator (1=failure, 0=no failure)

## Key Findings
- VOC (Volatile Organic Compounds) levels strongly indicate potential failures
- Machines with VOC levels ≥4 have significantly higher failure rates
- The combination of high VOC with low ultrasonic sensor readings represents a high-risk state
- The Random Forest classifier achieves 85% prediction accuracy

## Model Features
- Data preprocessing and exploratory analysis
- Correlation analysis between sensor readings and failures
- Random Forest classifier for prediction
- Feature importance ranking
- Visualization of key relationships

## Usage
```python
# Example usage
python machine_failure_prediction.py
```

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Future Work
- Implement real-time monitoring system
- Add time-series analysis for early warning detection
- Test additional machine learning algorithms
