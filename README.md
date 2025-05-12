# FPL Assistant - ML Service & Data Pipeline

This repository contains the Python-based machine learning service and data pipeline for the FPL Assistant project. It is responsible for fetching Fantasy Premier League (FPL) data, performing ETL (Extract, Transform, Load) operations, training predictive models, and generating weekly player performance predictions.

**Part of the larger FPL Assistant Project:**
* **Main Project Repository (MERN App):** [https://github.com/honeykeys/a2b-mernapp]
* **Live Frontend Application:** [fplandml.netlify.app]

---

## Table of Contents

* [Project Overview](#project-overview)
* [Architecture & Pipeline Flow](#architecture--pipeline-flow)
* [Key Features](#key-features)
* [Technology Stack](#technology-stack)
* [Local Development Setup](#local-development-setup)
    * [Prerequisites](#prerequisites)
    * [Cloning](#cloning)
    * [Virtual Environment](#virtual-environment)
    * [Installing Dependencies](#installing-dependencies)
    * [Environment Variables (`.env.example`)](#environment-variables)
    * [AWS Credentials Setup (for S3)](#aws-credentials-setup)
* [Running the Scripts Locally](#running-the-scripts-locally)
    * [Running the Full ETL & Training Pipeline (`run_etl.py`)](#running-the-full-etl--training-pipeline)
    * [Running Individual Model Training Scripts](#running-individual-model-training-scripts)
    * [Running the Prediction Generation Script (`run_scheduled_etl.py`)](#running-the-prediction-generation-script)
* [Data Sources](#data-sources)
* [Model Details & Performance (Brief)](#model-details--performance)
* [Deployment (Conceptual)](#deployment-conceptual)
* [Known Limitations & Future Work](#known-limitations--future-work)
* [Author](#author)

---

## Project Overview

The `a2b-ml-service` provides the predictive analytics backbone for the FPL Assistant. It automates the process of:
1.  Ingesting historical and current FPL data.
2.  Cleaning and transforming this data.
3.  Engineering features relevant for predicting FPL player performance.
4.  Training machine learning models (e.g., for predicting next gameweek points and price changes).
5.  Generating weekly predictions and storing them for use by the main application backend.

---

## Architecture & Pipeline Flow

This service operates as an offline batch processing pipeline:

1.  **Data Ingestion:** Raw data is fetched from sources like the [Vaastav FPL GitHub repository](https://github.com/vaastav/Fantasy-Premier-League/) and potentially the live FPL API.
2.  **ETL & Feature Engineering:** Python scripts using Pandas clean the data, merge datasets (player data, fixture data, gameweek results), and engineer features (e.g., lagged performance, rolling averages, fixture difficulty).
3.  **Model Training (Offline):** Machine learning models (currently Decision Tree and RandomForestRegressor via Scikit-learn) are trained on historical data. Trained models are saved as `.joblib` files.
4.  **Prediction Generation:** A scheduled script (`run_scheduled_etl.py`) performs the ETL for the latest available data, loads the pre-trained models from AWS S3, generates predictions for the upcoming gameweek, and saves the output as JSON.
5.  **Output Storage:** Prediction JSON files and trained model artifacts are stored in an AWS S3 bucket.

---

## Key Features

* Automated data fetching and processing for multiple FPL seasons.
* Feature engineering tailored for FPL player performance.
* Training and evaluation of regression models for points and price change predictions.
* Generation of structured JSON output for predictions, ready for API consumption.
* Integration with AWS S3 for model and prediction storage.
* Designed for containerization with Docker and scheduled execution (e.g., via AWS Fargate).

---

## Technology Stack

* **Language:** Python 3.x
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Model Persistence:** Joblib
* **HTTP Requests:** Requests
* **AWS Integration:** Boto3 (for S3)
* **Data Serialization:** JSON, Parquet (for intermediate processed data)
* **Containerization:** Docker

---