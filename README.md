# Data Automation - AG News Processing

## Author: Nimantha Karunanayake

---

## Introduction

This script processes the AG News dataset using PySpark to generate two Parquet files:

1. A fixed word count table for the words: `["president", "the", "Asia"]`.
2. A table counting all unique words from the `description` column.

---

## GitHub Repository

Check the source code and updates on GitHub:

[GitHub Repository](https://github.com/nixx49/data-automation)

---

## Steps to Run the Application

### Step 1: Pull the Docker Container

Run the following command to pull the latest Docker image:

```sh
docker pull nixx49/data-automation:latest
```

### Step 2: Run Combined Command (Shell Script)

To generate both Parquet files at once, execute:

```sh
docker run nixx49/data-automation:latest
```

### Step 3: Generate Word Count for Specific Words

To generate a file containing counts for the words `["president", "the", "Asia"]`, run:

```sh
docker run nixx49/data-automation:latest process_data --cfg config/cfg.yaml --dataset news --dirout ztmp/data
```

### Step 4: Generate Word Count for All Unique Words

To generate a file counting all unique words from the dataset, use:

```sh
docker run nixx49/data-automation:latest process_data_all --cfg config/cfg.yaml --dataset news --dirout ztmp/data
```

### Step 5: Run Tests

To test the application, execute:

```sh
docker run nixx49/data-automation:latest run_tests --cfg config/cfg.yaml --dataset news --dirout ztmp/data
```

---

