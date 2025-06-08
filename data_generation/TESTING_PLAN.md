# Testing Plan for Cerberus Data Generation Pipeline

## Overview
This document outlines the testing strategy for the Cerberus data generation pipeline. Each component is tested for functionality, error handling, and integration with other components.

## Test Environment Setup
- Python 3.12+
- Required packages:
  ```bash
  uv add pytest pytest-mock pytest-cov mock typer
  ```

## Test Data Structure
```
tests/
├── conftest.py           # Common fixtures
├── data/                 # Test data
│   ├── sample.jsonl     # Sample input data
│   ├── judged.jsonl     # Sample judged data
│   └── argilla_export/  # Sample Argilla export
├── mocks/               # Mock responses
│   ├── llm_responses.py # Mock LLM responses
│   └── api_responses.py # Mock API responses
└── test_*.py           # Test files
```

## Component Testing Plans

### 1. main.py
**Complexity: LOW**
**Status: Not Started**

#### Test Objectives
- [ ] Test app initialization
- [ ] Test command registration
- [ ] Test help messages
- [ ] Test error handling

#### Required Mocks
- None (simple app composition)

#### Test Cases
1. App Initialization
   - [ ] Verify all subcommands are registered
   - [ ] Verify help messages are correct

2. Command Registration
   - [ ] Verify all commands are accessible
   - [ ] Verify command help messages

3. Error Handling
   - [ ] Test invalid command handling
   - [ ] Test help message display

### 2. deduplicate.py
**Complexity: MEDIUM**
**Status: Not Started**

#### Test Objectives
- [ ] Test deduplication logic
- [ ] Test file I/O operations
- [ ] Test metrics collection
- [ ] Test error handling

#### Required Mocks
- [ ] Mock SemHash responses
- [ ] Mock file operations

#### Test Cases
1. Deduplication Logic
   - [ ] Test exact duplicate removal
   - [ ] Test semantic duplicate detection
   - [ ] Test label preservation

2. File Operations
   - [ ] Test input file reading
   - [ ] Test output file writing
   - [ ] Test directory creation

3. Metrics Collection
   - [ ] Test duplicate ratio calculation
   - [ ] Test exact duplicate ratio
   - [ ] Test similarity metrics

4. Error Handling
   - [ ] Test invalid input file
   - [ ] Test missing columns
   - [ ] Test permission errors

### 3. annotate.py
**Complexity: MEDIUM**
**Status: Not Started**

#### Test Objectives
- [ ] Test Argilla integration
- [ ] Test data transformation
- [ ] Test file I/O operations
- [ ] Test error handling

#### Required Mocks
- [ ] Mock Argilla API responses
- [ ] Mock file operations

#### Test Cases
1. Argilla Integration
   - [ ] Test dataset creation
   - [ ] Test record upload
   - [ ] Test dataset download

2. Data Transformation
   - [ ] Test label mapping
   - [ ] Test data cleaning
   - [ ] Test format conversion

3. File Operations
   - [ ] Test input file reading
   - [ ] Test output file writing
   - [ ] Test directory creation

4. Error Handling
   - [ ] Test API errors
   - [ ] Test invalid data
   - [ ] Test file errors

### 4. dataset_preprocess.py
**Complexity: MEDIUM**
**Status: Not Started**

#### Test Objectives
- [ ] Test dataset merging
- [ ] Test train/test splitting
- [ ] Test HuggingFace upload
- [ ] Test error handling

#### Required Mocks
- [ ] Mock Argilla API responses
- [ ] Mock HuggingFace API
- [ ] Mock file operations

#### Test Cases
1. Dataset Merging
   - [ ] Test data combination
   - [ ] Test label resolution
   - [ ] Test duplicate handling

2. Train/Test Splitting
   - [ ] Test split ratios
   - [ ] Test stratification
   - [ ] Test data distribution

3. HuggingFace Upload
   - [ ] Test dataset creation
   - [ ] Test split upload
   - [ ] Test metadata handling

4. Error Handling
   - [ ] Test API errors
   - [ ] Test invalid data
   - [ ] Test file errors

### 5. generate.py
**Complexity: HIGH**
**Status: Not Started**

#### Test Objectives
- [ ] Test prompt generation
- [ ] Test LLM integration
- [ ] Test pipeline execution
- [ ] Test error handling

#### Required Mocks
- [ ] Mock OpenRouterLLM responses
- [ ] Mock distilabel pipeline
- [ ] Mock file operations

#### Test Cases
1. Prompt Generation
   - [ ] Test safe prompt generation
   - [ ] Test unsafe prompt generation
   - [ ] Test prompt formatting

2. LLM Integration
   - [ ] Test model selection
   - [ ] Test response parsing
   - [ ] Test error handling

3. Pipeline Execution
   - [ ] Test batch processing
   - [ ] Test caching
   - [ ] Test output generation

4. Error Handling
   - [ ] Test API errors
   - [ ] Test invalid responses
   - [ ] Test file errors

### 6. evaluate.py
**Complexity: HIGH**
**Status: Not Started**

#### Test Objectives
- [ ] Test evaluation logic
- [ ] Test metrics calculation
- [ ] Test visualization
- [ ] Test error handling

#### Required Mocks
- [ ] Mock OpenRouterLLM responses
- [ ] Mock DeepEval responses
- [ ] Mock file operations

#### Test Cases
1. Evaluation Logic
   - [ ] Test label prediction
   - [ ] Test score calculation
   - [ ] Test threshold handling

2. Metrics Calculation
   - [ ] Test accuracy calculation
   - [ ] Test precision/recall
   - [ ] Test F1 score

3. Visualization
   - [ ] Test confusion matrix
   - [ ] Test plot generation
   - [ ] Test file saving

4. Error Handling
   - [ ] Test API errors
   - [ ] Test invalid data
   - [ ] Test file errors

## Progress Tracking

### Current Status
- [ ] main.py
- [ ] deduplicate.py
- [ ] annotate.py
- [ ] dataset_preprocess.py
- [ ] generate.py
- [ ] evaluate.py

### Next Steps
1. Set up test environment
2. Create test data
3. Implement main.py tests
4. Progress through complexity levels
5. Review and refine

## Notes
- All external API calls must be mocked
- Test data should be minimal but representative
- Error cases should be comprehensive
- Tests should be isolated and independent
- Coverage should be maintained at >80%

## Dependencies
- pytest
- pytest-mock
- pytest-cov
- typer
- mock
- pandas
- numpy
- scikit-learn
- datasets
- argilla
- semhash
- distilabel
- deepeval 