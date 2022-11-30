# Animal sound classification

Project to classify animal species from their sound.

## Structure

```
Classifier
├─	data/
   ├─ raw
	   sounds, labels and metadata
├─ figures/
      exported figures
├─	lab/
		quickly test new features, plot some data...
├─ references/
      papers, repositories in relation with our task
├─ scripts/
      launch common tasks (train, tests)
├─ src/
   ├─ data/
      data manipulation code
   ├─ dataset/
      scripts generating intermediary datasets from raw data
   ├─ features/
      compute typical audio features
   ├─ networks/
      models definition
   config.py : project configuration
   train.py : training script
├─ tests/
      unitary tests
├─ weights/
      exported weights
```

## Workflow

1. Data processing
   1. Load data
   2. Features extraction
   3. Normalization
   4. (Augmentation)
2. Model definition
3. Model training

## 