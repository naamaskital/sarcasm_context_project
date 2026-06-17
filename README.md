# Sarcasm Detection with Conversational Context

## Overview

This project investigates whether conversational context improves sarcasm detection in Reddit comments.

The main goal is to compare sarcasm classification using only the target comment versus using the previous Reddit message as additional context. The project includes classical machine learning baselines, sentence embedding approaches, and LLM-based experiments with Qwen models.

The focus of the project is not only to improve accuracy, but also to understand when context helps, when it hurts, and how different models use conversational information.

## Research Question

Does adding the previous Reddit message improve sarcasm classification compared to using the reply alone?

## Dataset

The project uses a Reddit sarcasm dataset containing:

* `comment` - the target Reddit reply
* `parent_comment` - the previous message in the conversation
* `label` - sarcasm label

The experiments compare several input settings:

* **Comment only** - using only the target reply
* **Context only** - using only the previous Reddit message
* **Context + comment** - combining the previous message with the reply
* **Random context + comment** - using mismatched context as an ablation test

The random-context setup helps evaluate whether the model benefits from the true conversational context or simply from seeing more text.

## Methods

### Classical Machine Learning Baseline

Implemented TF-IDF feature extraction with Logistic Regression for sarcasm classification.

Tested multiple input configurations:

* Comment only
* Context only
* Context + comment
* Random context + comment

### Sentence Embedding Models

Used sentence embeddings to represent comments and conversational context in a dense semantic space.

Compared different representation strategies, including:

* Joint context-comment representation
* Separate embeddings for context and comment
* Feature combinations based on the relationship between the two texts

### LLM-Based Experiments

Experimented with Qwen models for sarcasm detection using prompting and fine-tuning.

The experiments included:

* Zero-shot prompting
* Few-shot prompting
* Model size comparison
* LoRA fine-tuning
* Context ablation experiments

## Experiments

The project includes the following main experimental settings:

| Setting                  | Description                                           |
| ------------------------ | ----------------------------------------------------- |
| Comment only             | Uses only the target Reddit comment                   |
| Context only             | Uses only the previous Reddit message                 |
| Context + comment        | Uses both the previous message and the target comment |
| Random context + comment | Uses an unrelated context as an ablation test         |

These settings allow a clearer analysis of whether context provides meaningful information for sarcasm detection.

## Key Findings

The results show that conversational context can help sarcasm detection, but the improvement depends on the model and the way context is represented.

Main insights:

* Context alone is usually not enough for reliable sarcasm detection.
* Adding the true context can improve performance compared to using random context.
* Some models benefit from context more than others.
* Random-context ablation is important because it shows whether the model is using meaningful conversational information or just benefiting from additional text.
* Qualitative error analysis shows that context can help in cases where sarcasm depends on contradiction, tone, or the previous message.

## Technologies Used

* Python
* Natural Language Processing
* Machine Learning
* TF-IDF
* Logistic Regression
* Sentence Embeddings
* Hugging Face
* Qwen
* LoRA Fine-Tuning
* scikit-learn
* PyTorch

## Repository Structure

```text
.
├── src/              # Experiment scripts and model training code
├── data_backup/      # Small dataset sample or backup files
├── reports_backup/   # Saved preliminary reports and experiment outputs
├── reports/          # Generated reports and evaluation results
└── README.md
```

## What I Learned

This project strengthened my experience with NLP research workflows, including dataset preparation, text representation, baseline modeling, LLM experimentation, ablation testing, and model evaluation.

It also helped me understand that improving a model is not only about achieving higher accuracy, but also about designing meaningful experiments that explain why a model succeeds or fails.

## Future Work

Possible future improvements include:

* Training larger LLMs on a larger dataset split
* Running additional fine-tuning epochs
* Adding more encoder-based baselines
* Performing deeper qualitative analysis of context-helped and context-hurt examples
* Improving prompt design for zero-shot and few-shot experiments
* Testing additional context representation methods
