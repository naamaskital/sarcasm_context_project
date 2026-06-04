# Sarcasm Detection with Context

This project studies whether conversational context helps detect sarcasm in Reddit comments.

## Research Question
Does adding the previous Reddit message improve sarcasm classification compared to using the reply alone?

## Current Experiments
1. TF-IDF + Logistic Regression:
   - context only
   - comment only
   - context + comment

2. Qwen zero-shot prompting:
   - comment only
   - context + comment

3. Qwen few-shot and model size comparison:
   - Qwen2.5 0.5B
   - Qwen2.5 1.5B

## Main Preliminary Finding
Adding context improves sarcasm detection in the baseline experiment.

## Repository Structure
- src/ - experiment scripts
- data_backup/ - small sample dataset backup
- reports_backup/ - saved preliminary results
- reports/ - generated reports, ignored by git
