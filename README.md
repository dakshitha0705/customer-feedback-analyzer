# Customer Feedback Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/q4mxjuh77r/badge.svg)](https://ask.thili.ai/b/q4mxjuh77r)

This Customer Feedback Analyzer helps retail managers quickly identify trends and prioritize improvements by automatically categorizing customer feedback comments. It leverages a fine-tuned GPT-2 model to classify comments into predefined categories like "product issue" or "staff praise", showcasing an end-to-end inference pipeline.

## What you are building

**Goal.** You will build a tool that automatically categorizes customer feedback comments using a fine-tuned GPT-2 model, demonstrating the fine-tuning logic and an inference pipeline to classify new comments.

**Done when.** The project is complete when you have a Streamlit application that allows a user to input customer feedback comments and displays the predicted category (e.g., "product issue", "staff praise") for each comment.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Define 3-5 feedback categories and create a small dataset of example comments with labels.**  
      Hint: Define your categories and create a small training dataset with example comments and their corresponding labels.
- [ ] **2. Extend GPT-2 tokenizer and model with new category tokens.**  
      Hint: Modify the GPT-2 tokenizer and model to recognize and generate your new category tokens, similar to how special tokens were added for legal clauses.
- [ ] **3. Fine-tune the model using the Hugging Face Trainer.**  
      Hint: Use the Hugging Face Trainer to fine-tune your model on your custom dataset, following the pattern from the GPT-2 fine-tuning drill.
- [ ] **4. Implement an inference function to classify new feedback.**  
      Hint: Develop a function that takes a new customer comment and uses your fine-tuned model to predict and return its category.
- [ ] **5. Build a simple Streamlit app to input comments and show classifications.**  
      Hint: Build a simple web interface using Streamlit to accept user input and display the classification results from your inference function.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Expand the dataset with more diverse examples and additional feedback categories.
- Implement a feature to process multiple comments at once (batch inference).
- Add basic reporting in the Streamlit app, such as a bar chart showing the distribution of classified categories.
