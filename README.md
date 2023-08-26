---
title: Test
emoji: 🚀
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 3.18.0
app_file: app.py
pinned: false
license: unknown
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Gestalt
Gestalt is an AI system designed to enhance the mathematical problem-solving capabilities of large language models like GPT-3 and GPT-4. It was created as part of a research project exploring new approaches to augmenting LLMs for complex problem solving tasks.

## Overview
Generative large language models (GLLMs) like GPT-3 and GPT-4 have shown impressive capabilities in generating coherent text. However, they still struggle with reliably executing precise computations and retrieving factual knowledge, which are critical for mathematical problem solving.

Gestalt aims to overcome these limitations by integrating GLLMs with external deterministic tools like WolframAlpha and Python. This allows the creative language generation capabilities of LLMs to be combined with the precision of computational engines.

Key features of Gestalt:

- Built using the LangChain framework for easily integrating LLMs with external resources
- GPT-3 is used as the core LLM to analyze problems and develop solution strategies
- WolframAlpha queries are generated to retrieve necessary facts and execute computations
- Python code is generated for additional complex calculations
- The system provides step-by-step explanations of its reasoning and actions, increasing transparency

## Results
Gestalt was evaluated on a subset of mathematical problems from the MATH benchmark dataset. It achieved 59% accuracy compared to 53.9% for GPT-4 alone on the same problems. This demonstrates the value of Gestalt's integrated approach for strengthening LLM math problem solving abilities.

## Usage
A demo version of Gestalt is deployed on Hugging Face and can be accessed via the link below

The system will return the answer along with an explanation of its reasoning and computations.

## Reference
For more details on Gestalt, refer to the full paper:
