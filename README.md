## Kraken Architecture

![alt text](https://github.com/cognitivecomputations/kraken/blob/main/kraken.png?raw=true)

## Overview

The Kraken Architecture is a sophisticated machine learning framework designed for dynamic text generation tasks. It utilizes the Hugging Face transformers library to orchestrate multiple causal language models (CLMs) and intelligently route input through different models based on the context and content of the input text. The architecture is powered by a custom configuration class (CoEConfig) that facilitates the integration and management of various components such as tokenizers, models, and routing mechanisms.

## Features

Dynamic Model Routing: Uses a sequence classification model to route inputs to the most suitable language model based on the input's characteristics.
Multiple Language Models: Supports integration of various pre-trained causal language models, allowing for flexible, context-appropriate responses.
Customizable Templates: Includes support for input formatting using predefined templates, enhancing the model's adaptability to different conversational contexts.
Extensible Configuration: Leverages a custom configuration setup that can be easily extended and adapted for various use cases involving causal language modeling.
Requirements

Python 3.11+
transformers 4.40+
torch 2.2+
