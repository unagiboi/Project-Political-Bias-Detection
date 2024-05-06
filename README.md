# Political Bias Detection in Media

## Table of Contents
- [Introduction](#introduction)
  - [Problem Description](#problem-description)
  - [Input and Output](#input-and-output)
  - [Contributors](#contributors)
- [Methodology](#methodology)
  - [Dataset](#dataset)
  - [Preprocessing](#preprocessing)
  - [Initial BERT Model Analysis](#initial-bert-model-analysis)
  - [Dataset Adjustments](#dataset-adjustments)
  - [Model Improvement Strategies](#model-improvement-strategies)
  - [Model Adaptability and Real-World Application](#model-adaptability-and-real-world-application)
- [Results and Discussion](#results-and-discussion)
- [Conclusion and Future Directions](#conclusion-and-future-directions)

## Introduction

### Problem Description
Public perception is heavily influenced by the media, and understanding the political bias present in news articles is crucial for informed decision-making. This project aimed to develop a model capable of predicting the political leaning (left, center, or right) of an article based on its content and title.

### Input and Output
**Input:** The model receives textual input, including the content and title of an article.
**Output:** The model generates a classification label indicating the political leaning of the article (left-leaning, center, or right-leaning). Additionally, it provides a confidence value for the predicted label.

### Contributors
- Jeongin Bae
- Kyle Hoffmeyer
- Dhravid Kumar

## Methodology

### Dataset
Our dataset comes from allsides.com, encompassing a total of 37,554 texts categorized into three distinct political leanings: left, center, and right.

### Preprocessing
The dataset underwent a thorough cleaning process involving multiple steps. This involved removing HTML tags and URLs to eliminate noise. Text standardization steps included removing punctuation, converting text to lowercase, and eliminating stopwords. Vocabulary reduction techniques, such as stemming and lemmatization, were applied to further refine the text.

## Initial BERT Model Analysis

The table below illustrates the test accuracy and GPU training time at various sample sizes, highlighting the trade-off between accuracy and training time:

| Samples | Test Accuracy | GPU Training Time (min) |
|---------|---------------|--------------------------|
| 200     | 0.427         | 0.32                     |
| 1000    | 0.642         | 6                        |
| 3000    | 0.712         | 17                       |
| 6000    | 0.735         | 30                       |
| 10,000  | 0.728         | 55                       |

Despite the reasonable accuracy achieved, we recognized the need for further exploration through hyperparameter tuning and optimization methods, with the objective of surpassing the 0.8 test accuracy threshold.

## Dataset Adjustments

### Challenge

The initial dataset, comprising 37,554 articles, posed challenges related to training time and computational resources. The baseline model consistently achieved approximately 0.74 accuracy from the initial project proposal through the project progress report. However, articles exceeding 512 tokens presented difficulties for BERT-based models, given their input limitations.

### Solution

To address the limitations of the initial dataset, we made significant adjustments. First, we implemented the cutting of all data with fewer than 512 tokens, leading to an increase in the baseline model's performance from 0.735 to 0.799. This adjustment aimed to enhance the model's ability to handle longer articles consistently. Our new baseline thus became 0.799 rather than 0.735.


### Model Improvement Strategies
1. **Hyperparameter Tuning:**
   - Accuracy achieved: 0.824

2. **Naive Bayes Integration:**
   - Accuracy achieved: 0.428

3. **LSTM Model Integration:**
   - Accuracy achieved: 0.843

4. **T5 Summarization Model Integration:**
   - Accuracy achieved: 0.414

### Model Adaptability and Real-World Application
- **Summarization Technique:**
  - Recognizing the need for model adaptability to handle longer articles, we incorporated a summarization model to summarize the original content before passing it through our preprocessing functions. The model: `Falconsai/text_summarization` was downloaded from Hugging Face, and it is a fine-tuned 60.5 million parameter T5 model. It was employed to generate concise summaries before tokenization, enabling the model to retain crucial information within the token limit. This adaptation aimed to address potential information loss in lengthy texts and add support to articles over 512 tokens in length.

- **Web Application Development:**
  - With a focus on real-world applicability, we wanted to develop a web application. The app accepts any article, loads the trained model, and outputs a political bias label along with an associated accuracy. The Flask framework facilitated the implementation of this API-driven web app, offering users a practical tool to assess the bias in articles.



## Results and Discussion
The achieved test accuracy of 0.84 marked a significant milestone in the project. Despite extensive experimentation with different model configurations, statistical significance in accuracy improvement proved elusive. 

## Conclusion and Future Directions
In conclusion, the project successfully addressed the core objective of political bias detection in media. The integration of advanced models, hyperparameter tuning, and real-world adaptability measures significantly improved the model's performance. 

