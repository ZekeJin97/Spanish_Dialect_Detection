# 🗣️ Spanish Dialect Detection

This project classifies Spanish dialects from both **text and audio** using deep learning and traditional ML approaches. It supports regional classification for Spain, Mexico, Chile, Argentina, and the Dominican Republic.

## 🌍 Goal

To build a robust, speaker-disjoint, multilingual dialect classification system using open-source data and models.

## 📦 Dataset

- Source: [Hugging Face Spanish Dialects](https://huggingface.co/datasets/rjnieto/spanish-dialects/tree/main)
- Includes `.parquet` files with aligned audio + transcription and speaker metadata

## 🔊 Audio Classification

- Model: Fine-tuned `Wav2Vec2` (wav2vec2-large-xlsr-53-spanish)
- Preprocessing:
  - Dynamic range normalization
  - Silence removal
  - Truncation
- Tools: `Wav2Vec2Processor`, Hugging Face `Trainer`
- Accuracy: **~92.1%**

## 📝 Text Classification

- Preprocessing: TF-IDF vectorization of transcripts
- Models compared:
  - Logistic Regression
  - SVM (linear kernel)
  - Fine-tuned multilingual BERT
- Result: Lexical cues outperformed semantic embeddings (BERT) on this task

## 🧪 Evaluation

- **Speaker-disjoint train/test split** to reduce leakage and simulate real-world unseen speakers
- Measured accuracy and F1-score across both modalities


## 💡 Future Work

- Combine text + audio embeddings for multimodal classification
- Scale to additional dialects (e.g., Colombia, Venezuela, Peru)
