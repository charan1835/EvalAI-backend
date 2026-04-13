---
title: EvalAi
emoji: 🚀
colorFrom: blue
colorTo: blue
sdk: docker
---

# 🧠 EvalAI Backend - NLP Engine

This is the core intelligence engine for EvalAI, built with **FastAPI** and **Hugging Face Transformers**.

## 🚀 Live Demo
The API is currently hosted on Hugging Face Spaces:
[https://huggingface.co/spaces/chimbilicharan/EvalAi](https://huggingface.co/spaces/chimbilicharan/EvalAi)

## 🛠 Features
- **Semantic Evaluation**: Uses `all-MiniLM-L6-v2` for cross-encoding similarity.
- **Generative Insights**: Integrated with Gemini 1.5 Flash for deep critique.
- **Asynchronous Processing**: Non-blocking handles for high-concurrency requests.
- **Dockerized**: Containerized for seamless deployment.

## 🚦 Endpoints
- `POST /evaluate`: Core answer analysis.
- `GET /quiz/generate`: AI-generated MCQ quizzes.
- `POST /auth/request-otp`: Secure OTP dispatch.

## 📦 Local Development
1. `pip install -r requirements.txt`
2. `python main.py`