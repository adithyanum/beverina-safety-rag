# 🧠 Beverina — AI-Powered Ingredient Safety Analyzer

End-to-end LLM system for analyzing beverage ingredient safety using **RAG, FAISS, and QLoRA**, deployed with fully local GGUF inference.

---

## 🔍 Overview

Beverina analyzes beverages and ingredients using a **fine-tuned Mistral-7B model + Retrieval-Augmented Generation (RAG)**. It provides ingredient breakdowns, health impact insights, and regulatory-backed safety information (WHO, FSSAI, FDA), while minimizing hallucinations through controlled retrieval and prompt design.

---

## 🖥️ Demo

### 🧾 UI

![UI](docs/ui.png)

### ⚡ Ingredient Breakdown

![Output](docs/output_sting.png)

### 🧠 Health Analysis

![Output](docs/output_frooti.png)

### 💡 Reasoned Response

![Output](docs/output_water.png)

---

## ⚙️ Highlights

* RAG pipeline using **FAISS + MiniLM embeddings**
* Fine-tuned **Mistral-7B (QLoRA)**
* Custom ingredient + product knowledge base
* Hallucination control via retrieval + system prompt
* **GGUF conversion + 4-bit quantization** for local inference
* Fully offline (no external APIs)

---

## ⚠️ Note

Model weights are not included due to size. Reproduce via training and conversion scripts.

---
