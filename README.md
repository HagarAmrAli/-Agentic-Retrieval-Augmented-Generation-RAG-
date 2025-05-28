Here is a complete `README.md` file for your **Agentic RAG Travel Chatbot** project:

---

# 🧠 Agentic RAG Travel Chatbot

This project implements an **Agentic Retrieval-Augmented Generation (RAG) chatbot** for intelligent travel guidance using **CrewAI**, **Qdrant**, **OpenAI (GPT-4)**, and **Streamlit**.

---

## 🚀 Features

* 📄 **Data Ingestion**: Clean and chunk travel reviews from TripAdvisor dataset
* 🔍 **Vector Search**: Index and retrieve chunks using **Qdrant** and **OpenAI/SBERT embeddings**
* 🧠 **Multi-Agent Reasoning**: Use **CrewAI agents** for retrieval, summarization, and response composition
* 🖥️ **Interactive UI**: Real-time chatbot built using **Streamlit**
* 🌐 **Deployment**: Easily share your chatbot via **ngrok**

---

## 📁 Project Structure

```
agentic-travel-chatbot/
│
├── app.py                     # Streamlit app entry point
├── compose_response.py        # GPT-4 response generation logic
├── crew_config.py             # CrewAI agent definitions
├── data_ingest.ipynb          # Data cleaning and chunking notebook
├── index_and_retrieve.py      # Qdrant setup and search logic
├── processed_reviews.json     # Preprocessed chunks (optional)
├── tripadvisor_hotel_reviews.csv  # Raw TripAdvisor dataset
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/your/repo.git
   cd repo
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set your OpenAI API key**

   ```bash
   export OPENAI_API_KEY=your-api-key-here
   ```

---

## 🧠 Run the Model

### 1. Preprocess and Index Reviews

You can either use the notebook `data_ingest.ipynb` or call `load_qdrant_and_index()` function from `app.py`.

Ensure Qdrant is running locally or use `:memory:` for in-memory development.

### 2. Launch the Chatbot

```bash
streamlit run app.py
```

This will open a browser interface where you can interact with the travel assistant.

---

## ⚙️ How It Works

* **Query Input**: User enters a travel question
* **Vector Retrieval**: Top-k relevant review chunks fetched from Qdrant
* **CrewAI Agents**:

  * **Retriever**: Fetches relevant info
  * **Summarizer**: Reduces redundancy
  * **Composer**: Uses GPT-4 to generate final answer
* **Streamlit UI**: Displays result + retrieved context

---

## 📦 Dependencies

Key libraries used:

* `openai`
* `crewai`
* `streamlit`
* `qdrant-client`
* `pandas`, `re`, `tqdm`, `json`
* `sentence-transformers` (optional alternative to OpenAI embeddings)

Install manually if needed:

```bash
pip install openai crewai streamlit qdrant-client sentence-transformers pandas tqdm
```

---

## 🌐 Deployment (Ngrok)

To make your app accessible via a public link:

1. Install ngrok: [https://ngrok.com/download](https://ngrok.com/download)
2. Start the app:

   ```bash
   streamlit run app.py
   ```
3. In a new terminal, run:

   ```bash
   ngrok http 8501
   ```
4. Share the `https://xxxxx.ngrok.io` link

---

## 📈 Example Query

**User:** “What are the best hotels in Cairo?”
**Response:**

> "Top-rated hotels in Cairo include the Four Seasons Nile Plaza and the Kempinski Nile Hotel, praised for views, service, and location..."

---

## 🛠️ Future Improvements

* Add support for **multi-turn conversation**
* Integrate **LlamaIndex** or **LangChain**
* Deploy to **Streamlit Cloud** or **Hugging Face Spaces**
* Add **map-based hotel visualization**
* Support **PDF or itinerary upload**

---
