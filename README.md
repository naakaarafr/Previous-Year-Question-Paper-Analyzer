# Previous Year Question Paper Analyzer

AI‑powered PDF & PPTX Question Paper Analyzer with Chat Interface

---

## 📖 Overview

**PYQ Analyzer** lets you upload PDF and PowerPoint files (`.pdf`, `.pptx`, `.ppt`), extract text (with OCR support for scanned docs), and build semantic vector embeddings. Powered by Google’s Gemini via LangChain and FAISS, it provides an interactive chat interface so you can ask questions and get context‑aware answers drawn directly from your documents. Built with Streamlit for a sleek, user‑friendly UI.

---

## ⚙️ Features

- **Multi‑format support**: PDF (text & scanned) and PowerPoint (`.pptx`/`.ppt`)
- **OCR integration**: `pytesseract` + Poppler for image‑based PDFs
- **Text splitting & embeddings**: Chunk large docs and encode with GoogleGenerativeAI
- **Vector search**: FAISS index for fast semantic retrieval
- **Chat interface**: Natural‑language Q&A powered by LangChain + Gemini
- **Session state**: Persistent upload, processing status, and conversation history
- **Dark‑theme UI**: Custom CSS for immersive experience

---

## 📦 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/naakaarafr/Previous-Year-Question-Paper-Analyzer.git
   cd Previous-Year-Question-Paper-Analyzer
   ```

2. **Create & activate virtualenv**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Poppler & Tesseract**

   * **Ubuntu/Debian**

     ```bash
     sudo apt-get install poppler-utils tesseract-ocr
     ```
   * **macOS (Homebrew)**

     ```bash
     brew install poppler tesseract
     ```

---

## 🔧 Configuration

1. Create a `.env` file in the project root:

   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   ```

2. (Optional) Adjust chunk size/overlap or model settings in `app.py`.

---

## 🚀 Usage

1. **Run the app**

   ```bash
   streamlit run app.py
   ```

2. **Upload Papers**

   * Navigate to **“Upload Papers”** tab
   * Drag & drop one or more `.pdf` / `.pptx` files
   * Click **“Process Question Papers”** to extract text & build the FAISS index

3. **Ask Questions**

   * Switch to **“Ask Questions”** tab
   * Type any query about your uploaded documents
   * Receive AI‑generated answers with citations

4. **Clear Conversation**

   * Click **“Clear Conversation”** to reset chat history

---

## 📂 Project Structure

```
.
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── faiss_index/         # Saved FAISS embeddings (generated at runtime)
└── README.md            # This file
```

---

## 🛠 How It Works

1. **Extraction**

   * **PDF**: Uses PyPDF2; falls back to OCR (pdf2image + pytesseract) if text is insufficient
   * **PPTX**: Reads slide shapes with python‑pptx

2. **Chunking & Embedding**

   * Splits text into overlapping chunks via LangChain’s `RecursiveCharacterTextSplitter`
   * Encodes chunks using `GoogleGenerativeAIEmbeddings`

3. **Indexing & Search**

   * Builds a FAISS vector store (`faiss_index`) for fast similarity search
   * On query, retrieves top‑k relevant chunks

4. **Q\&A Chain**

   * Feeds retrieved chunks and user query into LangChain’s QA chain
   * Uses `ChatGoogleGenerativeAI` (Gemini) for answer generation

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/XYZ`)
3. Commit your changes (`git commit -m "feat: add XYZ"`)
4. Push to your branch (`git push origin feature/XYZ`)
5. Open a Pull Request

Please ensure all new code is covered by tests and follows PEP8.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📫 Support

For bugs or feature requests, please open an issue on GitHub or contact \[[divvyanshkudesiaa1@gmail.com](divvyanshkudesiaa1@gmail.com)].
