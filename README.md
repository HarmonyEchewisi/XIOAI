# Xio AI Assistant

Xio is a conversational AI assistant with both text and voice interaction, built using Python, Kivy, LangChain, OpenAI, Together AI, and Azure Cognitive Services. It features persistent chat history, multi-model LLM support, and a modern, user-friendly interface.

## Features

- **Conversational UI**: Choose between text or voice input modes.
- **Voice Recognition & Speech Synthesis**: Speak to Xio and hear responses using Azure and Google APIs.
- **Multi-Model LLM**: Uses both OpenAI and Llama (Together AI); responds with the fastest model.
- **Persistent Chat History**: Stores conversation context in Upstash Redis.
- **Interruptible & Natural Flow**: Interrupt AI speech or response at any time by clicking 'Speak'.
- **Smart Exit**: Say or type "bye" (in any context) to close the app.
- **Professional, Modern UI**: Scrollable chat, responsive design, and clear UX.

## Setup & Installation

### 1. Clone the Repository
```sh
git clone <your-gitlab-repo-url>
cd <repo-folder>
```

### 2. Create and Activate a Virtual Environment
```sh
python -m venv .venv
source .venv/bin/activate  
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the project root with the following keys:
```
OPENAI_API_KEY=your_openai_key
UPSTASH_URL=your_upstash_redis_url
UPSTASH_TOKEN=your_upstash_redis_token
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=your_azure_region
TOGETHER_API_KEY=your_together_api_key
REDIS_URL=your_upstash_redis_url
REDIS_TOKEN=your_upstash_redis_token
```

> **Note:** You need valid API keys for OpenAI, Together AI, Azure Cognitive Services, and Upstash Redis.

### 5. Platform-Specific Notes
- **Windows:**
  - nstall `portaudio` via your package manager if you have issues with PyAudio.
- **Linux/macOS:**
  - Install `portaudio` via your package manager if you have issues with PyAudio.

## Usage

1. Run the app:
   ```sh
   python main.py
   ```
2. On startup, choose between text or voice mode.
3. Type or speak to Xio. In voice mode, you can interrupt the AI at any time by clicking 'Speak'.
4. To exit, say or type any phrase containing "bye" (e.g., "thank you, bye").

## Dependencies
- Python 3.8+
- Kivy
- openai
- together
- azure-cognitiveservices-speech
- speechrecognition
- pyaudio
- langchain
- upstash-redis
- python-dotenv

(See `requirements.txt` for full list)

## Screenshots
> _Add screenshots or a demo GIF here to showcase the UI and features._

## License
MIT License

## Contact
- **Author:** Harmony Echewisi
- **Email:** Afrozorro@protonmail.com
- **LinkedIn:** www.linkedin.com/in/harmonyechewisi

---

_This project demonstrates modern Python, AI, and UI engineering best practices. Feel free to fork, contribute, or reach out for collaboration!_ 