import os
from dotenv import load_dotenv
import openai
import azure.cognitiveservices.speech as speechsdk
import speech_recognition as sr
from langchain_core.runnables import RunnableSequence
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
from upstash_redis import Redis
from together import Together
from datetime import datetime
import threading
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
import queue
import re

# =========================
# Environment Setup
# =========================

def load_env_variables():
    """
    Load required environment variables from a .env file.
    Raises ValueError if any required variable is missing.
    """
    load_dotenv()
    required_vars = [
        "OPENAI_API_KEY", "UPSTASH_URL", "UPSTASH_TOKEN", "AZURE_SPEECH_KEY",
        "AZURE_SPEECH_REGION", "TOGETHER_API_KEY", "REDIS_URL", "REDIS_TOKEN"
    ]
    env_vars = {}
    for key in required_vars:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Missing required environment variable: {key}")
        env_vars[key] = value
    return env_vars

# =========================
# External Service Initializers
# =========================

def initialize_together_client(api_key):
    """Initialize the Together API client."""
    return Together(api_key=api_key)

def initialize_openai_client(api_key):
    """Set the OpenAI API key for the openai library."""
    openai.api_key = api_key

def initialize_redis_client(env_vars):
    """Initialize the Upstash Redis client."""
    return Redis(url=env_vars["REDIS_URL"], token=env_vars["REDIS_TOKEN"])

# =========================
# LLM Model, Prompt, and Memory Setup
# =========================

def initialize_model_and_prompt(env_vars):
    """
    Set up the OpenAI chat model, prompt template, and conversation memory using Upstash Redis.
    Returns the model, prompt, and memory objects.
    """
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
    prompt = ChatPromptTemplate.from_messages([
        ("assistant", "You are a helpful AI assistant called Xio"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    history = UpstashRedisChatMessageHistory(
        url=env_vars["UPSTASH_URL"],
        token=env_vars["UPSTASH_TOKEN"],
        session_id="chat1",
        ttl=0
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, chat_memory=history)
    chain = RunnableSequence(prompt | model)
    return model, prompt, memory

# =========================
# User Data Utilities
# =========================

def get_user_name(redis_client):
    """Retrieve the user's name from Redis, if set."""
    return redis_client.get("user_name")

# =========================
# Chat Processing Logic
# =========================

def process_chat(agent_executor, user_input, chat_history, together_client, redis_client, callback=None, interrupted_ref=None):
    """
    Run both OpenAI and Llama LLMs in parallel and return the first response via callback.
    If callback is None, returns the response synchronously (legacy mode).
    interrupted_ref: a function returning True if the user has interrupted (for cancellation support).
    """
    result_queue = queue.Queue()
    done_event = threading.Event()

    def call_openai():
        try:
            openai_response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })["output"]
        except Exception as e:
            openai_response = f"Error with OpenAI: {e}"
        # Only put result if not interrupted and not already done
        if not done_event.is_set() and (not interrupted_ref or not interrupted_ref()):
            result_queue.put(openai_response)
            done_event.set()

    def call_llama():
        # Prepare message history for Llama
        history_messages = [{"role": "system", "content": "You are Xio, a helpful AI assistant."}]
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                continue
            history_messages.append({"role": role, "content": msg.content})
        llama_messages = history_messages + [{"role": "user", "content": user_input}]
        try:
            llama_response = together_client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                messages=llama_messages,
                max_tokens=150,
                temperature=0,
                stream=False
            )
            if hasattr(llama_response, 'choices') and len(llama_response.choices) > 0:
                first_choice = llama_response.choices[0]
                if hasattr(first_choice, 'message'):
                    llama_response_text = first_choice.message.content.strip()
                elif hasattr(first_choice, 'text'):
                    llama_response_text = first_choice.text.strip()
                else:
                    llama_response_text = "Error: Unexpected response structure from LLaMA model."
            else:
                llama_response_text = "Error: No valid response from LLaMA model."
        except Exception as e:
            llama_response_text = f"Error with Together LLaMA: {e}"
        # Only put result if not interrupted and not already done
        if not done_event.is_set() and (not interrupted_ref or not interrupted_ref()):
            result_queue.put(llama_response_text)
            done_event.set()

    # Start both LLM calls in parallel
    t1 = threading.Thread(target=call_openai)
    t2 = threading.Thread(target=call_llama)
    t1.start()
    t2.start()

    if callback:
        def wait_and_callback():
            response = result_queue.get()
            if not interrupted_ref or not interrupted_ref():
                chat_history.append(AIMessage(content=response))
                callback(response)
        threading.Thread(target=wait_and_callback).start()
        return None
    else:
        response = result_queue.get()
        if not interrupted_ref or not interrupted_ref():
            chat_history.append(AIMessage(content=response))
            return response

# =========================
# User Name Storage Utility
# =========================

def store_user_name(user_input, chat_history, redis_client):
    """
    If the user says 'my name is ...', store their name in Redis and acknowledge.
    """
    if "my name is" in user_input.lower():
        name = user_input.split("my name is")[-1].strip()
        redis_client.set("user_name", name)
        chat_history.append(AIMessage(content=f"Nice to meet you, {name}!"))
        return name
    return None

# =========================
# Agent Setup
# =========================

def initialize_custom_agent(model, prompt, memory):
    """
    Create a LangChain agent with search tool and memory.
    """
    search = TavilySearchResults()
    tools = [search]
    agent = create_openai_functions_agent(llm=model, prompt=prompt, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)
    return agent_executor

# =========================
# Speech Synthesis and Recognition
# =========================

def speak(text, env_vars):
    """
    Synthesize speech from text using Azure Cognitive Services.
    """
    speech_config = speechsdk.SpeechConfig(subscription=env_vars["AZURE_SPEECH_KEY"],
                                           region=env_vars["AZURE_SPEECH_REGION"])
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    synthesizer.speak_text_async(text).get()

def listen():
    """
    Listen to the user's voice and return recognized text using Google Speech Recognition.
    Returns None if recognition fails.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, Xio did not understand that.")
            return None
        except sr.RequestError:
            print("Sorry, Xio's speech service is down.")
            return None

# =========================
# Time Formatting Utility
# =========================

def convert_time_to_text(time_string):
    """
    Convert a time string (HH:MM:SS) to a human-friendly spoken format.
    """
    hours, minutes, _ = time_string.split(':')
    hours = int(hours)
    minutes = int(minutes)

    period = "AM"
    if hours >= 12:
        period = "PM"
        if hours > 12:
            hours -= 12
    elif hours == 0:
        hours = 12

    minute_text = f"{minutes:02d}".replace('00', "o'clock")

    if minutes == 0:
        time_text = f"It's {hours} o'clock {period}"
    else:
        time_text = f"It's {hours} {minute_text} {period}"

    return time_text

# =========================
# Main Kivy Application
# =========================

class AIApp(App):
    """
    Main Kivy application for the Xio assistant.
    Handles UI, user input (text/voice), chat history, and LLM/speech orchestration.
    """
    def build(self):
        # Initialize environment and all services
        self.env_vars = load_env_variables()
        initialize_openai_client(self.env_vars["OPENAI_API_KEY"])
        self.redis_client = initialize_redis_client(self.env_vars)
        model, prompt, memory = initialize_model_and_prompt(self.env_vars)
        self.agent_executor = initialize_custom_agent(model, prompt, memory)
        self.together_client = initialize_together_client(self.env_vars["TOGETHER_API_KEY"])

        self.chat_history = []
        self.input_mode = None  # Will be set by popup
        self.layout = BoxLayout(orientation='vertical')

        # --- Chat history area with scroll ---
        self.scrollview = ScrollView(size_hint=(1, 0.8))
        self.chat_box = BoxLayout(orientation='vertical', size_hint_y=None)
        self.chat_box.bind(minimum_height=self.chat_box.setter('height'))
        self.scrollview.add_widget(self.chat_box)
        self.layout.add_widget(self.scrollview)
        # --- End chat history area ---

        # Placeholder for input widgets
        self.input_widget = None
        self.send_button = None
        self.speak_button = None

        # Add Change Mode button
        self.change_mode_button = Button(text="Change Mode", size_hint=(1, 0.08))
        self.change_mode_button.bind(on_press=self.change_mode)
        self.layout.add_widget(self.change_mode_button)

        # For speech interruption and LLM cancellation
        self.is_speaking = False
        self.synthesizer = None
        self.speech_thread = None
        self.interrupted = False

        # Show mode selection popup on startup
        Clock.schedule_once(lambda dt: self.show_mode_selection_popup(), 0)

        return self.layout

    def show_mode_selection_popup(self):
        """
        Show a popup for the user to choose between text and voice input modes.
        """
        content = GridLayout(cols=1, spacing=10, padding=10)
        text_btn = Button(text="Text with Xio", size_hint=(1, None), height=40)
        voice_btn = Button(text="Speak to Xio", size_hint=(1, None), height=40)
        content.add_widget(text_btn)
        content.add_widget(voice_btn)
        popup = Popup(title="Choose Input Mode", content=content, size_hint=(0.7, 0.4), auto_dismiss=False)

        def select_text_mode(instance):
            self.input_mode = "text"
            popup.dismiss()
            self.setup_input_widgets()

        def select_voice_mode(instance):
            self.input_mode = "voice"
            popup.dismiss()
            self.setup_input_widgets()

        text_btn.bind(on_press=select_text_mode)
        voice_btn.bind(on_press=select_voice_mode)
        popup.open()

    def setup_input_widgets(self):
        """
        Set up the input widgets (text or voice) based on the selected mode.
        """
        # Remove old input widgets if any
        if self.input_widget:
            self.layout.remove_widget(self.input_widget)
        if self.send_button:
            self.layout.remove_widget(self.send_button)
        if self.speak_button:
            self.layout.remove_widget(self.speak_button)

        if self.input_mode == "text":
            self.input_widget = TextInput(size_hint=(1, 0.1), multiline=False)
            self.layout.add_widget(self.input_widget)
            self.send_button = Button(text="Send to Xio", size_hint=(1, 0.1))
            self.send_button.bind(on_press=self.send_message)
            self.layout.add_widget(self.send_button)
        elif self.input_mode == "voice":
            self.speak_button = Button(text="Speak to Xio", size_hint=(1, 0.2))
            self.speak_button.bind(on_press=self.speak_message)
            self.layout.add_widget(self.speak_button)
        # else: do nothing

    def update_chat_history(self, message):
        """
        Add a message to the chat history UI and scroll to the bottom.
        """
        label = Label(text=message, size_hint_y=None, halign='left', valign='top')
        label.bind(texture_size=lambda instance, value: setattr(label, 'height', value[1]))
        label.text_size = (self.scrollview.width * 0.95, None)
        self.chat_box.add_widget(label)
        # Schedule scroll to bottom
        def scroll_to_bottom(dt):
            self.scrollview.scroll_y = 0
        Clock.schedule_once(scroll_to_bottom, 0.1)

    def send_message(self, instance):
        """
        Handle sending a message from the text input.
        Supports app exit if user says 'bye' in any context.
        """
        user_input = self.input_widget.text.strip()
        if not user_input:
            return
        if "bye" in user_input.strip().lower():
            App.get_running_app().stop()
            return
        self.interrupted = False
        self.chat_history.append(HumanMessage(content=user_input))
        self.update_chat_history(f"You: {user_input}")

        store_user_name(user_input, self.chat_history, self.redis_client)

        if "time" in user_input.lower():
            current_time = datetime.now().strftime('%I:%M:%S %p')
            response = convert_time_to_text(current_time)
            self.update_chat_history(f"Xio: {response}")
            threading.Thread(target=self.speak_and_auto_listen, args=(response,)).start()
        else:
            self.update_chat_history("Xio is thinking...")
            def on_response(response):
                if not self.interrupted:
                    Clock.schedule_once(lambda dt: self.update_chat_history(f"Xio: {response}"))
                    threading.Thread(target=self.speak_and_auto_listen, args=(response,)).start()
            process_chat(self.agent_executor, user_input, self.chat_history, self.together_client, self.redis_client, callback=on_response, interrupted_ref=lambda: self.interrupted)

        self.input_widget.text = ""

    def speak_message(self, instance):
        """
        Handle the Speak to Xio button: if Xio is speaking, interrupt and start listening; otherwise, just start listening.
        """
        if self.is_speaking and self.synthesizer:
            self.interrupted = True
            try:
                self.synthesizer.stop_speaking_async().get()
            except Exception:
                pass
            self.is_speaking = False
            self.synthesizer = None
            # Wait a moment to ensure speech is stopped
            Clock.schedule_once(lambda dt: self._start_listening(), 0.1)
        else:
            self.interrupted = True
            self._start_listening()

    def _start_listening(self):
        """
        Start listening for user voice input and process the result.
        """
        self.update_chat_history("Listening for your message...")
        def recognize_and_send():
            user_input = listen()
            if user_input:
                Clock.schedule_once(lambda dt: self.process_voice_input(user_input), 0)
            else:
                Clock.schedule_once(lambda dt: self.update_chat_history("Sorry, Xio did not catch that. Please try again."), 0)
        threading.Thread(target=recognize_and_send).start()

    def process_voice_input(self, user_input):
        """
        Handle a recognized voice input. Supports app exit if user says 'bye' in any context.
        """
        self.interrupted = False
        if "bye" in user_input.strip().lower():
            App.get_running_app().stop()
            return
        self.chat_history.append(HumanMessage(content=user_input))
        self.update_chat_history(f"You: {user_input}")

        store_user_name(user_input, self.chat_history, self.redis_client)

        if "time" in user_input.lower():
            current_time = datetime.now().strftime('%I:%M:%S %p')
            response = convert_time_to_text(current_time)
            self.update_chat_history(f"Xio: {response}")
            threading.Thread(target=self.speak_and_auto_listen, args=(response,)).start()
        else:
            self.update_chat_history("Xio is thinking...")
            def on_response(response):
                if not self.interrupted:
                    Clock.schedule_once(lambda dt: self.update_chat_history(f"Xio: {response}"))
                    threading.Thread(target=self.speak_and_auto_listen, args=(response,)).start()
            process_chat(self.agent_executor, user_input, self.chat_history, self.together_client, self.redis_client, callback=on_response, interrupted_ref=lambda: self.interrupted)

    def clean_text_for_speech(self, text):
        """
        Clean up text for speech synthesis by removing code blocks, markdown, and common symbols.
        Also expands common abbreviations to their full forms for natural speech.
        """
        # Remove triple backtick code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        # Remove inline code
        text = re.sub(r'`[^`]+`', '', text)
        # Expand common abbreviations (do this before symbol removal)
        abbr_map = {
            r'\be\.g\.[,\.:;]?': 'for example',
            r'\bi\.e\.[,\.:;]?': 'that is',
            r'\betc\.[,\.:;]?': 'and so on',
            r'\bvs\.[,\.:;]?': 'versus',
            r'\bapprox\.[,\.:;]?': 'approximately',
            r'\binfo\b': 'information',
            r'\basap\b': 'as soon as possible',
            r'\bbtw\b': 'by the way',
        }
        for abbr, full in abbr_map.items():
            text = re.sub(abbr, full, text, flags=re.IGNORECASE)
        # Remove markdown formatting symbols
        text = re.sub(r'[\*_#>\[\]{}\(\)\-~]', '', text)
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def speak_and_auto_listen(self, text):
        """
        Speak the given text using Azure TTS, and in voice mode, auto-listen after speaking (unless interrupted).
        """
        def do_speak():
            import azure.cognitiveservices.speech as speechsdk
            self.is_speaking = True
            speech_config = speechsdk.SpeechConfig(subscription=self.env_vars["AZURE_SPEECH_KEY"],
                                                   region=self.env_vars["AZURE_SPEECH_REGION"])
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
            self.synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            clean_text = self.clean_text_for_speech(text)
            self.synthesizer.speak_text_async(clean_text).get()
            self.is_speaking = False
            self.synthesizer = None
            # If in voice mode, auto-listen after speaking, but only if not interrupted
            if self.input_mode == "voice" and not self.interrupted:
                Clock.schedule_once(lambda dt: self.speak_message(None), 0)
        self.speech_thread = threading.Thread(target=do_speak)
        self.speech_thread.start()

    def change_mode(self, instance):
        """
        Allow the user to return to the input mode selection popup.
        Removes current input widgets and shows the mode selection popup again.
        """
        # Remove input widgets if present
        if self.input_widget:
            self.layout.remove_widget(self.input_widget)
            self.input_widget = None
        if self.send_button:
            self.layout.remove_widget(self.send_button)
            self.send_button = None
        if self.speak_button:
            self.layout.remove_widget(self.speak_button)
            self.speak_button = None
        self.input_mode = None
        self.show_mode_selection_popup()

if __name__ == "__main__":
    AIApp().run()
