# ╔═══════════════════════════════════════════════╗
# ║  📄 app.py
# ║  👤 alikoaik
# ║  🔗 github.com/alikoaik
# ║  📅 01/04/2026
# ╚═══════════════════════════════════════════════╝

from dotenv import load_dotenv
import os
import sys

load_dotenv()

from src.loader import loadFile
from src.split import spliting
from src.embedding import embedding
from src.qa import ask_question as ask_question_ollama
from src.qa_openai import ask_question as ask_question_openai

def main() :
    # Choose LLM provider
    print("Choose your LLM provider:")
    print("1. Ollama (Local)")
    print("2. OpenAI (API Key)")

    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            ask_question = ask_question_ollama
            print("Using Ollama (Local LLM)")
            break
        elif choice == "2":
            ask_question = ask_question_openai
            print("Using OpenAI API")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    while True:
        name = input("\nEnter the name of the file in the data folder: ").strip()
        if not name:
            print("File name cannot be empty. Please try again.")
            continue
        try:
            document = loadFile(name)
            break
        except FileNotFoundError as e:
            print(f"Error: {e}. Please try again.")
        except Exception as e:
            print(f"Error loading file: {e}. Please try again.")

    try:
        chunks = spliting(document)
    except Exception as e:
        print(f"Error splitting document: {e}")
        sys.exit(1)

    try:
        vectorstore = embedding(chunks)
    except Exception as e:
        print(f"Error creating vector store: {e}")
        sys.exit(1)

    print("\n✓ Document loaded and ready for questions!")

    # Q&A loop
    while True:
        question = input("\nAsk a question (or 'quit' to exit): ").strip()
        if not question:
            print("Please enter a question.")
            continue
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        os.system('clear' if os.name != 'nt' else 'cls')
        print(f"Question: {question}")
        try:
            answer = ask_question(vectorstore, question)
            print(f"\nAnswer: {answer}")
        except ConnectionError as e:
            print(f"Connection error: {e}")
        except Exception as e:
            print(f"Error getting answer: {e}")

if __name__ == "__main__" :
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)
