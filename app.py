# ╔═══════════════════════════════════════════════╗
# ║  📄 app.py
# ║  👤 alikoaik
# ║  🔗 github.com/alikoaik
# ║  📅 01/04/2026
# ╚═══════════════════════════════════════════════╝

from dotenv import load_dotenv
import os

# Load environment variables first
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
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        ask_question = ask_question_ollama
        print("Using Ollama (Local LLM)")
    elif choice == "2":
        ask_question = ask_question_openai
        print("Using OpenAI API")
    else:
        print("Invalid choice. Defaulting to Ollama.")
        ask_question = ask_question_ollama

    name = input("\nEnter the name of the file exist in data folder\n")
    document = loadFile(name)
    chunks = spliting(document)
    vectorstore = embedding(chunks)

    print("\n✓ Document loaded and ready for questions!")

    # Q&A loop
    while True:
        question = input("\nAsk a question (or 'quit' to exit): ")
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        os.system('clear' if os.name != 'nt' else 'cls')
        print(f"Question: {question}")
        answer = ask_question(vectorstore, question)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__" :
    main()
