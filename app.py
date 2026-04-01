# ╔═══════════════════════════════════════════════╗
# ║  📄 app.py
# ║  👤 alikoaik
# ║  🔗 github.com/alikoaik
# ║  📅 01/04/2026
# ╚═══════════════════════════════════════════════╝

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from src.loader import loadFile
from src.split import spliting
from src.embedding import embedding
from src.qa import ask_question

def main() :
    name = input("Enter the name of the file exist in data folder\n")
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

        answer = ask_question(vectorstore, question)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__" :
    main()
