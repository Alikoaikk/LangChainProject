from src.loader import loadFile

def main() :
    name = input("Enter the name of the file exist in data folder\n")
    document = loadFile(name)

if __name__ == "__main__" :
    main()
