from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    print("Hello langGraph")
    print(os.getenv("TAVILY_API_KEY"))