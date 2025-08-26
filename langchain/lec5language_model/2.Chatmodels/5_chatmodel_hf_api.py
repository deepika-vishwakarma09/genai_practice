from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()
print("Your Hugging Face Token is:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))

llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-xl",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)