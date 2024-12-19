from pydantic import BaseModel, Field
from LLMManager import LLMManager
from fastapi import FastAPI

app = FastAPI()

class Question(BaseModel):
    text: str = Field(examples=["What policies did de Gaulle impelement and when did he retire?"])
 
with open('document.txt', 'r') as file:
  data=file.read()
  documents=data.split(".")
 
@app.post("/api/question/")
async def get_question(question: Question):  
  llmManager=LLMManager(documents)
  user_prompt=llmManager.generate_user_prompt(question.text)
  response=llmManager.generate_question_response(user_prompt)
  return response