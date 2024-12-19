## Overview
- This repository showcases very basic but important first steps on how to implement a Retrieval Augemented Generation (RAG) solution.
- In this example, we used text extracted from Wikipedia about Charles de Gaulle (https://en.wikipedia.org/wiki/Charles_de_Gaulle).
- The end goal is to the query the text for fairly correct responses.

## Approach
The approach here is:
 - ### Retrieval
  - We take contents in the document.txt file and then embed them.
  - We then embed the user query and then do a similary search between document and user query embedding to retrieve similar documents.
  - The similar documents are then used to create our context for the next stage.
 - ### Augmented
  - Here we construct a prompt (piece of string) with system instruction - instruction to guide the model how to generate the output. 
  - We then inject the user query and context which help to contrain the model to our local data. 
 - ### Generation
  - We get the contructed prompt and pass it to the model. The model then return fairly meaningful responses.
## Solution Set up
 - Clone repo
 - Create virtual environment to restore packages/dependencies
 - Run `pip freeze > requirements.txt` in the virtual environment to install dependencies
 - Download both `Phi-3-mini-128k-instruct.Q6_K.gguf` and `bge-small-en-v1.5-q4_k_m.gguf` into the models folder.
  - The Phi 3 model is used to generate text. 
  - The BGE model is used to create the document embeddings. 
 - Run `fastapi run main.py` to start server and browse to `http://127.0.0.1:8000/docs`

## ToDo:
 - Run in docker
 - Add script to check if models exist in models folder
  - If no models found, download them