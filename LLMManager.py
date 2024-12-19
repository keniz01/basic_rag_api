import textwrap
from llama_cpp import Llama, LLAMA_POOLING_TYPE_NONE
import numpy as np

class LLMManager():

    def __init__(self, documents: list[str]):

        self.documents=documents

        self.chat_model = Llama(
            model_path="../models/Phi-3-mini-128k-instruct.Q6_K.gguf",
            n_ctx=2048,
            verbose=False,
            n_gpu_layers=0,
            logits_all=True,
            chat_format="chatml" 
        )   

        self.embed_model=Llama(
            model_path='../models/bge-small-en-v1.5-q4_k_m.gguf',
            embedding=True,
            verbose=False,
            pooling_type=LLAMA_POOLING_TYPE_NONE
        )

    def generate_document_embeddings(self):
        embeddings = [self.embed_model.embed(document)[1] for _, document in enumerate(self.documents)] 
        return embeddings 

    def generate_question_embeddings(self, question: str):
        embeddings = self.embed_model.embed(question)[1]
        return embeddings     
    
    def generate_question_response(self, user_prompt: str):
        '''
        Generates the LLM response to the user question.
        
        :param str user_prompt: The user question prompt.
        :return: the response string
        :rtype: str
        :raises Error: when an error is encountered
        '''
        response = self.chat_model(
            prompt=user_prompt,
            max_tokens=512,
            temperature=0,
            top_p=0.1
        )

        return response["choices"][0]["text"].strip()

    def generate_context(self, question: str) -> str:
        query_embeddings=self.generate_question_embeddings(question)
        doc_embeddings=self.generate_document_embeddings()
        similarities=np.dot(doc_embeddings, query_embeddings)

        top_3_idx=np.argsort(similarities, axis=0)[-3:][::1].tolist()
        most_similar_documents=[self.documents[idx] for idx in top_3_idx]

        context = ""
        for _, document in enumerate(most_similar_documents):
            wrapped_text=textwrap.fill(document,width=100)
            context += wrapped_text + "\n"

        return context
    
    def generate_user_prompt(self, question: str) -> str:
        '''
        Generates the LLM prompt using the user question and context.

        :param str question: The user question.
        :return: the response string
        :rtype: str
        :raises Error: when an error is encountered
        '''
        def construct_prompt(question: str) -> str:   

            context=self.generate_context(question)
            llm_prompt = f"""
            <|system|>
            Use the following context to answer the question at the end.
            If the question does not make sense or you dont know the answer just say "I dont know", dont try to make up the answer or give advice.
            <|end|>
            <|user|>
            {context}
            {question}
            <|end|>
            <|assistant|>
            """

            return llm_prompt
        
        return construct_prompt(question)    