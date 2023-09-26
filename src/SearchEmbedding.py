
import openai
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import numpy as np


class SearchEmbedding():
    def __init__(self, api_key):
        openai.api_key = api_key

    def process_prompt(self, prompt):
        # Process the prompt, to have a better similarity with the paragraphs
        processed_prompt = prompt.replace('?', '').replace('¿', '').replace('!', '')
        return processed_prompt


    def embed_text(self,busqueda):
        busqueda_embed = get_embedding(busqueda, engine="text-embedding-ada-002")
        return busqueda_embed

    def search_similarity_df(self, busqueda_embed, paragraphs, n_resultados=1):        
        paragraphs["Similitud"] = paragraphs['Embedding'].apply(lambda x: cosine_similarity(x, busqueda_embed))
        paragraphs = paragraphs.sort_values("Similitud", ascending=False) 

        # Return the first data with the highest similarity
        return paragraphs.iloc[:n_resultados][["texto", "Similitud", "Embedding", "Nodo", "AIProcessedText", "WordContext"]]
    
    def search_similarity_list(self,text1, text2):
        percentage_similarity_list  = openai.embeddings_utils.cosine_similarity(text1, text2)
        return percentage_similarity_list

    def validate_node_redirect(self,resultados):
        porcentaje = resultados.iloc[0]['Similitud']
        if porcentaje < 0.77:
            node_redirect = 'General.OpenAI.Word_Embedding.Fallback' 
        else:
            node_redirect = str(resultados.iloc[0]["Nodo"]).replace("\n", "").strip()
        return node_redirect
    
    def validate_context(self,context,prompt,df_paragraphs,percentage_fallback,percentage_prompt_complete,percentage_use_context,percentage_no_prompt_use_context):
         
        prompt_embeb = self.embed_text(prompt)
        # Convert the embedding strings to numeric arrays so that we can apply the lambda function on the cos of similarity   
        df_paragraphs['Embedding'] = df_paragraphs['Embedding'].apply(lambda x: np.array(eval(x)))
        df_result = self.search_similarity_df(prompt_embeb, df_paragraphs)
        paragraph1 = df_result.iloc[0]['AIProcessedText']
        context_paragraph1= df_result.iloc[0]['WordContext']
       
        if df_result.iloc[0]['Similitud'] < percentage_fallback:
            node_redirect = "General.OpenAI.Word_Embedding.Fallback"

        elif df_result.iloc[0]['Similitud'] > percentage_prompt_complete:
                # The user completes the prompt and is not related to the context
                print("The user completes the prompt and is not related to the context")
                context =context_paragraph1
                bot_result = df_result.iloc[0]['AIProcessedText']
        else:
            # Compare context and paragraph 1
            context_embeb = self.embed_text(context)
            percentage_similarity = self.search_similarity_list(context_embeb, df_result.iloc[0]['Embedding']) 

            if percentage_similarity > percentage_use_context:
                # The user completes the prompt and is related to the context
                print("The user completes the prompt and is related to the context")
                context =context_paragraph1
                bot_result = df_result.iloc[0]['AIProcessedText']
            else:
                # The prompt and the context are concatenated to do the search
                final_prompt = prompt + " " +context
                print(final_prompt)
                final_prompt_list = self.embed_text(final_prompt)
                df_result = self.search_similarity_df(final_prompt_list, df_paragraphs)
                percentage_similarity_context_prompt = df_result.iloc[0]['Similitud']
                context_paragraph2= df_result.iloc[0]['WordContext']
                if percentage_similarity_context_prompt > percentage_no_prompt_use_context:
                    # Non-specific prompt and use of context
                    print("Non-specific prompt and use of context")
                    bot_result = df_result.iloc[0]['AIProcessedText']
                    context =context_paragraph2
                else:
                    # User is talking about a new topic in the prompt, return paragraph1
                    print("User is talking about a new topic in the prompt")
                    bot_result = paragraph1
                    context =context_paragraph1
        node_redirect = self.validate_node_redirect(df_result)
        return node_redirect,bot_result,context
        