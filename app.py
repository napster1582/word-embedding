
import openai
import pandas as pd
from flask import Flask, jsonify, request
import numpy as np
import os
from src.DocumentProcessor import DocumentProcessor
from src.SearchEmbedding import SearchEmbedding

app = Flask(__name__)


@app.route('/embedding/DocumentProcessor', methods=['POST'])
def execute_document_processor():
    data = request.json
    api_key = data['apiKey']
    container_name = data['containerName']
    blob_name = data['blobName']
    role= data['rol']
    queryTunning= data['queryTunning']

    openai.api_key = api_key
    classDocumentProcessor = DocumentProcessor() 

    downloaded_data = classDocumentProcessor.dowload_azure(container_name, blob_name)
    data_readed, extension = classDocumentProcessor.read_file(blob_name, downloaded_data)

    if extension == "pdf":
         paragraphs_separate = classDocumentProcessor.separate_paragraphs_pdf(blob_name, data_readed)

    classDocumentProcessor.define_node_redirect(extension, paragraphs_separate, blob_name, data_readed)
    full_path = os.path.join('./files-embbeding', blob_name[:-3] + "csv")
    paragraphs = pd.read_csv(full_path)
    paragraphs_df = classDocumentProcessor.embed_text_dataframe(paragraphs)
    paragraphs_df=classDocumentProcessor.delete_empty_rows(paragraphs_df)
    paragraphs_df =classDocumentProcessor.chatgptProcessText(paragraphs_df,queryTunning,role)
    paragraphs_df.to_csv(full_path, index=False)

    return jsonify()

@app.route('/embedding/searchEmbedding', methods=['POST'])
def search_embedding():

    data = request.json
    prompt = data['Prompt']
    api_key = data['ApiKey']
    blob_name = data['BlobName']
    context = data['Context']
    classSearchEmbedding = SearchEmbedding(api_key)
 
    full_path = os.path.join('./files-embbeding', blob_name[:-3] + "csv")  
    df_paragraphs = pd.read_csv(full_path)
    prompt = classSearchEmbedding.process_prompt(prompt)

    if context:
          percentage_fallback= 0.77
          percentage_prompt_complete = 0.88
          percentage_use_context= 0.85
          percentage_no_prompt_use_context = 0.88

        #Based on similarity percentages, evaluate if the user is using the context
          node_redirect,bot_result,context = classSearchEmbedding.validate_context(context,prompt,df_paragraphs,percentage_fallback,percentage_prompt_complete,percentage_use_context,percentage_no_prompt_use_context)
    else: 
        prompt_embed = classSearchEmbedding.embed_text(prompt)
        # Convert the embedding strings to numeric arrays so that we can apply the lambda function on the cos of similarity
        df_paragraphs['Embedding'] = df_paragraphs['Embedding'].apply(lambda x: np.array(eval(x)))
        df_result = classSearchEmbedding.search_similarity_df(prompt_embed, df_paragraphs)
        node_redirect = classSearchEmbedding.validate_node_redirect(df_result)
        print(node_redirect)
        bot_result = df_result.iloc[0]['AIProcessedText']
        context = df_result.iloc[0]['WordContext']

    if node_redirect == 'General.OpenAI.Word_Embedding.Fallback':
                    return jsonify(
            {
                "set_attributes": {
                    "context": context
                },
                "redirect_to_node": [node_redirect],
                "messages": [],
                "files": []
            }

            )  
    else:  
            return jsonify(
                {
                    "set_attributes": {
                        "context": context,

                    },
                    "redirect_to_node": [node_redirect],
                    "messages": [
                        {
                            "attachment": {
                                "type": "text",
                                "payload": {
                                    "template_type": "message",
                                    "text": bot_result,
                                    "attribute": "resultadoGPT",
                                    "buttons": []
                                }
                            }
                        }
                    ],
                    "files": []
                }

                )
        


@app.route('/', methods=['GET'])
def home():
    return "Online service to search for answers."

if __name__ == '__main__':
    app.run(debug=True)