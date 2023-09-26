
import pandas as pd
from openai.embeddings_utils import get_embedding
from azure.storage.blob import BlobServiceClient
import os
import fitz 
from src.ChatGPTFunctions import ChatGPT
from io import StringIO 

class DocumentProcessor:
    def __init__(self):
        pass

    def dowload_azure(self,container_name,blob_name):
        azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)

        # Download the file from Azure Blob Storage

        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        downloaded_data = blob_client.download_blob()
        return downloaded_data

    def read_file(self,blob_name,data):
            file_extension = blob_name.split('.')[-1]        
            if file_extension == 'pdf':
                extension= "pdf"
                data_readed = data.readall()    
            else:
                extension= "csv"
                data_readed = data.readall().decode('utf-8')  # Decode the CSV data     

            return data_readed,extension
    
    def separate_paragraphs_pdf(self,blob_name, data_readed):
        
        if os.path.splitext(blob_name)[1] == ".pdf":
            # If the file extension is .pdf, character division is performed
            pdf_document = fitz.open(stream=data_readed, filetype="pdf") 
            textos = []
            # Loop through each page of the PDF and divide its content into paragraphs
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text = page.get_text()
                # Split the content by the 'END' separator
                paragraphs = text.split('END')
                if paragraphs[-1] == '':
                    paragraphs.pop()
                textos.extend(paragraphs)   
            return textos

    def define_node_redirect(self,extension,textos,blob_name,data_readed):

        if extension == "pdf":
            parrafos = pd.DataFrame(textos, columns=["texto"])
            parrafos["Nodo"] = ""

            for index, row in parrafos.iterrows():
                if len(row["texto"]) > 5:
                    if 'node:' in row["texto"]:
                        nodo_texto = row["texto"].split('node:')[1]
                        parrafos.at[index, "Nodo"] = nodo_texto
                        parrafos["texto"] = parrafos["texto"].str.replace("node:"+nodo_texto, "")
                    else:
                        parrafos.at[index, "Nodo"] = 'General.OpenAI.Word_Embedding'
            ruta_completa = os.path.join('./files-embbeding', blob_name[:-3] + "csv")
            parrafos.to_csv(ruta_completa, index=False)

        else:
            # Validate that the empty cells remain with the node type module
            csv_io = StringIO(data_readed)
            csv_df = pd.read_csv(csv_io)
            for index, row in csv_df.iterrows():
                if pd.isnull(row['Nodo']):
                    csv_df.at[index, 'Nodo'] = 'General.OpenAI.Word_Embedding'

            ruta_completa = os.path.join('./files-embbeding', blob_name[:-3] + "csv")
            csv_df.to_csv(ruta_completa, index=False)

    def chatgptProcessText(self,parrafosdataframe,QueryTunning,rol):
            classGTP = ChatGPT()
            for index, row in parrafosdataframe.iterrows():
                # Process with ChatGPT
                prompt = 'Necesito que '+str(QueryTunning)+' y proceses el siguiente texto '+str(parrafosdataframe.at[index, "texto"])
                resultadoGPT=classGTP.process_intention(prompt,rol)
                parrafosdataframe.at[index, "AIProcessedText"] = resultadoGPT
                 # Define context
                prompt = "Por favor, extrae 3 palabras clave importantes del siguiente párrafo que me ayuden a mantener el contexto de la conversación para hacer una contrapregunta. en el siguiente formato: palabra1 palabra2" +str(parrafosdataframe.at[index, "texto"])
                resultadoGPT=classGTP.process_intention(prompt,rol)
                parrafosdataframe.at[index, "WordContext"] = resultadoGPT

            return parrafosdataframe
    
    def embed_text_dataframe(self,dataframe):
        dataframe['Embedding'] = dataframe['texto'].apply(lambda x: get_embedding(str(x), engine='text-embedding-ada-002'))
        return dataframe
    
    def delete_empty_rows(self,paragraphs_df):
        # Filter and remove rows where the 'text' column is not empty
        paragraphs_df = paragraphs_df.dropna(subset=['texto'])
        # reindex DataFrame after deleting rows
        paragraphs_df = paragraphs_df.reset_index(drop=True)
        return paragraphs_df

