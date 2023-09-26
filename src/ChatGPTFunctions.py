import openai
import json

class ChatGPT():
    def __init__(self):
        pass
    
    def process_functions(self, text):
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                    #Si no te gusta que te hable feo, cambia aqui su descripcion
                    {"role": "system", "content": "Eres un asistente muy atento y divertido"},
                    {"role": "user", "content": text},
            ], 
            functions=[
                {
                    "name": "get_weather",
                    "description": "Obtener el clima actual",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ubicacion": {
                                "type": "string",
                                "description": "La ubicación, debe ser una ciudad",
                            }
                        },
                        "required": ["ubicacion"],
                    },
                },
                {
                    "name": "send_email",
                    "description": "Enviar un correo",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recipient": {
                                "type": "string",
                                "description": "La dirección de correo que recibirá el correo electrónico",
                            },
                            "subject": {
                                "type": "string",
                                "description": "El asunto del correo",
                            },
                            "body": {
                                "type": "string",
                                "description": "El texto del cuerpo del correo",
                            }
                        },
                        "required": [],
                    },
                },
                {
                    "name": "open_chrome",
                    "description": "Abrir el explorador Chrome en un sitio específico",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "website": {
                                "type": "string",
                                "description": "El sitio al cual se desea ir"
                            }
                        }
                    }
                },
                {
                    "name": "disconnect_bot",
                    "description": "Desconectar inteligencia artificial",
                    "parameters": {
                        "type": "object",
                        "properties": {
                        }
                    },
                },
                {
                    "name": "word_embedding",
                    "description": "Utilizar Word Embedding",
                    "parameters": {
                        "type": "object",
                        "properties": {
                             "query": {
                                "type": "string",
                                "description": "La pregunta sobre el archivo"
                            }
                        }
                    },
                }
            ],
            function_call="auto",
        )
        
        message = response["choices"][0]["message"]
        
        #Nuestro amigo GPT quiere llamar a alguna funcion?
        if message.get("function_call"):
            #Sip
            function_name = message["function_call"]["name"] #Que funcion?
            args = message.to_dict()['function_call']['arguments'] #Con que datos?
            print("Funcion a llamar: " + function_name)
            args = json.loads(args)
            return function_name, args, message
        
        return None, None, message
    
    #Una vez que llamamos a la funcion (e.g. obtener clima, encender luz, etc)
    #Podemos llamar a esta funcion con el msj original, la funcion llamada y su
    #respuesta, para obtener una respuesta en lenguaje natural (en caso que la
    #respuesta haya sido JSON por ejemplo
    def process_response(self, text, message, function_name, function_response):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                #Aqui tambien puedes cambiar como se comporta
                {"role": "system", "content": "Eres un asistente muy atento y divertido"},
                {"role": "user", "content": text},
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
            ],
        )
        return response["choices"][0]["message"]["content"]
    
    def process_intention(self, text,rol):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                    #Si no te gusta que te hable feo, cambia aqui su descripcion
                    {"role": "system", "content": rol},
                    {"role": "user", "content": text},    
            ]        
        )
        
        message = response["choices"][0]["message"]['content']          
        return message


