## Simple chat with history

This is a simple demo showing how to use Neo4j as the store for the chat history. A local Neo4j instance is used, 
which is started using the docker-compose.yml file.
```bash
docker-compose up -d
```
The Neo4j database can be visualised at http://localhost:7474/browser/. You can the nodes created by running the 
following Cypher query in the browser:
```query 
MATCH (n) RETURN n
```

Remember to pull it down when you're done:
```bash
docker-compose down
```

### Installing dependencies
```bash
pip install streamlit langchain_community python-dotenv neo4j
```
or simply:

```bash
pip install -r requirements.txt
```

### Ollama
This demo is using the open source model mistral, which you can download using Ollama. Follow the instructions at 
[Ollam](https://ollama.com/) to install ollama. Once you have installed ollama, then run:
```bash
ollama pull mistral:7b-instruct
```
to download the model.

### Running the app
Rename .env.sample to .env for the environment variables. Start the app:
```bash
streamlit run main.py
```