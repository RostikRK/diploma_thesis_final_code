## Project description

This repository contains the code for a diploma with a theme "Development of a System for Data Collection from Electronic Component Distributors' Websites", by Kostiuk Rostyslav.

The main directory contains the following directories/files:

- Gathering System: This directory contains the directories with the code for each module of the system. The system diagram is shown in the image
![overall_system.png](readme_images%2Foverall_system.png)
- Interfaces: this directory contains two interfaces that were developed to interact with the collected data: cross-reference tool and Rag-based system
- Tests: this directory contains the scripts that were used to conduct the experiments
- Images Server: here you can find the API that is deployed in the docker container to interact with the Images Database
- helpers: various auxiliary functions for scripts, grouped by purpose
- docker-compose. yaml - a file for deploying containers needed for the system
- requirements.txt - a file with required libraries that must be installed on the local machine for the scripts/modules to run successfully
- create_weaviate_schema.py - script to define the needed schema in weaviate

# General Set Up Steps:

Clone the repository into the directory:
```
git clone https://github.com/RostikRK/Test_upload.git
```

Move into the project root directory:
```
cd ./Test_upload
```

Set up venv environment:
```
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Install the requirements:
```
pip install -r .\requirements.txt
```

Run the docker-compose:
```
docker-compose up -d

# If will be a need to stop use:
docker-compose down
```


I did not organise the credentials management through an .env file because in this thesis I often use Large Language Models and in such cases it is recommended to run the script on a GPU server from Vast AI (I will describe the launch process in further instructions). Therefore, in each script that requires some special credentials, I have highlighted a block that is worth reviewing and adjusting to your needs. The block is marked with the following comments:
```
# Set up start
...
# Set up end
```


To fully deploy the system and use all modules, you will need to obtain the following credentials:

| Credential          | Link where to get it|
|---------------------|-------|
| DIGI_CLIENT_ID      | [link](https://developer.digikey.com/) |
| DIGI_CLIENT_SECRET  | [link](https://developer.digikey.com/) |
| NEXAR_CLIENT_ID     | [link](https://nexar.com/api) |
| NEXAR_CLIENT_SECRET | [link](https://nexar.com/api) |
| MISTRAL_API_KEY     | [link](https://docs.mistral.ai/api/) |
| GOOGLE_API_KEY      | [link](https://aistudio.google.com/app/apikey) |


# Detailed Instructions

Before going to the modules, run the script to define the proper weaviate schema:

```
python .\create_weaviate_schema.py
```

## Gathering System

### Digikey Service

In respective folder there are 2 python script responsible to get information for their category. To run them use those commands from project root directory:

```
python .\Gathering_System\Digikey\process_memory.py

python .\Gathering_System\Digikey\process_transceivers.py
```

### Nexar Service

In this folder there is just one script responsible for data extraction. To run it use this command:

```
 python .\Gathering_System\Nexar\program_nexar.py
```

### Description Generator

This module uses LLM. Below is the set of commands to run this script on GPU server.

Firstly connect to the server:

```
ssh -p {your_server_port} root@{your_server_ip_address} -L 8080:localhost:8080
```

Then open the other terminal to copy the scripts on server:

```
scp -P {your_server_port} .\Gathering_System\description_generator\attribute_description_generation.py .\Gathering_System\description_generator\package_description_generation.py root@{your_server_ip_address}:/workspace
```

Then we also should open the local Neo4j port to be available from server (same terminal that was used for copying):
```
ssh -R {Neo4j port}:localhost:{Local Neo4j port} root@{your_server_ip_address}  -p {your_server_port}
```

Next step is to install required libraries:
```
pip install neo4j unsloth transformers torch sentence_transformers
```

Finally we can run the scripts on server:

```
python attribute_description_generation.py

python package_description_generation.py
```

### Datasheet Processor

There are two scripts:

- add_single_datasheet.py - that can add entities and split datasheet into the chunks uploaded by the user.
- split-datasheet.ipynb - it allow to chunk all of the still unprocessed datasheets

You can run them locally as the previous local scripts. 

### Image validator

Here is just one script that validates the datasheet images and classifies them into different categories for optimization.

It should be run on the server. So here is a set of commands to apply (by analogy to description generator that was running on the GPU server):

```
# Connect
ssh -p {your_server_port} root@{your_server_ip_address} -L 8080:localhost:8080

# Copy and allow connections (in other terminal)

scp -P {your_server_port} .\Gathering_System\image_validator\image_validator.py  root@{your_server_ip_address}:/workspace

ssh -R {Neo4j port}:localhost:{Local Neo4j port} -R {Images API port}:localhost:{Local Images API port} root@{your_server_ip_address}  -p {your_server_port}

# Install requirements
pip install neo4j unsloth transformers torch pillow

# Run
python image_validator.py
```

### Image processor

Single script - mistral_service.py that applies the Mistral OCR processing on the complex and useful images.

Run locally

### Data Extractor

In this module I provide one smooth script that works on the image chunks.

Again apply steps to run them on server:

```
# Connect
ssh -p {your_server_port} root@{your_server_ip_address} -L 8080:localhost:8080

# Copy and allow connections (in other terminal)

## For image-based
scp -P {your_server_port} .\Gathering_System\data_extractor\multimodal_extraction.py .\helpers\extraction_utils.py .\helpers\attributes_matcher.py root@{your_server_ip_address}:/workspace

ssh -R {Neo4j port}:localhost:{Local Neo4j port} -R {Images API port}:localhost:{Local Images API port} root@{your_server_ip_address}  -p {your_server_port}

# Install requirements
pip install neo4j unsloth transformers torch matplotlib seaborn pydantic jellyfish pymupdf4llm sentence_transformers

# Run
python multimodal_extraction.py
```

### Feature Vector Calculator

In this module is provided a single script that computes the feature vector for the products.

Run locally.


## Interfaces

### Cross-reference search tool

Here is single tool script that based on the computed feature vector performs the cross-reference products search across the gathered data.

You can run it locally using the command:

```
python .\Interface\cross-reference_tool\cross_reference_search_tool.py
```

### RAG system

This interface provides an chat alike interface through the API, that under the hood uses the RAG approach. This interface is show on the diagram:
![Rag-based-interface.png](readme_images%2FRag-based-interface.png)

To run this interface you should firstly configure the inference server, so run those commands:
```
# Connect
ssh -p {your_server_port} root@{your_server_ip_address} -L 8080:localhost:8080

# Copy and allow connections (in other terminal)

scp -P {your_server_port} .\Interfaces\Rag_system\remote_llm_service.py .\Interfaces\Rag_system\models.py .\Interfaces\Rag_system\config.py root@{your_server_ip_address}:/workspace

ssh -L 5000:localhost:5000 -p {your_server_port} root@{your_server_ip_address}

# Install requirements
pip install neo4j unsloth transformers torch fastapi uvicorn

# Run
python remote_llm_service.py
```

Now you can run the local API:
```
python .\Interface\Rag_system\api.py
```

It is running on localhost:8000 and has a main endpoint to which you can make the requests: POST ("/query"). In the body you should specify your request in the form of simple JSON with key "query".

On image is an example in Postman:
![Rag-example.png](readme_images%2FRag-example.png)

## Tests

### Cross-reference Tests

In the directory is provided a part of the test dataset that was used for the experiments in the diploma thesis ("Test cross-reference.xlsx"). Actually 50 of manually gathered test pairs, other 100 provided by the company can not be published. But if there is a need crucial need write to the rostyslav.kostiuk@ucu.edu.ua with request.

The test script "cross_reference_test_main.ipynb" can be run locally. Also, to reproduce the experiment that are presented in diploma thesis there should be beforehand configured the parameters in the "Feature Vector Calculator" module. Such parameters are number of top attributes and embedding model. Also keep in mind that some embedding models can not compute the similarity score using cosine similarity function. The experiments were based on the fully gathered database from two categories described in the diploma.

### Extraction Tests

The test dataset for those experiments can be downloaded by the [link](https://huggingface.co/datasets/Rostik7272/semiconductor_datasheets_JSON_extraction).

There are 3 script that perform various models tests:
- gemini_script_main - it uses the close source SOTA models using the API to perform the extraction from image chunks. It can be run locally.
- multimodal_script_main - it uses the open source multimodal model to perform the extraction from image chunks. This script should be run on GPU Server.
- text_script_main - it uses the open source model to perform the extraction from text chunks. This script should be run on GPU Server.

So to run the test scripts firstly connect to the server:

```
ssh -p {your_server_port} root@{your_server_ip_address} -L 8080:localhost:8080
```

Copy the script with utils and test dataset on server:

```
# Text based
scp -P {your_server_port} ./Tests/extraction_test_scripts/text_script_main.py ./Tests/extraction_test_scripts/extraction_evaluation_utils.py ./helpers/extraction_utils.py PART_TO_TEST_DF:/test_df.zip root@{your_server_ip_address}:/workspace

# Image based
scp -P {your_server_port} ./Tests/extraction_test_scripts/multimodal_script_main.py ./Tests/extraction_test_scripts/extraction_evaluation_utils.py ./helpers/extraction_utils.py PART_TO_TEST_DF:/test_df.zip root@{your_server_ip_address}:/workspace
```

Unzip test df:
```
unzip test_df.zip
```

Install requirements:
```
pip install neo4j unsloth transformers torch matplotlib seaborn pydantic jellyfish pymupdf4llm sentence_transformers
```

Run scripts:
```
python text_script_main.py --test_dir ./test_df --output_dir ./outputs

python multimodal_script_main.py --test_dir ./test_df --output_dir ./outputs
```

