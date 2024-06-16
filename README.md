# LangGraphRAG

LangGraphRAG is a terminal-based Retrieval-Augmented Generation (RAG) system implemented using LangGraph. The architecture is designed to handle queries by routing them through a series of processes involving message history caching, query transformation, and document retrieval from a vector database.

## Project Structure

The project is divided into several modules, each responsible for specific functionalities:
1. **Architecture**: Defines the flow of the RAG system.
2. **Data**: Contains data files and models.
3. **Modules**: Houses the core logic and functions.

## Setup Instructions

Follow these steps to set up and run the project:

1. **Clone the repository**:
   ```sh
   git clone https://github.com/ranguy9304/LangGraphRAG.git
   cd LangGraphRAG
   ```

2. **Create a virtual environment**:
   ```sh
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the requirements**:
   ```sh
   pip install -r requirements.txt
   choco install wkhtmltopdf
   ```

4. **Configure the environment variables**:
   - Copy the example environment file:
      ```sh
      cp .env.example .env
      ```
   - Modify the `.env` file to add your GPT key:
      ```env
      OPENAI_API_KEY=your_gpt_key_here
      ```
   - Add your webpage urls if using webpages (keep it comma seperated without quotations):
      ```env
      URLS =url1,url2
      ```
   - set the `GET_WEB_PAGES_TO_PDF` to True if downloading webpages else False:
      ```env
      GET_WEB_PAGES_TO_PDF=False

      ```
   - set the `CONVERT_PDF_TO_MD` to True if already have pdf else False:
      ```env
      CONVERT_PDF_TO_MD=True

      ```
   - If using PDF docs directly store them in the path used in `INTERMEDIATE_PDF_DIR`
   - If using .md docs directly store them in the path used in `DATA_DIR`


5. **Setup documents**:
   from the root directory run
   ```sh
   python modules/processDocs.py
   ```
   this sets up the webpages and docs. Dont forget to modify the document processing parameters in .env as per your needs.


6. **Run the main program**:
   ```sh
   python main.py
   ```

## Usage

- The system handles queries by routing them through different processes.
- It uses LangGraph to manage the flow and interactions between modules.

## Diagrams

### Vector DB Creation
![Vector DB Creation](https://github.com/ranguy9304/LangGraphRAG/raw/main/architecture/vectordb_creation.png)

### RAG Architecture
![RAG Architecture](https://github.com/ranguy9304/LangGraphRAG/raw/main/architecture/RAG.png)


## Contribution

Feel free to fork the repository and submit pull requests. For major changes, please open an issue to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

