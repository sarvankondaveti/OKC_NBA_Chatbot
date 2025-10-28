#  2025 Applied AI Engineer Internship Project

Your work must be your own and original. You may use AI tools to help aid your work if you include a single text file containing an ordered list of any AI prompts, along with the specific model queried (e.g. ChatGPT 5 Thinking) in the `prompts` directory. Do not include the AI's output.

### Internship Program Disclosures

* You must be eligible to work in the United States to be able to qualify for this internship.

* The pay for this internship is the greater of your local minimum wage and $13/hour.

* This application is for the purposes of an internship taking place in the Spring, Summer, or Fall of 2026.


## Assignment: 

The goal of this project is to build an end-to-end RAG pipeline with an interactive chat interface for answering basic NBA stats questions.

1. Load data – Ingest CSVs related to NBA game information from the 2023-24 and 2024-25 seasons into PostgresSQL tables. Note this data is limited to only matchups involving at least one Western Conference team for size considerations.
2. Create embeddings – Generate text embeddings with Ollama [`nomic-embed-text`](https://ollama.com/library/nomic-embed-text) and store them alongside the source rows.
3. Retrieve and join – Perform semantic retrieval using the `pgvector` extension to find relevant game summaries, then join the matched embeddings back to the original structured table rows to provide factual context.
4. Answer questions – Use Llama [`llama3.2:3b`](https://ollama.com/library/llama3.2:3b) to produce answers grounded on the retrieved data to the questions under the **Submission Requirements** section. If you find this model too large for your machine, feel free to use a smaller model and note this in your submission.

**The data provided in this repository is proprietary and strictly confidential. It is provided exclusively for use within this technical project and must not be copied, shared, or distributed.**

## Quick Start: Part 1
1) Install [`Docker Desktop`](https://www.docker.com/get-started/) and open it (to ensure the docker daemon is running).
2) Clone this repository.
3) Start services and pull models by running the following commands:
```bash
docker compose up -d db ollama
docker exec ollama ollama pull nomic-embed-text
docker exec ollama ollama pull llama3.2:3b
docker compose build app
```


Edit these files and run them using the following commands:

1) Ingestion (`backend/ingest.py`) for schema details
```
docker compose run --rm app python -m backend.ingest
```

2) Embedding (`backend/embed.py`) for text serialization strategy. **Note the embedding process can take a long time to complete depending on your machine**
```
docker compose run --rm app python -m backend.embed
```

3) RAG Script (`backend/rag.py`) for retrieval joins, prompt, and answer formatting. This script generates answers to the 10 prompts in Part 1.
```
docker compose run --rm app python -m backend.rag
```

## Quick Start: Part 2

### Run the backend server
```
docker compose run --rm --service-ports app uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

### Installing Prerequisites
Install Node.js (16.x.x), then in a new tab, run the following commands
```
cd /path/to/project/frontend
# Install Angular-Cli
npm install -g @angular/cli@15.1.0 typescript@4.9.4 --force
# Install dependencies
npm install --force
# Start the frontend
npm start
```

The frontend should run on http://localhost:4200/. Visit this address to see the app in your browser.


## Submission Requirements

Before starting with the project, __please fill out the [`SUBMISSION.md`](SUBMISSION.md) file__ to ensure we have your name and email address you applied to the role with.

**Part 1: RAG Backend**

- Answer the 10 game-level prompts in [`part1/questions.json`](part1/questions.json) using your retrieval pipeline and record results in a new file named [`answers.json`](part1/answers.json) by simply running the command for the [`rag.py`](backend/rag.py) file. Each answer should include an `evidence` array listing the `game_details` or `player_box_scores` rows used in the response.


**Part 2: Frontend Solution**

- Create a chat interface for interacting with the backend retrieval pipeline. Some minimal Angular skeleton code is provided in the [`frontend/src/app`](frontend/src/app) directory, feel free to edit it as you wish.

Submit a video in the [`part2`](part2) folder that demonstrates how your UI functions.


**Part 3: Writeup**

Note: for this particular section, we **strongly** suggest you avoid using any AI tools to answer any of these questions. We want these responses to be your own voice and show your true understanding of the content. Please limit each response to 500 words or fewer.

1. Discuss your approach to answering the questions in Parts 1 and 2. Include your experimental process in regards to data preparation, embedding design, retrieval, prompt engineering, user experience, etc. Describe any challenges you faced and how you overcame them.

2. Describe your technical skillset and how it relates to the questions answered in this assignment. What did you learn as you went through this assignment, and what do you hope to learn in these related areas?

3. You have data at the sub-possession level capturing basic events like passes, screens, and drives with court coordinates for the ball as well as all 10 players on the court. Describe any ways you would explore this data to answer in-game strategy questions.

4. You have [player positional tracking data](https://pr.nba.com/nba-sony-hawk-eye-innovations-partnership/); assume 29 skeletal points per player (e.g. joints like left shoulder, right knee, right ankle, etc), sampled at 60 frames per second for the full game. How you would harness this high-dimensional data to generate actionable insights for Basketball Operations? Propose three new features, describe how you’d explore the data, and note which technologies you’d apply.

5. You have a large text corpus, including scouting reports, internal notes, and other documents that define decision constraints for Basketball Operations. How would you design an end-to-end process to turn this into findings to support front-office decision-making?  Describe your technical strategies (e.g., NLP, embeddings, retrieval, LLMs), including data representation, analysis, validation, and integration into workflows.


Put your responses to the questions in Part 3 in [`part3/responses.txt`](part3/responses.txt).


## Optional

**Part 4: Embedding Fine-Tuning**
- Build a small dataset of question–context pairs about NBA games and fine-tune the Hugging Face [`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2) model. [`Text Embeddings by Weakly-Supervised Contrastive Pre-training`](https://arxiv.org/pdf/2212.03533) (Wang et al., 2022) shows how E5's contrastive objective yields strong universal embeddings, making it a good candidate for customization.

1. Assemble training data – Create at least 20 question–context pairs from the game summaries in this repo (or generate synthetic ones). Each pair should contain a question and a short text snippet that answers it.
2. Fine-tune an embedding model – Start from the encoder and train it with contrastive learning so matching question/context pairs obtain high cosine similarity. Log the training configuration and any hyperparameters you change.
3. Evaluate retrieval – Compare your fine-tuned model against the baseline `nomic-embed-text` using a held-out set of queries. Report metrics such as Recall@k or MRR.
4. Document results – Summarize your approach, hyperparameters, and evaluation numbers in [`part4/responses.txt`](part4/responses.txt). Include any code or commands used to run the experiment.

Put all relevant files in the [`part4`](part4) folder in this repository.
Note that this part of the project is optional. We do not expect every applicant to complete this portion. 
