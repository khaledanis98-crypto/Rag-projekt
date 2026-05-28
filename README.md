# Netflix RAG med Gemini och LangChain

Det här projektet bygger ett RAG-system för Netflix Originals-datasetet.

## Filer

- `data/NetflixOriginals_clean.csv` = städad data
- `01_preprocess_data.ipynb` = undersöker och beskriver datakvalitet
- `02_rag_app.ipynb` = bygger RAG med Gemini, LangChain och Chroma
- `rag_app.py` = samma RAG-system som Python-program
- `requirements.txt` = paket som behövs
- `.env.example` = mall för API-nyckel

## Viktigt

Ladda inte upp din `.env`-fil till GitHub.

## Installera

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

På Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Lägg in API-nyckel

Skapa en fil som heter `.env` och skriv:

```bash
GOOGLE_API_KEY="DIN_API_KEY_HÄR"
```

## Kör RAG-programmet

```bash
python rag_app.py
```

## Exempelfrågor

- Which Netflix original has the highest IMDB score?
- Vilka dokumentärer finns i datasetet?
- Which movies are in Spanish?
- Give me examples of English movies with high IMDB score.
