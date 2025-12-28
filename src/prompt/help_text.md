### 📚 PanKnode Help

Available commands:
- `/addfile`: Summarizes the content of uploaded file and saves it to the knowledge base.
- `/scanfiles`: Scans the `db/kb_pdf` folder, converts PDF files to text, summarizes them, and saves them to the knowledge base.
- `/embedding`: Vectorizes unprocessed `.md` files in the knowledge base using the configured embedding model and chunking settings.
- `/list`: Lists all summarized `.md` files in the knowledge base.
- `[filename]/cat`: Displays the content of the specified `.md` file.
- `[keyword]/search`: Searches for files with names containing the keyword (case-insensitive).
- `[query] [type]/[count]/query`: Performs a semantic search. `type` can be `file`, `sumr`, `abs`, `litr`, `ref`, or `origin` (default: `file`). `count` is the number of top relevant results to retrieve (default: 20).
- `[topic] [type]/[count]/generate`: Triggers a two-round research agent. `type` specifies the search source. `count` is the number of context chunks to retrieve for research synthesis (default: 20).
- `/debug`: Checks the database for files with missing or empty vector sections.
- `[filename] [type]/re`: Re-generates a specific section of an existing file. `type` can be `absref`, `sumr`, or `litr`.
- `/help`: Shows this help message.
