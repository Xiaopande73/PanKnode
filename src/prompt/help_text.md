### 📚 PanKnode Help

Available commands:
- `/addfile`: Summarizes the content of uploaded file and saves it to the knowledge base.
- `/scanfiles`: Scans the `db/kb_pdf` folder, converts PDF files to text, summarizes them, and saves them to the knowledge base.
- `/embedding`: Vectorizes unprocessed `.md` files in the knowledge base using the configured embedding model and chunking settings.
- `/list`: Lists all summarized `.md` files in the knowledge base.
- `[filename]/cat`: Displays the content of the specified `.md` file.
- `[keyword]/search`: Searches for files with names containing the keyword (case-insensitive).
- `[query][count]/queryfile`: Performs a semantic search for files related to the natural language `query`. `count` is optional (default 30).
- `[query] [count]/querysumr`: Performs a semantic search in Summaries for content related to `query`. `count` is optional (default 20).
- `[query] [count]/queryabs`: Performs a semantic search in Abstracts for content related to `query`. `count` is optional (default 20).
- `[query] [count]/querylitr`: Performs a semantic search in Literature Reviews for content related to `query`. `count` is optional (default 20).
- `[query] [count]/queryref`: Performs a semantic search in references for chunks related to `query`. `count` is optional (default 30).
- `[query] [count]/queryorigin`: Performs a semantic search in original content for chunks related to `query`. `count` is optional (default 10).
- `[topic] [type]/[count]/generate`: Triggers a two-round research agent. `type` can be `sumr`, `abs`, `litr`, `origin`, or `ref`. `count` is optional(default 30).
- `/help`: Shows this help message.