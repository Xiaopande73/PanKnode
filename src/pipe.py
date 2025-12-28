import os
import re
import sys
import logging
from typing import Any, Callable, Awaitable
from pydantic import BaseModel, Field
from open_webui.models.users import User
from open_webui.constants import TASKS

sys.path.append("/app/backend/data/FunctionLib/PanKnode/src")
from PanKnode import PanKnode

def setup_logger():
    logger = logging.getLogger("PanKnode")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

logger = setup_logger()

class Pipe(PanKnode):
    __event_emitter__: Callable[[dict], Awaitable[None]]
    __user__: User
    __request__: Any

    # Fixed paths for internal use
    PROMPT_PATH = "/app/backend/data/FunctionLib/PanKnode/src/prompt"
    OUTPUT_PATH = "/app/backend/data/FunctionLib/PanKnode/db/kb_md"
    PDF_PATH = "/app/backend/data/FunctionLib/PanKnode/db/kb_pdf"
    DB_PATH = "/app/backend/data/FunctionLib/PanKnode/db/kb_embeddings.db"
    GEN_PATH = "/app/backend/data/FunctionLib/PanKnode/db/gen"

    class Valves(BaseModel):
        MODEL_ID: str = Field(
            default="deepseek-chat",
            description="Model ID for summarizing content",
        )
        EMBEDDING_BASE_URL: str = Field(
            default="https://openrouter.ai/api",
            description="Base URL for embedding API (will append /v1/embeddings)",
        )
        EMBEDDING_MODEL: str = Field(
            default="qwen/qwen3-embedding-8b",
            description="Model ID for embeddings",
        )
        EMBEDDING_API_KEY: str = Field(
            default="",
            description="API Key for embedding service",
        )
        SUMMARY_CHUNK_SIZE: int = Field(
            default=100,
            description="Word count for Summary/Abstract chunks",
        )
        SUMMARY_OVERLAP: int = Field(
            default=30,
            description="Word count for Summary overlap",
        )
        REF_CHUNK_SIZE: int = Field(
            default=50,
            description="Word count for Reference chunks",
        )
        REF_OVERLAP: int = Field(
            default=20,
            description="Word count for Reference overlap",
        )
        ORIGINAL_CHUNK_SIZE: int = Field(
            default=200,
            description="Word count for Original Content chunks",
        )
        ORIGINAL_OVERLAP: int = Field(
            default=50,
            description="Word count for Original Content overlap",
        )
        LITR_CHUNK_SIZE: int = Field(
            default=100,
            description="Word count for Literature Review chunks",
        )
        LITR_OVERLAP: int = Field(
            default=30,
            description="Word count for Literature Review overlap",
        )

    def __init__(self):
        super().__init__(
            db_path=self.DB_PATH,
            kb_path=self.OUTPUT_PATH,
            gen_path=self.GEN_PATH
        )
        self.type = "manifold"
        self.valves = self.Valves()
        self.citation = False

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": "deep-research-pipe", "name": "PanKnode"}]

    async def emit_status(self, level: str, message: str, done: bool = False):
        if self.__event_emitter__:
            await self.__event_emitter__({
                "type": "status",
                "data": {
                    "status": "complete" if done else "in_progress",
                    "level": level,
                    "description": message,
                    "done": done
                },
            })

    async def emit_message(self, message: str):
        if self.__event_emitter__:
            await self.__event_emitter__({
                "type": "message",
                "data": {"content": message}
            })

    async def handle_help(self):
        """Handle the /help command."""
        help_path = f"{self.PROMPT_PATH}/help_text.md"
        help_text = "### 📚 PanKnode Help\n\nHelp file not found."

        if os.path.exists(help_path):
            try:
                with open(help_path, "r", encoding="utf-8") as f:
                    help_text = f.read()
            except Exception as e:
                logger.error(f"Error reading help file: {e}")

        await self.emit_message(help_text)

    async def handle_debug(self):
        """Handle the /debug command to check for database issues."""
        await self.emit_status("info", "Checking for empty embeddings...", False)
        empty_files = self.get_empty_embedding_files()

        if not empty_files:
            await self.emit_message("✅ No empty embeddings found in the database. All files seem complete.")
            await self.emit_status("success", "Database check complete", True)
            return

        report = ["### 🛠️ Debug Report: Empty Embeddings Found\n"]
        report.append(f"Found {len(empty_files)} files with missing or empty vector sections.\n")

        for filename, sections in empty_files.items():
            report.append(f"- **{filename}**: {', '.join(sections)}")

        report.append("\n**Possible reasons:** Missing headers in Markdown files or content too short to chunk.")
        await self.emit_message("\n".join(report))
        await self.emit_status("warning", f"Found {len(empty_files)} incomplete files", True)

    async def handle_re(self, content_before: str):
        """Handle the /re command to re-generate specific sections."""
        # Expected format: filename.md type/re
        # re_type can be absref, sumr, litr
        match = re.search(r"^(.*?)\s+(absref|sumr|litr)/re$", f"{content_before}/re", re.IGNORECASE)
        if not match:
            await self.emit_message("⚠️ Invalid format for `/re`. Use `filename.md type/re` where type is `absref`, `sumr`, or `litr`.")
            return

        filename = match.group(1).strip()
        re_type = match.group(2).lower()

        await self.emit_message(f"🔄 Re-generating {re_type} for `{filename}`. Please wait...")
        await self.emit_status("info", f"Reprocessing {re_type}...", False)

        success = await self.reprocess_section(
            filename=filename,
            re_type=re_type,
            output_path=self.OUTPUT_PATH,
            prompt_path=self.PROMPT_PATH,
            model_id=self.valves.MODEL_ID,
            request=self.__request__,
            user=self.__user__
        )

        if success:
            await self.emit_message(f"✅ Successfully re-generated and updated {re_type} in `{filename}`.")
            await self.emit_status("success", "Reprocessing complete", True)
        else:
            await self.emit_message(f"❌ Failed to re-generate {re_type} for `{filename}`. Check logs for details.")
            await self.emit_status("error", "Reprocessing failed", True)

    async def handle_list(self):
        """Handle the /list command."""
        await self.emit_status("info", "Listing files...", False)
        files = await self.list_files(self.OUTPUT_PATH)
        if files:
            file_list = "\n".join([f"- {f}" for f in files])
            await self.emit_message(f"### 📁 Knowledge Base Files\n\n{file_list}")
            await self.emit_status("success", f"Found {len(files)} files", True)
        else:
            await self.emit_message("📁 The knowledge base is currently empty.")
            await self.emit_status("info", "No files found", True)

    async def handle_cat(self, filename: str):
        """Handle the /cat command."""
        if not filename.strip():
            await self.emit_message("⚠️ Please specify a filename before `/cat` (e.g., `filename.md/cat`).")
            return

        await self.emit_status("info", f"Reading {filename}...", False)
        content = await self.get_file_content(self.OUTPUT_PATH, filename.strip())
        if content:
            # Strip Original Content for display
            if "## Original Content" in content:
                content = content.split("## Original Content")[0].strip()

            await self.emit_message(f"### 📄 Content of {filename}\n\n{content}")
            await self.emit_status("success", "File read successfully", True)
        else:
            await self.emit_message(f"❌ File `{filename}` not found or is empty.")
            await self.emit_status("error", "File not found", True)

    async def handle_search(self, query: str):
        """Handle the /search command."""
        if not query.strip():
            await self.emit_message("⚠️ Please specify a keyword before `/search` (e.g., `transformer/search`).")
            return

        await self.emit_status("info", f"Searching for '{query}'...", False)
        files = await self.search_files(self.OUTPUT_PATH, query.strip())
        if files:
            file_list = "\n".join([f"- {f}" for f in files])
            await self.emit_message(f"### 🔍 Search Results for '{query}'\n\n{file_list}")
            await self.emit_status("success", f"Found {len(files)} matches", True)
        else:
            await self.emit_message(f"🔍 No files found containing '{query}' in their name.")
            await self.emit_status("info", "No matches found", True)

    async def handle_addfile(self, content_before: str):
        """Handle the /addfile command by summarizing the content before it."""
        if not content_before.strip():
            await self.emit_message("⚠️ No content found to summarize. Please provide text or upload a file before the `/addfile` command.")
            return

        processed_content = self.extract_context_content(content_before)
        if not processed_content:
            await self.emit_message("⚠️ No content found within `<context></context>` tags to summarize.")
            return

        await self.emit_message("Starting summarization process. This may take a while, please wait patiently... \n")
        await self.emit_status("info", "Summarizing content...", False)
        try:
            summary, saved_path = await self.summarize_content(
                content=processed_content,
                prompt_path=self.PROMPT_PATH,
                model_id=self.valves.MODEL_ID,
                request=self.__request__,
                user=self.__user__,
                output_path=self.OUTPUT_PATH,
                original_content=processed_content
            )

            if summary == "TOO_SHORT":
                await self.emit_message("⚠️ The content provided is too short to be meaningfully summarized. Please provide a more substantial text (at least 500 characters).")
                await self.emit_status("warning", "Content too short", True)
                return

            if summary == "ALREADY_EXISTS":
                filename = os.path.basename(saved_path)
                await self.emit_message(f"ℹ️ A summary for this paper already exists: `{filename}`. Skipping generation.")
                await self.emit_status("success", "File already exists", True)
                return

            if summary:
                await self.emit_message(f"### 📝 Content Summary\n\n{summary}")

                if saved_path:
                    filename = os.path.basename(saved_path)
                    await self.emit_status("info", f"Saved to {filename}", False)
                else:
                    await self.emit_status("warning", "Failed to save summary to file", False)

                await self.emit_status("success", "Summarization complete", True)
            else:
                await self.emit_message("❌ Failed to generate summary. Please check the logs.")
                await self.emit_status("error", "Summarization failed", True)
        except Exception as e:
            await self.emit_message(f"❌ Error during summarization: {e}")
            await self.emit_status("error", "Error occurred", True)

    async def handle_scanfiles(self):
        """Handle the /scanfiles command."""
        await self.emit_message("Starting PDF scan and summarization process. This may take a while, please wait patiently... \n")
        await self.emit_status("info", f"Scanning directory: {self.PDF_PATH}", False)
        pdf_files = await self.scan_pdf_directory(self.PDF_PATH)

        if not pdf_files:
            await self.emit_message(f"📁 No PDF files found in `{self.PDF_PATH}`.")
            await self.emit_status("info", "Scan complete: No files", True)
            return

        total = len(pdf_files)
        success_list = []
        fail_list = []

        await self.emit_status("info", f"Found {total} PDF files. Starting sequential processing...", False)

        for i, filename in enumerate(pdf_files):
            file_path = os.path.join(self.PDF_PATH, filename)
            current_status = f"Processing ({i+1}/{total}): {filename}"
            await self.emit_status("info", current_status, False)

            try:
                # 1. Convert PDF to text
                text_content = await self.process_pdf_to_text(file_path)

                if not text_content or len(text_content.strip()) < 50:
                    raise Exception("Extracted text is too short or empty.")

                # Clean text: remove newlines and multiple spaces
                text_content = self.clean_text(text_content)

                # 2. Summarize (Like /addfile)
                # We use the text_content directly as processed_content
                summary, _ = await self.summarize_content(
                    content=text_content,
                    prompt_path=self.PROMPT_PATH,
                    model_id=self.valves.MODEL_ID,
                    request=self.__request__,
                    user=self.__user__,
                    output_path=self.OUTPUT_PATH,
                    original_content=text_content
                )

                if summary == "TOO_SHORT":
                    raise Exception("Content too short for summarization.")

                if summary == "ALREADY_EXISTS":
                    filename_md = os.path.basename(_)
                    await self.emit_status("info", f"Skipped (already exists): {filename_md}", False)
                    success_list.append(f"{filename} (Skipped)")
                    # Delete the PDF file since it's already processed
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Failed to delete {filename}: {e}")
                    continue

                if not summary:
                    raise Exception("Failed to generate summary.")

                success_list.append(filename)
                # Delete the PDF file after successful processing
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Failed to delete {filename}: {e}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                fail_list.append((filename, str(e)))

        # Final Report
        report = [f"### 🚀 Scan and Process Report\n"]
        report.append(f"**Total files found:** {total}")
        report.append(f"**Successfully processed:** {len(success_list)}")

        if success_list:
            report.append("\n✅ **Success:**")
            for s in success_list:
                report.append(f"- {s}")

        if fail_list:
            report.append("\n❌ **Failed:**")
            for f_name, reason in fail_list:
                report.append(f"- {f_name}: `{reason}`")

        await self.emit_message("\n".join(report))
        await self.emit_status("success", "Scan and processing complete", True)

    async def handle_embedding(self):
        """Handle the /embedding command."""
        if not self.valves.EMBEDDING_API_KEY:
            await self.emit_message("⚠️ `EMBEDDING_API_KEY` is not set in Valves. Please configure it first.")
            await self.emit_status("error", "API Key missing", True)
            return

        await self.emit_message("Starting vectorization process. This may take a while, please wait patiently... \n")
        await self.emit_status("info", "Scanning for markdown files...", False)
        md_files = await self.list_files(self.OUTPUT_PATH)
        processed_files = self.get_processed_files()

        # Filter out files that have already been processed
        to_process = [f for f in md_files if f not in processed_files]

        if not to_process:
            await self.emit_message("📁 No new markdown files found to vectorize. Everything is up to date.")
            await self.emit_status("success", "Nothing to process", True)
            return

        total = len(to_process)
        success_count = 0
        fail_list = []

        await self.emit_status("info", f"Found {total} new files. Starting vectorization...", False)

        for i, filename in enumerate(to_process):
            file_path = os.path.join(self.OUTPUT_PATH, filename)
            current_status = f"Vectorizing ({i+1}/{total}): {filename}"
            await self.emit_status("info", current_status, False)

            success = await self.process_file(
                filename=filename,
                file_path=file_path,
                valves=self.valves,
                api_key=self.valves.EMBEDDING_API_KEY
            )

            if success:
                success_count += 1
            else:
                fail_list.append(filename)

        # Final Report
        report = [f"### 🧬 Embedding Process Report\n"]
        report.append(f"**Total new files found:** {total}")
        report.append(f"**Successfully vectorized:** {success_count}")

        if fail_list:
            report.append("\n❌ **Failed:**")
            for f_name in fail_list:
                report.append(f"- {f_name}")

        await self.emit_message("\n".join(report))
        await self.emit_status("success", "Vectorization complete", True)

    async def handle_query(self, query_text: str, search_type: str = "file", top_n: int = 20):
        """Unified handler for semantic search queries."""
        if not self.valves.EMBEDDING_API_KEY:
            await self.emit_message("⚠️ `EMBEDDING_API_KEY` is not set in Valves.")
            await self.emit_status("error", "API Key missing", True)
            return

        if not query_text.strip():
            await self.emit_message(f"⚠️ Please provide a query before `/query` (e.g., `transformer {search_type}/{top_n}/query`).")
            await self.emit_status("error", "Empty query", True)
            return

        # Mapping of search types to their descriptive names, functions, and result labels
        type_map = {
            "file": ("files", self.query_top_files, "relevant files"),
            "origin": ("original content", self.query_top_origin_chunks, "relevant chunks"),
            "sumr": ("Summary content", self.query_top_sumr, "relevant summaries"),
            "abs": ("Abstract content", self.query_top_abs, "relevant abstracts"),
            "litr": ("Literature Review sections", self.query_top_litr, "relevant sections"),
            "ref": ("references", self.query_top_ref_chunks, "relevant chunks")
        }

        if search_type not in type_map:
            search_type = "file"

        display_name, query_func, result_desc = type_map[search_type]

        await self.emit_message(f"Searching for '{query_text}' in {display_name}. This may take a while, please wait patiently... \n")
        await self.emit_status("info", f"Searching for '{query_text}' in {display_name} (top {top_n})...", False)

        try:
            results = await query_func(
                query=query_text,
                top_n=top_n,
                valves=self.valves,
                api_key=self.valves.EMBEDDING_API_KEY
            )

            if not results:
                await self.emit_message(f"🔍 No relevant {display_name} found for '{query_text}'.")
                await self.emit_status("info", "No results", True)
                return

            report = [f"### 🔍 Semantic Search Results ({search_type}) for '{query_text}'\n"]

            for i, item in enumerate(results):
                # query_top_files returns (filename, score)
                if search_type == "file":
                    filename, score = item
                    clean_title = filename.replace(".md", "").replace("_", " ")
                    report.append(f"### {i+1}. {clean_title}")
                    report.append(f"**Score:** {score:.4f}")
                    report.append(f"- **Filename:** `{filename}`\n")
                else:
                    # Others return dict with filename, score, and text
                    clean_title = item["filename"].replace(".md", "").replace("_", " ")
                    report.append(f"### {i+1}. {clean_title}")
                    report.append(f"**Score:** {item['score']:.4f}")
                    report.append(f"- {item['text']}\n")

            await self.emit_message("\n".join(report))
            await self.emit_status("success", f"Found {len(results)} {result_desc}", True)

        except Exception as e:
            logger.error(f"Error during unified query ({search_type}): {e}")
            await self.emit_message(f"❌ Error during search: {e}")
            await self.emit_status("error", "Search failed", True)

    async def handle_generate(self, content_before: str, search_type: str = "core", top_n: int = 20):
        """Handle the /generate command."""
        if not content_before.strip():
            await self.emit_message("⚠️ Please provide a research topic before `/generate` (e.g., `Impact of LLMs on medicine/generate`).")
            return

        if not self.valves.EMBEDDING_API_KEY:
            await self.emit_message("⚠️ `EMBEDDING_API_KEY` is not set in Valves. It is required for semantic search.")
            return

        await self.emit_message(f"🚀 Starting deep research agent flow ({search_type}, top_{top_n}). This involves multiple rounds of LLM processing and semantic search. Please wait... \n")
        await self.emit_status("info", "Planning research queries...", False)

        try:
            # Step 1: Planning and Searching
            context_items = await self.generate_research_context(
                user_request=content_before,
                valves=self.valves,
                request=self.__request__,
                user=self.__user__,
                prompt_dir=self.PROMPT_PATH,
                __event_emitter__=self.__event_emitter__,
                search_type=search_type,
                top_n=top_n
            )

            if not context_items:
                await self.emit_message("🔍 No relevant information found in the knowledge base.")
                await self.emit_status("info", "No relevant content", True)
                return

            await self.emit_status("info", f"Search complete. Found {len(context_items)} relevant sources. Starting synthesis...", False)

            # Step 2: Synthesis
            final_report = await self.synthesize_research_report(
                user_request=content_before,
                context_items=context_items,
                valves=self.valves,
                request=self.__request__,
                user=self.__user__,
                prompt_dir=self.PROMPT_PATH
            )

            await self.emit_message(final_report)
            await self.emit_status("success", "Research report generated successfully", True)
        except Exception as e:
            logger.error(f"Error during generate flow: {e}")
            await self.emit_message(f"❌ Error during research generation: {e}")
            await self.emit_status("error", "Generation failed", True)

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __request__=None,
    ) -> str:
        self.__event_emitter__ = __event_emitter__
        self.__user__ = User(**__user__)
        self.__request__ = __request__

        messages = body.get("messages", [])
        if not messages:
            return ""

        raw_message = messages[-1].get("content", "")

        # Skip special tasks (title generation, etc.)
        if __task__ and __task__ != TASKS.DEFAULT:
            return ""

        # Handle the new /generate command format specially
        # XXX {sumr/abs/origin/ref/litr}/{number}/generate
        if raw_message.strip().lower().endswith("/generate"):
            # Default values
            search_type = "sumr"
            top_n = 20

            # Use regex to extract type and number
            match = re.search(r"^(.*?)\s*(?:(sumr|abs|origin|ref|litr)/)?\s*(?:(\d+)/)?generate$", raw_message.strip(), re.IGNORECASE | re.DOTALL)

            if match:
                content_before = match.group(1).strip()
                extracted_type = match.group(2)
                extracted_n = match.group(3)

                if extracted_type:
                    search_type = extracted_type.lower()

                if extracted_n:
                    top_n = int(extracted_n)

                await self.handle_generate(content_before, search_type, top_n)
                return ""

        # Handle the unified /query command format
        # XXX {file/sumr/abs/origin/ref/litr}/{number}/query
        if raw_message.strip().lower().endswith("/query"):
            # Default values
            search_type = "file"
            top_n = 20

            # Use regex to extract type and number
            match = re.search(r"^(.*?)\s*(?:(file|sumr|abs|origin|ref|litr)/)?\s*(?:(\d+)/)?query$", raw_message.strip(), re.IGNORECASE | re.DOTALL)

            if match:
                content_before = match.group(1).strip()
                extracted_type = match.group(2)
                extracted_n = match.group(3)

                if extracted_type:
                    search_type = extracted_type.lower()

                if extracted_n:
                    top_n = int(extracted_n)

                await self.handle_query(content_before, search_type, top_n)
                return ""

        # Find the LAST forward slash to determine the command and the content before it
        last_slash_index = raw_message.rfind("/")

        if last_slash_index == -1:
            # No command found
            await self.emit_message(f"**PanKnode** is a command-driven tool. Please start your message with a existing `/` command.\n\nType `/help` to see what I can do for you.")
            return ""

        content_before = raw_message[:last_slash_index].strip()
        command = raw_message[last_slash_index:].strip().lower()

        # Command Dispatcher
        if command == "/help":
            await self.handle_help()
        elif command == "/debug":
            await self.handle_debug()
        elif command == "/re":
            await self.handle_re(content_before)
        elif command == "/addfile":
            await self.handle_addfile(content_before)
        elif command == "/scanfiles":
            await self.handle_scanfiles()
        elif command == "/embedding":
            await self.handle_embedding()
        elif command == "/list":
            await self.handle_list()
        elif command == "/cat":
            await self.handle_cat(content_before)
        elif command == "/search":
            await self.handle_search(content_before)
        else:
            # Unknown command
            await self.emit_message(f"**PanKnode** is a command-driven tool. Please start your message with a existing `/` command.\n\nType `/help` to see what I can do for you.")

        return ""
