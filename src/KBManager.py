import os
import re
import logging
from typing import Any
from open_webui.main import generate_chat_completions
from open_webui.retrieval.loaders.main import Loader

logger = logging.getLogger("PanKnode")

class KBManager:
    """
    KBManager: Handles core knowledge base operations.
    Current version focuses on text summarization and content processing.
    """

    async def reprocess_section(
        self,
        filename: str,
        re_type: str,
        output_path: str,
        prompt_path: str,
        model_id: str,
        request: Any,
        user: Any,
        max_retries: int = 10
    ) -> bool:
        """
        Reload a specific section of a markdown file using its Original Content.
        """
        try:
            # 1. Read existing file
            if not filename.endswith(".md"):
                filename += ".md"
            file_path = os.path.join(output_path, filename)
            if not os.path.exists(file_path):
                logger.error(f"File not found for reprocessing: {file_path}")
                return False

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 2. Extract Original Content
            original_content = self.extract_section_content(content, "Original Content")
            if not original_content:
                logger.error(f"Original Content not found in {filename}")
                return False

            # Helper to read prompt
            def read_prompt(fname):
                p_path = os.path.join(prompt_path, fname)
                if os.path.exists(p_path):
                    with open(p_path, "r", encoding="utf-8") as f:
                        return f.read()
                return ""

            new_content = ""
            if re_type == "absref":
                # Re-run Step 1: Extract Abstract & Reference (and Title)
                extract_prompt = read_prompt("kb_extract_prompt.md") or "Extract Abstract and Reference."
                for attempt in range(max_retries):
                    new_content = await self._call_llm_internal(extract_prompt, original_content, model_id, request, user)
                    if any(line.strip().startswith('#') for line in new_content.split('\n')):
                        break
                    logger.warning(f"Attempt {attempt + 1}: No title found in re-extraction.")

                if not new_content: return False

                # Replacement logic for absref is complex because it involves the # Title.
                # We assume the file structure starts with Step 1 content.
                # Find where Summary or Lit Review starts to preserve them.
                parts = re.split(r"(?m)^#+\s+(Summary|Literature Review)", content)
                # re.split with groups returns [before, group, before, group, ...]
                # structure: [absref_part, "Summary", summary_part, "Literature Review", lit_part, ...]

                # A safer approach: build the new file content by stitching parts.
                summary_val = self.extract_section_content(content, "Summary")
                lit_val = self.extract_section_content(content, "Literature Review")

                combined = new_content.strip()
                if "## Reference" in combined:
                    header_abs, ref_part = combined.split("## Reference", 1)
                    combined = f"{header_abs.strip()}\n\n## Summary\n{summary_val}\n\n## Literature Review\n{lit_val}\n\n## Reference\n{ref_part.strip()}"
                else:
                    combined = f"{combined}\n\n## Summary\n{summary_val}\n\n## Literature Review\n{lit_val}"

                new_full_content = f"{combined}\n\n## Original Content\n\n{original_content}"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_full_content)
                return True

            elif re_type == "sumr":
                # Re-run Step 2: Summary
                summary_prompt = read_prompt("kb_summary_prompt.md") or "Provide a long 1000-word summary in one paragraph."
                for attempt in range(max_retries):
                    new_content = await self._call_llm_internal(summary_prompt, original_content, model_id, request, user)
                    if len(new_content.split()) >= 300:
                        break

                if not new_content: return False

                # Replace Summary section
                pattern = r"(?m)^#+\s+Summary\s*$(.*?)(?=^#+\s+|\Z)"
                replacement = f"## Summary\n{new_content.strip()}\n\n"

                # Check if it exists to replace, or append
                if re.search(pattern, content, re.DOTALL | re.IGNORECASE):
                    new_full_content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)
                else:
                    # Fallback if Summary didn't exist
                    new_full_content = content.replace("## Original Content", f"{replacement}## Original Content")

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_full_content)
                return True

            elif re_type == "litr":
                # Re-run Step 3: Literature Review
                lit_prompt = read_prompt("kb_lit_review_prompt.md") or "Summarize literature review with (Author et al. Year) citations."
                for attempt in range(max_retries):
                    new_content = await self._call_llm_internal(lit_prompt, original_content, model_id, request, user)
                    if len(new_content.split()) >= 300:
                        break

                if not new_content: return False

                # Replace Literature Review section
                pattern = r"(?m)^#+\s+Literature Review\s*$(.*?)(?=^#+\s+|\Z)"
                replacement = f"## Literature Review\n{new_content.strip()}\n\n"

                if re.search(pattern, content, re.DOTALL | re.IGNORECASE):
                    new_full_content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)
                else:
                    new_full_content = content.replace("## Original Content", f"{replacement}## Original Content")

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_full_content)
                return True

            return False
        except Exception as e:
            logger.error(f"Error in reprocess_section: {e}")
            return False

    async def summarize_content(
        self,
        content: str,
        prompt_path: str, # This now points to the directory containing the prompts
        model_id: str,
        request: Any,
        user: Any,
        output_path: str = None,
        min_length: int = 500,
        original_content: str = None
    ) -> tuple[str, str]:
        """
        Refactored multi-step synthesis with existence check.
        """
        if not output_path and hasattr(self, 'kb_path'):
            output_path = self.kb_path

        if len(content.strip()) < min_length:
            logger.warning(f"Content too short for summarization: {len(content)} characters")
            return "TOO_SHORT", ""

        # Helper to read prompt
        def read_prompt(filename):
            p_path = os.path.join(prompt_path, filename)
            if os.path.exists(p_path):
                with open(p_path, "r", encoding="utf-8") as f:
                    return f.read()
            return ""

        try:
            # Step 1: Extract Abstract & Reference (This also contains the Title)
            extract_prompt = read_prompt("kb_extract_prompt.md") or "Extract Abstract and Reference."

            # Retry logic for title extraction
            res_extract = ""
            max_retries = 10
            for attempt in range(max_retries):
                res_extract = await self._call_llm_internal(extract_prompt, content, model_id, request, user)
                if any(line.strip().startswith('#') for line in res_extract.split('\n')):
                    break
                logger.warning(f"Attempt {attempt + 1}: No title found in LLM extraction. Retrying...")

            # --- Existence Check ---
            predicted_path = self.predict_file_path(res_extract, output_path)
            if predicted_path and os.path.exists(predicted_path):
                logger.info(f"File already exists: {predicted_path}. Skipping remaining steps.")
                return "ALREADY_EXISTS", predicted_path

            # Step 2: 1000-word Summary (One Paragraph)
            summary_prompt = read_prompt("kb_summary_prompt.md") or "Provide a long 1000-word summary in one paragraph."
            res_summary = ""
            for attempt in range(max_retries):
                res_summary = await self._call_llm_internal(summary_prompt, content, model_id, request, user)
                word_count = len(res_summary.split())
                if word_count >= 300:
                    break
                logger.warning(f"Attempt {attempt + 1}: Summary too short ({word_count} words). Retrying...")

            # Step 3: Literature Review (with citations)
            lit_prompt = read_prompt("kb_lit_review_prompt.md") or "Summarize literature review with (Author et al. Year) citations."
            res_lit = ""
            for attempt in range(max_retries):
                res_lit = await self._call_llm_internal(lit_prompt, content, model_id, request, user)
                word_count = len(res_lit.split())
                if word_count >= 300:
                    break
                logger.warning(f"Attempt {attempt + 1}: Literature Review too short ({word_count} words). Retrying...")

            # Combine
            combined_md = res_extract
            ref_marker = "## Reference"
            if ref_marker in combined_md:
                parts = combined_md.split(ref_marker)
                summary_block = res_summary if "## Summary" in res_summary else f"## Summary\n{res_summary}"
                lit_block = res_lit if "## Literature Review" in res_lit else f"## Literature Review\n{res_lit}"
                combined_md = f"{parts[0].strip()}\n\n{summary_block.strip()}\n\n{lit_block.strip()}\n\n{ref_marker}\n{parts[1].strip()}"
            else:
                combined_md += f"\n\n{res_summary}\n\n{res_lit}"

            saved_path = ""
            if output_path and combined_md:
                saved_path = await self.save_summary_to_file(combined_md, output_path, original_content)

            return combined_md, saved_path

        except Exception as e:
            logger.error(f"Error during multi-step summarization: {e}")
            raise e

    def _sanitize_filename(self, title: str) -> str:
        """Sanitize title to create a valid filename."""
        filename = "".join([c for c in title if c.isalnum() or c in (' ', '.', '_', '-')]).rstrip()
        filename = filename.replace(' ', '_')
        return filename

    def predict_file_path(self, summary_text: str, output_path: str) -> str:
        """Predict the filename based on the title in the summary text."""
        if not output_path:
            return ""
        lines = summary_text.split('\n')
        title = ""
        for line in lines:
            if line.strip().startswith('#'):
                title = line.strip().lstrip('#').strip()
                break
        if not title:
            return ""

        filename = self._sanitize_filename(title)
        if not filename:
            return ""
        return os.path.join(output_path, f"{filename}.md")

    async def _call_llm_internal(self, system_prompt: str, user_content: str, model_id: str, request: Any, user: Any) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        form_data = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "temperature": 0.3,
        }
        response = await generate_chat_completions(request, form_data, user=user)
        if response and "choices" in response:
            return response["choices"][0]["message"]["content"]
        return ""

    async def save_summary_to_file(self, summary: str, output_path: str, original_content: str = None) -> str:
        """
        Extract title from summary and save it to output_path.
        Appends original_content if provided.
        Returns the path of the saved file if successful, empty string otherwise.
        """
        try:
            # Extract title (first line starting with #)
            lines = summary.split('\n')
            title = ""
            for line in lines:
                if line.strip().startswith('#'):
                    title = line.strip().lstrip('#').strip()
                    break

            if not title:
                title = "Untitled_Summary"

            # Sanitize filename
            filename = self._sanitize_filename(title)
            if not filename:
                filename = "summary"

            file_path = os.path.join(output_path, f"{filename}.md")

            # Ensure directory exists
            os.makedirs(output_path, exist_ok=True)

            # Prepare content to write
            content_to_write = summary
            if original_content:
                content_to_write += f"\n\n## Original Content\n\n{original_content}"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content_to_write)

            logger.info(f"Summary saved to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving summary to file: {e}")
            return ""

    def extract_context_content(self, text: str) -> str:
        """
        Extract content between <context> and </context> tags.
        Replaces all newlines with spaces.
        """
        pattern = r"<context>(.*?)</context>"
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            return ""

        # Join all matches if there are multiple, or just take the first one
        combined_content = " ".join(matches)
        return self.clean_text(combined_content)

    def clean_text(self, text: str) -> str:
        """
        Replace all newlines with spaces and remove multiple spaces.
        """
        # Replace newlines with spaces
        processed_content = text.replace('\n', ' ').replace('\r', ' ')
        # Remove multiple spaces
        processed_content = re.sub(r'\s+', ' ', processed_content).strip()
        return processed_content

    async def list_files(self, directory: str = None) -> list[str]:
        """
        List all .md files in the specified directory or self.kb_path.
        """
        if not directory and hasattr(self, 'kb_path'):
            directory = self.kb_path

        if not directory or not os.path.exists(directory):
            return []
        try:
            files = [f for f in os.listdir(directory) if f.endswith('.md')]
            return sorted(files)
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []

    async def get_file_content(self, directory: str = None, filename: str = "") -> str:
        """
        Read and return the content of a specific file in the directory or self.kb_path.
        """
        if not directory and hasattr(self, 'kb_path'):
            directory = self.kb_path

        if not directory:
            return ""

        # Ensure the filename has .md extension if not provided
        if not filename.endswith('.md'):
            filename += '.md'

        file_path = os.path.join(directory, filename)
        if not os.path.exists(file_path):
            return ""

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
            return ""

    def extract_section_content(self, md_content: str, section_name: str) -> str:
        """
        Extract content under a specific heading (e.g., #summary or #Abstract).
        Supports case-insensitive matching and both # and ## styles.
        """
        # Normalize section name for regex (escape special chars, although usually just word chars)
        escaped_name = re.escape(section_name.lstrip('#'))

        # Pattern matches:
        # 1. A heading line: any number of # followed by space and the section name
        # 2. All content until the next heading of the same or higher level, or end of file
        # We use re.IGNORECASE to handle variations
        pattern = rf"(?m)^#+\s+{escaped_name}\s*$(.*?)(?=^#+\s+|\Z)"

        match = re.search(pattern, md_content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback for some common patterns if exact match fails
        if section_name.lower() in ["summary", "#summary"]:
            # If #summary not found, maybe it's the first section of the file
            lines = md_content.split('\n')
            if lines and lines[0].strip().lower().startswith('# summary'):
                 # It's already there but maybe pattern failed, try a simpler split
                 parts = re.split(r'(?m)^#+\s+', md_content)
                 for part in parts:
                     if part.strip().lower().startswith('summary'):
                         return part[7:].strip()

        return ""

    async def search_files(self, directory: str = None, query: str = "") -> list[str]:
        """
        Search for files in the directory (or self.kb_path) that contain the query string in their name (case-insensitive).
        """
        if not directory and hasattr(self, 'kb_path'):
            directory = self.kb_path

        if not directory or not os.path.exists(directory):
            return []
        try:
            query = query.lower()
            files = [f for f in os.listdir(directory) if f.endswith('.md') and query in f.lower()]
            return sorted(files)
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return []

    async def scan_pdf_directory(self, directory: str) -> list[str]:
        """
        Scan directory for PDF files.
        """
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            return []
        try:
            files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
            return sorted(files)
        except Exception as e:
            logger.error(f"Error scanning PDF directory: {e}")
            return []

    async def process_pdf_to_text(self, file_path: str) -> str:
        """
        Convert PDF file to text using Open WebUI Loader.
        """
        filename = os.path.basename(file_path)
        try:
            loader = Loader(engine="")
            docs = loader.load(
                filename=filename,
                file_content_type="application/pdf",
                file_path=file_path
            )
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {e}")
            raise e
