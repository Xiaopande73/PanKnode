import os
import sqlite3
import json
import logging
import httpx
import numpy as np
import asyncio
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("PanKnode")

class EmbeddingManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database with separate columns for all content sections."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kb_embeddings (
                    filename TEXT PRIMARY KEY,
                    abs_vectors BLOB,
                    abs_texts BLOB,
                    sum_vectors BLOB,
                    sum_texts BLOB,
                    litr_vectors BLOB,
                    litr_texts BLOB,
                    ref_vectors BLOB,
                    ref_texts BLOB,
                    original_vectors BLOB,
                    original_texts BLOB,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def split_text_by_words(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into chunks based on word count with overlap."""
        words = text.split()
        if not words:
            return []

        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i : i + chunk_size]
            chunks.append(" ".join(chunk_words))
            if i + chunk_size >= len(words):
                break
            i += (chunk_size - overlap)
        return chunks

    async def get_embeddings(self, texts: List[str], base_url: str, api_key: str, model: str) -> List[List[float]]:
        """Fetch embeddings from OpenRouter API."""
        if not texts:
            return []

        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        url += "/embeddings"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "input": [t.replace("\n", " ") for t in texts] # Clean newlines for API
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]

    def parse_markdown(self, content: str) -> Dict[str, str]:
        """Parse the custom markdown format into specific sections for precise embedding."""
        # Use KBManager's robust extraction logic from inherited methods
        sections = {
            "abs": self.extract_section_content(content, "Abstract"),
            "sum": self.extract_section_content(content, "Summary"),
            "litr": self.extract_section_content(content, "Literature Review"),
            "ref": self.extract_section_content(content, "Reference"),
            "original": self.extract_section_content(content, "Original Content")
        }
        return sections

    def get_processed_files(self) -> List[str]:
        """Get list of filenames already processed in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT filename FROM kb_embeddings")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get processed files: {e}")
            return []

    async def process_file(
        self,
        filename: str,
        file_path: str,
        valves: Any,
        api_key: str
    ) -> bool:
        """Process file: split into granular sections, embed each, and store in DB."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            sections = self.parse_markdown(content)

            # 1. Generate Chunks for each section
            abs_chunks = self.split_text_by_words(sections["abs"], valves.SUMMARY_CHUNK_SIZE, valves.SUMMARY_OVERLAP)
            sum_chunks = self.split_text_by_words(sections["sum"], valves.SUMMARY_CHUNK_SIZE, valves.SUMMARY_OVERLAP)
            lit_chunks = self.split_text_by_words(sections["litr"], valves.LITR_CHUNK_SIZE, valves.LITR_OVERLAP)
            ref_chunks = self.split_text_by_words(sections["ref"], valves.REF_CHUNK_SIZE, valves.REF_OVERLAP)
            original_chunks = self.split_text_by_words(sections["original"], valves.ORIGINAL_CHUNK_SIZE, valves.ORIGINAL_OVERLAP)

            # 2. Fetch Vectors in parallel
            abs_v, sum_v, lit_v, ref_v, orig_v = await asyncio.gather(
                self.get_embeddings(abs_chunks, valves.EMBEDDING_BASE_URL, api_key, valves.EMBEDDING_MODEL),
                self.get_embeddings(sum_chunks, valves.EMBEDDING_BASE_URL, api_key, valves.EMBEDDING_MODEL),
                self.get_embeddings(lit_chunks, valves.EMBEDDING_BASE_URL, api_key, valves.EMBEDDING_MODEL),
                self.get_embeddings(ref_chunks, valves.EMBEDDING_BASE_URL, api_key, valves.EMBEDDING_MODEL),
                self.get_embeddings(original_chunks, valves.EMBEDDING_BASE_URL, api_key, valves.EMBEDDING_MODEL)
            )

            # 3. Store in DB
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO kb_embeddings
                       (filename, abs_vectors, abs_texts, sum_vectors, sum_texts,
                        litr_vectors, litr_texts, ref_vectors, ref_texts,
                        original_vectors, original_texts, processed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                    (
                        filename,
                        json.dumps(abs_v).encode('utf-8'),
                        sections["abs"].encode('utf-8'), # Store full text string
                        json.dumps(sum_v).encode('utf-8'),
                        sections["sum"].encode('utf-8'), # Store full text string
                        json.dumps(lit_v).encode('utf-8'),
                        sections["litr"].encode('utf-8'), # Store full text string
                        json.dumps(ref_v).encode('utf-8'),
                        json.dumps(ref_chunks).encode('utf-8'),
                        json.dumps(orig_v).encode('utf-8'),
                        json.dumps(original_chunks).encode('utf-8')
                    )
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to process embedding for {filename}: {e}")
            return False

    async def query_top_files(
        self,
        query: str,
        top_n: int,
        valves: Any,
        api_key: str
    ) -> List[Tuple[str, float]]:
        """
        Calculates similarity with all core_vectors using numpy.
        Returns top_n filenames and their scores.
        """
        query_vectors = await self.get_embeddings([query], valves.EMBEDDING_BASE_URL, api_key, valves.EMBEDDING_MODEL)
        if not query_vectors:
            return []
        query_v = np.array(query_vectors[0])
        query_norm = np.linalg.norm(query_v)
        if query_norm == 0:
            return []

        # Default to summary vectors
        vector_col = "sum_vectors"

        results = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"SELECT filename, {vector_col} FROM kb_embeddings")
            for row in cursor:
                filename = row["filename"]
                v_data = row[vector_col]

                if not v_data:
                    continue

                vs = json.loads(v_data.decode('utf-8'))
                matrix = np.array(vs)
                dots = np.dot(matrix, query_v)
                norms = np.linalg.norm(matrix, axis=1)
                denom = norms * query_norm
                similarities = np.where(denom != 0, dots / denom, 0)

                if similarities.size == 0:
                    continue

                max_sim = float(np.max(similarities))
                results.append((filename, max_sim))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    async def query_top_origin_chunks(
        self,
        query: str,
        top_n: int,
        valves: Any,
        api_key: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top N relevant chunks from 'Original Content' sections.
        Fetches vectors and texts in a single SQL pass.
        """
        query_vectors = await self.get_embeddings([query], valves.EMBEDDING_BASE_URL, api_key, valves.EMBEDDING_MODEL)
        if not query_vectors:
            return []
        query_v = np.array(query_vectors[0])
        query_norm = np.linalg.norm(query_v)
        if query_norm == 0:
            return []

        candidates = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT filename, original_vectors, original_texts FROM kb_embeddings")
            for row in cursor:
                filename = row["filename"]
                v_data = row["original_vectors"]
                t_data = row["original_texts"]

                if not v_data or not t_data:
                    continue

                vs = json.loads(v_data.decode('utf-8'))
                ts = json.loads(t_data.decode('utf-8')) # This is already a list of chunks

                matrix = np.array(vs)
                dots = np.dot(matrix, query_v)
                norms = np.linalg.norm(matrix, axis=1)
                denom = norms * query_norm
                similarities = np.where(denom != 0, dots / denom, 0)

                if similarities.size == 0:
                    continue

                best_idx = int(np.argmax(similarities))
                max_sim = float(similarities[best_idx])

                if best_idx < len(ts):
                    candidates.append({
                        "filename": filename,
                        "score": max_sim,
                        "text": ts[best_idx]
                    })

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_n]

    async def query_top_sumr(
        self,
        query: str,
        top_n: int,
        valves: Any,
        api_key: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top N files most relevant to the query based on Summary.
        Returns the full content of that section directly from the DB.
        """
        query_vectors = await self.get_embeddings([query], valves.EMBEDDING_BASE_URL, api_key, valves.EMBEDDING_MODEL)
        if not query_vectors:
            return []
        query_v = np.array(query_vectors[0])
        query_norm = np.linalg.norm(query_v)
        if query_norm == 0:
            return []

        candidates = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT filename, sum_vectors, sum_texts FROM kb_embeddings")
            for row in cursor:
                filename = row["filename"]
                v_data = row["sum_vectors"]
                t_data = row["sum_texts"]

                if not v_data:
                    continue

                vs = json.loads(v_data.decode('utf-8'))
                matrix = np.array(vs)
                dots = np.dot(matrix, query_v)
                norms = np.linalg.norm(matrix, axis=1)
                denom = norms * query_norm
                similarities = np.where(denom != 0, dots / denom, 0)

                if similarities.size == 0:
                    continue

                max_sim = float(np.max(similarities))

                candidates.append({
                    "filename": filename,
                    "score": max_sim,
                    "text": t_data.decode('utf-8') if t_data else ""
                })

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_n]

    async def query_top_abs(
        self,
        query: str,
        top_n: int,
        valves: Any,
        api_key: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top N files most relevant to the query based on Abstract.
        Returns the full content of that section directly from the DB.
        """
        query_vectors = await self.get_embeddings([query], valves.EMBEDDING_BASE_URL, api_key, valves.EMBEDDING_MODEL)
        if not query_vectors:
            return []
        query_v = np.array(query_vectors[0])
        query_norm = np.linalg.norm(query_v)
        if query_norm == 0:
            return []

        candidates = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT filename, abs_vectors, abs_texts FROM kb_embeddings")
            for row in cursor:
                filename = row["filename"]
                v_data = row["abs_vectors"]
                t_data = row["abs_texts"]

                if not v_data:
                    continue

                vs = json.loads(v_data.decode('utf-8'))
                matrix = np.array(vs)
                dots = np.dot(matrix, query_v)
                norms = np.linalg.norm(matrix, axis=1)
                denom = norms * query_norm
                similarities = np.where(denom != 0, dots / denom, 0)

                if similarities.size == 0:
                    continue

                max_sim = float(np.max(similarities))

                candidates.append({
                    "filename": filename,
                    "score": max_sim,
                    "text": t_data.decode('utf-8') if t_data else ""
                })

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_n]

    async def query_top_litr(
        self,
        query: str,
        top_n: int,
        valves: Any,
        api_key: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top N files based on Literature Review content.
        Returns the full content of the Literature Review section directly from DB.
        """
        query_vectors = await self.get_embeddings([query], valves.EMBEDDING_BASE_URL, api_key, valves.EMBEDDING_MODEL)
        if not query_vectors:
            return []
        query_v = np.array(query_vectors[0])
        query_norm = np.linalg.norm(query_v)
        if query_norm == 0:
            return []

        candidates = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT filename, litr_vectors, litr_texts FROM kb_embeddings")
            for row in cursor:
                filename = row["filename"]
                v_data = row["litr_vectors"]
                t_data = row["litr_texts"]

                if not v_data:
                    continue

                vs = json.loads(v_data.decode('utf-8'))
                matrix = np.array(vs)
                dots = np.dot(matrix, query_v)
                norms = np.linalg.norm(matrix, axis=1)
                denom = norms * query_norm
                similarities = np.where(denom != 0, dots / denom, 0)

                if similarities.size == 0:
                    continue

                max_sim = float(np.max(similarities))

                candidates.append({
                    "filename": filename,
                    "score": max_sim,
                    "text": t_data.decode('utf-8') if t_data else ""
                })

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_n]

    async def query_top_ref_chunks(
        self,
        query: str,
        top_n: int,
        valves: Any,
        api_key: str
    ) -> List[Dict[str, Any]]:
        """
        Global Top N search across all reference chunks.
        Fetches vectors and texts in a single SQL pass.
        """
        query_vectors = await self.get_embeddings([query], valves.EMBEDDING_BASE_URL, api_key, valves.EMBEDDING_MODEL)
        if not query_vectors:
            return []
        query_v = np.array(query_vectors[0])
        query_norm = np.linalg.norm(query_v)
        if query_norm == 0:
            return []

        candidates = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT filename, ref_vectors, ref_texts FROM kb_embeddings")
            for row in cursor:
                filename = row["filename"]
                v_data = row["ref_vectors"]
                t_data = row["ref_texts"]

                if not v_data or not t_data:
                    continue

                vs = json.loads(v_data.decode('utf-8'))
                ts = json.loads(t_data.decode('utf-8'))

                matrix = np.array(vs)
                dots = np.dot(matrix, query_v)
                norms = np.linalg.norm(matrix, axis=1)
                denom = norms * query_norm
                similarities = np.where(denom != 0, dots / denom, 0)

                for idx, sim in enumerate(similarities):
                    if idx < len(ts):
                        candidates.append({
                            "filename": filename,
                            "score": float(sim),
                            "text": ts[idx]
                        })

        # Global Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_n]
