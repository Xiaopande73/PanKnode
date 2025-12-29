import json
import logging
import re
import os
from datetime import datetime
from typing import Any, Dict
from open_webui.main import generate_chat_completions

logger = logging.getLogger("PanKnode")

class GenerateManager:
    """
    GenerateManager: Handles the two-round agent logic for generating academic summaries.
    """

    async def generate_research_context(
        self,
        user_request: str,
        valves: Any,
        request: Any,
        user: Any,
        prompt_dir: str,
        __event_emitter__: Any = None,
        search_type: str = "sumr",
        top_n: int = 30
    ) -> list[dict]:
        """
        Round 1: Plan queries and execute searches.
        Returns a list of context items.
        """
        # --- Round 1: Query Planning ---
        agent_prompt_path = os.path.join(prompt_dir, "gen_agent_prompt.md")
        agent_instruction = "You are a research agent. Output JSON queries."
        if os.path.exists(agent_prompt_path):
            with open(agent_prompt_path, "r", encoding="utf-8") as f:
                agent_instruction = f.read().replace("{{user_request}}", user_request)

        # Call LLM for Round 1
        round1_response = await self._call_llm(agent_instruction, user_request, valves.MODEL_ID, request, user)

        # Parse JSON
        queries_data = self._parse_json_safely(round1_response)
        if not queries_data or "queries" not in queries_data:
            logger.error(f"Failed to parse queries from Round 1: {round1_response}")
            return []

        queries = queries_data["queries"]
        if len(queries) > 5:
            queries = queries[:5]

        # --- Action: Execute Queries ---
        all_raw_results = []
        for q_item in queries:
            q_text = q_item if isinstance(q_item, str) else q_item.get("query", "")
            if not q_text:
                continue

            results = []
            if search_type == "sumr":
                results = await self.query_top_sumr(q_text, top_n, valves, valves.EMBEDDING_API_KEY)
            elif search_type == "abs":
                results = await self.query_top_abs(q_text, top_n, valves, valves.EMBEDDING_API_KEY)
            elif search_type == "litr":
                results = await self.query_top_litr(q_text, top_n, valves, valves.EMBEDDING_API_KEY)
            elif search_type == "origin":
                results = await self.query_top_origin_chunks(q_text, top_n, valves, valves.EMBEDDING_API_KEY)
            elif search_type == "ref":
                results = await self.query_top_ref_chunks(q_text, top_n, valves, valves.EMBEDDING_API_KEY)

            all_raw_results.extend(results)

        # Process results: Deduplicate and merge by filename
        merged_data = {}
        for res in all_raw_results:
            fname = res["filename"]
            txt = res["text"]
            if fname not in merged_data:
                merged_data[fname] = []
            if txt not in merged_data[fname]:
                merged_data[fname].append(txt)

        context_items = []
        for fname, texts in merged_data.items():
            combined_text = "\n\n".join(texts)
            context_item = {
                "id": len(context_items) + 1,
                "filename": fname,
                "text": combined_text
            }
            context_items.append(context_item)

            # Emit citation
            if __event_emitter__:
                await __event_emitter__({
                    "type": "citation",
                    "data": {
                        "document": [context_item["text"]],
                        "metadata": [
                            {
                                "date_accessed": datetime.now().isoformat(),
                                "source": context_item["filename"],
                                "type": "knowledge_base"
                            }
                        ],
                        "source": {
                            "name": context_item["filename"].replace(".md", "").replace("_", " "),
                            "url": f"file://{context_item['filename']}"
                        }
                    }
                })

        return context_items

    async def synthesize_research_report(
        self,
        user_request: str,
        context_items: list[dict],
        valves: Any,
        request: Any,
        user: Any,
        prompt_dir: str
    ) -> str:
        """
        Round 2: Synthesize academic summary and save report.
        """
        if not context_items:
            return "🔍 No relevant information found in the knowledge base."

        # Prepare context string
        formatted_context = ""
        for item in context_items:
            formatted_context += f"[{item['id']}] Source: {item['filename']}\nContent: {item['text']}\n\n"

        summary_prompt_path = os.path.join(prompt_dir, "gen_summary_prompt.md")
        summary_instruction = "Write an academic summary based on context."
        if os.path.exists(summary_prompt_path):
            with open(summary_prompt_path, "r", encoding="utf-8") as f:
                summary_instruction = f.read().replace("{{user_request}}", user_request).replace("{{context}}", formatted_context)

        # Call LLM for Round 2
        final_summary = await self._call_llm(summary_instruction, "Please generate the summary now.", valves.MODEL_ID, request, user)

        # Post-processing: Bibliography and Saving
        bibliography = "\n\n### 📚 Sources\n"
        seen_files = {}
        for item in context_items:
            fname = item["filename"]
            if fname not in seen_files:
                seen_files[fname] = item["id"]
                clean_name = fname.replace(".md", "").replace("_", " ")
                bibliography += f"{item['id']}. {clean_name} (`{fname}`)\n"

        if hasattr(self, 'gen_path') and self.gen_path:
            try:
                os.makedirs(self.gen_path, exist_ok=True)
                safe_request = re.sub(r'[^\w\s-]', '', user_request).strip().replace(' ', '_')[:30]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_filename = f"report_{timestamp}_{safe_request}.md"
                report_path = os.path.join(self.gen_path, report_filename)

                full_report = f"# Research Report: {user_request}\n\n{final_summary}{bibliography}"
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(full_report)
                logger.info(f"Full report saved to {report_path}")
            except Exception as e:
                logger.error(f"Failed to save full report: {e}")

        return final_summary

    async def run_generate_flow(
        self,
        user_request: str,
        valves: Any,
        request: Any,
        user: Any,
        prompt_dir: str,
        __event_emitter__: Any = None,
        search_type: str = "core",
        top_n: int = 20
    ) -> str:
        """
        Backward compatible wrapper for the two-round flow.
        """
        context_items = await self.generate_research_context(
            user_request, valves, request, user, prompt_dir, __event_emitter__, search_type, top_n
        )
        if not context_items:
            return "🔍 No relevant information found."

        return await self.synthesize_research_report(
            user_request, context_items, valves, request, user, prompt_dir
        )

    async def _call_llm(self, system_prompt: str, user_msg: str, model_id: str, request: Any, user: Any) -> str:
        form_data = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            "stream": False,
            "temperature": 0.3,
        }
        response = await generate_chat_completions(request, form_data, user=user)
        if response and "choices" in response:
            return response["choices"][0]["message"]["content"]
        return ""

    def _parse_json_safely(self, text: str) -> Dict:
        """Extract and parse JSON from LLM response."""
        try:
            # Look for JSON block
            match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            # Try parsing the whole text
            return json.loads(text.strip())
        except Exception:
            # Fallback: try to find anything that looks like { ... }
            try:
                match = re.search(r'(\{.*\})', text, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
            except Exception:
                return {}
        return {}
