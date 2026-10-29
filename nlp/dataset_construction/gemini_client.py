"""Gemini API client wrapper.

Supports both Google AI Studio (API key) and Vertex AI (gcloud auth) backends.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


class GeminiClient:
    """Wrapper around Google GenAI client (online + batch modes)."""

    def __init__(self, model_name: str, model_config: dict, batch_mode: bool = False):
        from google import genai

        self.model_name = model_name
        self.model_id = model_config["model_id"]
        self.batch_mode = batch_mode

        api_key = os.getenv("GOOGLE_API_KEY")
        project = os.getenv("GOOGLE_CLOUD_PROJECT")

        if api_key:
            self.client = genai.Client(api_key=api_key)
            self._backend = "ai_studio"
            print("  Using Google AI Studio (API key)")
        elif project:
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
            self._backend = "vertex_ai"
            print(f"  Using Vertex AI (project={project}, location={location})")
        else:
            raise ValueError(
                "No credentials found. Set one of:\n"
                "  GOOGLE_API_KEY=... (Google AI Studio)\n"
                "  GOOGLE_CLOUD_PROJECT=... (Vertex AI)\n"
                "Add to .env or set as environment variable."
            )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        max_output_tokens: int = 4096,
    ) -> str:
        """Generate content synchronously (online mode)."""
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        return response.text

    def generate_batch(
        self,
        requests: List[Dict],
        output_dir: Path,
        temperature: float = 1.0,
        max_output_tokens: int = 4096,
    ) -> str:
        """Submit a batch job (50% cheaper, async)."""
        from google.genai import types

        inline_requests = []
        for req in requests:
            inline_requests.append(
                types.BatchJobSource.InlineRequest(
                    request=types.GenerateContentRequest(
                        model=self.model_id,
                        contents=req["user_prompt"],
                        config=types.GenerateContentConfig(
                            system_instruction=req["system_prompt"],
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                        ),
                    )
                )
            )

        batch_job = self.client.batches.create(
            model=self.model_id,
            src=types.BatchJobSource(inline_requests=inline_requests),
        )

        job_id = batch_job.name.split("/")[-1]
        job_meta = {
            "job_name": batch_job.name,
            "model": self.model_name,
            "n_requests": len(requests),
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "SUBMITTED",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = output_dir / f"batch_job_{job_id}.json"
        meta_path.write_text(json.dumps(job_meta, indent=2))

        print(f"  Batch job submitted: {batch_job.name}")
        print(f"  Metadata saved to: {meta_path}")

        return batch_job.name

    def retrieve_batch(self, job_name: str) -> Optional[List[str]]:
        """Check batch job status and retrieve results if complete."""
        job = self.client.batches.get(name=job_name)
        print(f"  Batch job status: {job.state}")

        if job.state.name != "JOB_STATE_SUCCEEDED":
            if job.state.name == "JOB_STATE_FAILED":
                print("  ERROR: Batch job failed")
                return []
            return None

        # Jobs submitted with inline requests return their results inline.
        results = []
        for inlined in job.dest.inlined_responses or []:
            if inlined.response and inlined.response.candidates:
                results.append(inlined.response.text)
        return results
