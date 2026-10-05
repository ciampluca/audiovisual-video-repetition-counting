from __future__ import annotations

from typing import Any

from lvlm_video_repetition_benchmark.video import VideoSample


class VLLMVideoClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout_seconds: float,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout_seconds,
                max_retries=0,
            )
        self.client = client
        self.model_name = model_name

    def generate(
        self,
        video: VideoSample,
        prompt: str,
        seed: int,
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": video.video_data_uri},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            extra_body={"media_io_kwargs": {"video": video.media_io_kwargs()}},
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        return "".join(
            part.text for part in content if getattr(part, "text", None) is not None
        )