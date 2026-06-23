import asyncio
import re
import sys
import tempfile
from pathlib import Path

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from edge_tts import Communicate
from tqdm import tqdm

CROP_START_PATTERN = "Restore scroll position".lower()
CROP_END_PATTERN = "Share to your friends".lower()
CHUNK_COUNT = 16
VOICE = "en-GB-RyanNeural"
WORD_RE = re.compile(r"\S+")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def split_markdown(markdown: list[str], chunk_count: int = CHUNK_COUNT) -> list[str]:
    total_words = sum(count_words(line) for line in markdown)
    if total_words == 0:
        return []

    actual_chunk_count = min(chunk_count, total_words)
    target_words = -(-total_words // actual_chunk_count)
    chunks: list[str] = []
    current_lines: list[str] = []
    current_words = 0

    for line in markdown:
        current_lines.append(line)
        current_words += count_words(line)

        if current_words >= target_words and len(chunks) < actual_chunk_count - 1:
            chunks.append("\n".join(current_lines).strip())
            current_lines = []
            current_words = 0

    if current_lines:
        chunks.append("\n".join(current_lines).strip())

    return chunks


async def text_to_speech_chunk(
    index: int,
    chunk_total: int,
    text: str,
    output_path: Path,
    progress: tqdm,
    progress_lock: asyncio.Lock,
) -> None:
    tts = Communicate(text=text, voice=VOICE, boundary="WordBoundary")

    with output_path.open("wb") as audio_file:
        async for chunk in tts.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                async with progress_lock:
                    if progress.total is None or progress.n < progress.total:
                        progress.update(1)

    async with progress_lock:
        progress.set_postfix_str(f"chunk {index + 1}/{chunk_total}", refresh=False)


async def text_to_speech(markdown: list[str], output_path: Path) -> None:
    chunks = split_markdown(markdown)
    total_words = sum(count_words(chunk) for chunk in chunks)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        chunk_paths = [
            temp_path / f"output.part{index:02d}.mp3" for index in range(len(chunks))
        ]

        progress_lock = asyncio.Lock()
        with tqdm(
            total=total_words, unit="word", desc="TTS", dynamic_ncols=True
        ) as bar:
            await asyncio.gather(
                *(
                    text_to_speech_chunk(
                        index,
                        len(chunks),
                        chunk,
                        chunk_paths[index],
                        bar,
                        progress_lock,
                    )
                    for index, chunk in enumerate(chunks)
                )
            )

            if bar.n < total_words:
                bar.update(total_words - bar.n)

        with output_path.open("wb") as combined_audio:
            for chunk_path in chunk_paths:
                combined_audio.write(chunk_path.read_bytes())


async def main():
    browser_config = BrowserConfig()
    run_config = CrawlerRunConfig()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=sys.argv[1], config=run_config)
        markdown = result.markdown.split("\n")

    crop_start = next(
        (i for i, line in enumerate(markdown) if CROP_START_PATTERN in line.lower()), 0
    )
    crop_end = next(
        (i for i, line in enumerate(markdown) if CROP_END_PATTERN in line.lower()), -1
    )
    markdown = markdown[crop_start + 1 : crop_end]

    with open("output.txt", "w") as f:
        f.write("\n".join(markdown))

    await text_to_speech(markdown, Path("output.mp3"))


if __name__ == "__main__":
    asyncio.run(main())
