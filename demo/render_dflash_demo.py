#!/usr/bin/env python3
"""Render a DFlash/DoMinO demo video from existing benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
HF_ANSWERS = ROOT / "outputs/hf_20260528_083842/gsm8k_t0.0_answers.jsonl"
DEMO_QUESTION_ID = 1
DEMO_PROMPT = (
    "A cat eats nine sausages in 30 minutes. A dog can eat the same number "
    "of sausages in 2/3 the amount of time the cat takes. Calculate the "
    "average time the two take to eat the sausages."
)

WIDTH = 1920
HEIGHT = 1080
FPS = 30
DURATION_S = 16.5
BASELINE_REVEAL_S = 15.6
DEMO_OUTPUT_TOKENS = 333

BG = (9, 11, 15)
SURFACE = (17, 20, 26)
PANEL_BG = (18, 21, 27)
PANEL_HEADER = (23, 27, 34)
TEXT = (232, 235, 239)
MUTED = (142, 149, 160)
DIM = (84, 91, 103)
PROMPT_BORDER = (219, 205, 84)
PROMPT_TITLE = (239, 226, 91)


@dataclass
class Method:
    title: str
    color: tuple[int, int, int]
    text: str
    tok_s: float
    badge: str
    reveal_s: float


# Paper table values: Qwen3-8B, GSM8K, temperature=0, concurrency=1.
PAPER_DEMO_METRICS = [
    ("Autoregressive Qwen3-8B", (238, 68, 74), 92.46, "1.0x"),
    ("EAGLE-3", (52, 142, 255), 190.83, "2.1x"),
    ("DFlash", (45, 196, 92), 382.65, "4.1x"),
    ("Domino", (43, 176, 220), 533.77, "5.8x"),
]


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    return ImageFont.truetype(str(font_dir / name), size)


FONT_MONO = load_font("DejaVuSansMono.ttf", 20)
FONT_MONO_BOLD = load_font("DejaVuSansMono-Bold.ttf", 20)
FONT_LABEL = load_font("DejaVuSans-Bold.ttf", 24)
FONT_LABEL_SMALL = load_font("DejaVuSans-Bold.ttf", 21)
FONT_STAT = load_font("DejaVuSans-Bold.ttf", 27)
FONT_STAT_SMALL = load_font("DejaVuSans-Bold.ttf", 23)
FONT_BADGE = load_font("DejaVuSans-Bold.ttf", 25)
FONT_PROMPT = load_font("DejaVuSansMono-Bold.ttf", 22)
FONT_FOOTER = load_font("DejaVuSansMono.ttf", 18)
FONT_STATUS = load_font("DejaVuSansMono-Bold.ttf", 18)


def read_hf_answers(path: Path) -> tuple[str, str, float, str, float]:
    prompt = (
        f"{DEMO_PROMPT}\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    fallback_base = (
        "Let's solve this step by step.\n\n"
        "The cat takes 30 minutes.\n"
        "The dog takes 2/3 * 30 = 20 minutes.\n\n"
        "Average time = (30 + 20) / 2 = 25 minutes.\n\n"
        "Final Answer: boxed{25}"
    )
    fallback_domino = fallback_base
    if not path.exists():
        return prompt, fallback_base, 26.9, fallback_domino, 252.8

    with path.open() as f:
        record = None
        for line in f:
            candidate = json.loads(line)
            if int(candidate.get("question_id", -1)) == DEMO_QUESTION_ID:
                record = candidate
                break
        if record is None:
            raise ValueError(f"Could not find question_id={DEMO_QUESTION_ID} in {path}")
    choices = record["choices"]
    baseline = choices[0]
    domino = choices[1]
    base_text = baseline["turns"][0]
    domino_text = domino["turns"][0]
    base_tok_s = baseline["new_tokens"][0] / baseline["wall_time"][0]
    domino_tok_s = domino["new_tokens"][0] / domino["wall_time"][0]
    return prompt, base_text, base_tok_s, domino_text, domino_tok_s


def sanitize(text: str) -> str:
    replacements = {
        "\u2014": "---",
        "\u2013": "-",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2705": "[OK]",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def wrap_text(text: str, width_chars: int) -> list[str]:
    lines: list[str] = []
    for raw_line in sanitize(text).splitlines():
        if not raw_line:
            lines.append("")
            continue
        parts = wrap(
            raw_line,
            width=width_chars,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
        )
        lines.extend(parts or [""])
    return lines


def text_size(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont
) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def mix(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    return tuple(int(a[i] * (1.0 - t) + b[i] * t) for i in range(3))


def draw_background(draw: ImageDraw.ImageDraw) -> None:
    for x in range(0, WIDTH, 48):
        draw.line((x, 0, x, HEIGHT), fill=(13, 16, 22), width=1)
    for y in range(0, HEIGHT, 48):
        draw.line((0, y, WIDTH, y), fill=(13, 16, 22), width=1)
    for i in range(0, WIDTH, 240):
        draw.line((i, HEIGHT, i + 420, 0), fill=(15, 18, 24), width=1)


def draw_shadowed_rect(
    img: Image.Image,
    box: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    width: int = 2,
    shadow_alpha: int = 100,
) -> None:
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    sx0, sy0, sx1, sy1 = box
    shadow_draw.rounded_rectangle(
        (sx0 + 8, sy0 + 14, sx1 + 8, sy1 + 14),
        radius=radius,
        fill=(0, 0, 0, shadow_alpha),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    img.alpha_composite(shadow)

    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=fill + (245,),
        outline=outline + (255,),
        width=width,
    )


def draw_pill(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    text_fill: tuple[int, int, int],
) -> None:
    draw.rounded_rectangle(box, radius=(box[3] - box[1]) // 2, fill=fill, outline=outline, width=1)
    tw, th = text_size(draw, text, font)
    draw.text(
        (box[0] + (box[2] - box[0] - tw) // 2, box[1] + (box[3] - box[1] - th) // 2 - 2),
        text,
        font=font,
        fill=text_fill,
    )


def draw_rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int,
    outline: tuple[int, int, int],
    fill: tuple[int, int, int],
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_outlined_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    stroke: int = 4,
) -> None:
    x, y = xy
    for dx in range(-stroke, stroke + 1):
        for dy in range(-stroke, stroke + 1):
            if dx * dx + dy * dy <= stroke * stroke:
                draw.text((x + dx, y + dy), text, font=font, fill=(255, 255, 255))
    draw.text((x, y), text, font=font, fill=fill)


def draw_prompt(draw: ImageDraw.ImageDraw, prompt: str) -> None:
    box = (56, 54, WIDTH - 56, 178)
    draw.rounded_rectangle(
        box,
        radius=10,
        fill=(18, 19, 20),
        outline=PROMPT_BORDER,
        width=2,
    )
    title = "Prompt"
    tw, _ = text_size(draw, title, FONT_LABEL)
    draw.rectangle((WIDTH // 2 - tw // 2 - 18, 42, WIDTH // 2 + tw // 2 + 18, 68), fill=BG)
    draw.text((WIDTH // 2 - tw // 2, 39), title, font=FONT_LABEL, fill=PROMPT_TITLE)
    prompt_lines = wrap_text(prompt, 128)[:3]
    y = 92
    for line in prompt_lines:
        tw, _ = text_size(draw, line, FONT_PROMPT)
        draw.text(((WIDTH - tw) // 2, y), line, font=FONT_PROMPT, fill=TEXT)
        y += 29


def format_tok_s(tok_s: float) -> str:
    if tok_s >= 1000 or tok_s.is_integer():
        return f"{tok_s:.0f}"
    return f"{tok_s:.1f}"


def format_time_s(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:04.1f}"


def render_panel(
    img: Image.Image,
    method: Method,
    box: tuple[int, int, int, int],
    t: float,
) -> None:
    x0, y0, x1, y1 = box
    draw_shadowed_rect(
        img,
        box,
        radius=12,
        fill=PANEL_BG,
        outline=mix(method.color, TEXT, 0.10),
        width=2,
        shadow_alpha=118,
    )
    draw = ImageDraw.Draw(img)

    header = (x0 + 2, y0 + 2, x1 - 2, y0 + 106)
    draw.rounded_rectangle(header, radius=10, fill=PANEL_HEADER)
    draw.rectangle((x0 + 2, y0 + 72, x1 - 2, y0 + 108), fill=PANEL_HEADER)
    draw.line((x0 + 18, y0 + 106, x1 - 18, y0 + 106), fill=(37, 43, 53), width=1)

    title_font = FONT_LABEL
    title = method.title
    max_title_w = x1 - x0 - 44
    if text_size(draw, title, title_font)[0] > max_title_w:
        title_font = FONT_LABEL_SMALL
    draw.text((x0 + 22, y0 + 17), title, font=title_font, fill=method.color)

    speed = f"{format_tok_s(method.tok_s)} TPS"
    draw_pill(
        draw,
        (x0 + 22, y0 + 58, x0 + 196, y0 + 90),
        speed,
        FONT_STAT_SMALL,
        fill=mix(PANEL_HEADER, method.color, 0.16),
        outline=mix(PANEL_HEADER, method.color, 0.44),
        text_fill=TEXT,
    )
    draw_pill(
        draw,
        (x1 - 122, y0 + 58, x1 - 22, y0 + 90),
        method.badge,
        FONT_BADGE,
        fill=mix(PANEL_HEADER, method.color, 0.20),
        outline=mix(PANEL_HEADER, method.color, 0.55),
        text_fill=method.color,
    )

    progress = min(1.0, max(0.0, t / method.reveal_s))
    visible_chars = int(len(method.text) * progress)
    shown = method.text[:visible_chars]

    token_count = min(DEMO_OUTPUT_TOKENS, int(round(DEMO_OUTPUT_TOKENS * progress)))
    if progress >= 1.0:
        status = f"DONE at {method.reveal_s:.1f}s"
        status_fill = method.color
    else:
        status = f"{token_count:03d}/{DEMO_OUTPUT_TOKENS} tokens"
        status_fill = MUTED
    draw.text((x0 + 22, y0 + 112), status, font=FONT_STATUS, fill=status_fill)

    pad = 22
    text_top = y0 + 142
    text_bottom = y1 - 58
    max_chars = max(22, (x1 - x0 - 2 * pad) // 12)
    lines = wrap_text(shown, max_chars)
    line_h = 27
    max_lines = max(1, (text_bottom - text_top) // line_h)
    if len(lines) > max_lines:
        lines = lines[-max_lines:]

    y = text_top
    for line in lines:
        draw.text((x0 + pad, y), line, font=FONT_MONO, fill=TEXT)
        y += line_h

    if progress < 1.0 and lines:
        last_line = lines[-1]
        tw, _ = text_size(draw, last_line, FONT_MONO)
        cursor_x = min(x0 + pad + tw + 5, x1 - pad - 9)
        cursor_y = max(text_top, y - line_h + 3)
        if int(t * 2) % 2 == 0:
            draw.rounded_rectangle(
                (cursor_x, cursor_y, cursor_x + 8, cursor_y + 22),
                radius=2,
                fill=method.color,
            )

    for i in range(96):
        alpha = i / 95
        yy = y1 - 96 + i
        color = tuple(int(PANEL_BG[j] * alpha + BG[j] * (1 - alpha)) for j in range(3))
        draw.line((x0 + 2, yy, x1 - 2, yy), fill=color)

    bar_x0, bar_y0 = x0 + 22, y1 - 34
    bar_x1, bar_y1 = x1 - 22, y1 - 24
    draw.rounded_rectangle(
        (bar_x0, bar_y0, bar_x1, bar_y1),
        radius=5,
        fill=(34, 39, 48),
    )
    draw.rounded_rectangle(
        (bar_x0, bar_y0, bar_x0 + int((bar_x1 - bar_x0) * progress), bar_y1),
        radius=5,
        fill=method.color,
    )


def draw_footer(draw: ImageDraw.ImageDraw, t: float) -> None:
    margin = 70
    y = HEIGHT - 44
    draw.line((margin, y, WIDTH - margin, y), fill=(43, 49, 58), width=6)
    progress = min(1.0, t / DURATION_S)
    draw.line(
        (margin, y, margin + int((WIDTH - 2 * margin) * progress), y),
        fill=(188, 193, 202),
        width=6,
    )
    stamp = f"{format_time_s(t)} / {format_time_s(DURATION_S)}"
    draw.text((margin, y - 34), stamp, font=FONT_FOOTER, fill=MUTED)


def make_methods() -> tuple[str, list[Method]]:
    prompt, base_text, _base_tok_s, domino_text, _domino_tok_s = read_hf_answers(HF_ANSWERS)
    demo_text = domino_text or base_text
    methods: list[Method] = []
    baseline_tps = PAPER_DEMO_METRICS[0][2]
    for idx, (title, color, tok_s, badge) in enumerate(PAPER_DEMO_METRICS):
        speedup = tok_s / baseline_tps
        reveal_s = max(2.4, BASELINE_REVEAL_S / speedup)
        methods.append(
            Method(
                title=title,
                color=color,
                text=demo_text,
                tok_s=tok_s,
                badge=badge,
                reveal_s=reveal_s,
            )
        )
    return prompt, methods


def render_frame(prompt: str, methods: list[Method], t: float) -> Image.Image:
    img = Image.new("RGBA", (WIDTH, HEIGHT), BG + (255,))
    draw = ImageDraw.Draw(img)
    draw_background(draw)
    draw_prompt(draw, prompt)

    left = 56
    gap = 18
    top = 232
    bottom = HEIGHT - 72
    panel_w = (WIDTH - 2 * left - 3 * gap) // 4
    for idx, method in enumerate(methods):
        x0 = left + idx * (panel_w + gap)
        x1 = x0 + panel_w
        render_panel(img, method, (x0, top, x1, bottom), t)

    draw = ImageDraw.Draw(img)
    draw_footer(draw, t)
    return img.convert("RGB")


def encode_video(output: Path) -> None:
    prompt, methods = make_methods()
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    total = int(math.ceil(DURATION_S * FPS))
    try:
        for frame_idx in range(total):
            t = frame_idx / FPS
            frame = render_frame(prompt, methods, t)
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with code {ret}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "asset/DFlash_demo.mp4",
        help="Output mp4 path.",
    )
    args = parser.parse_args()
    encode_video(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
