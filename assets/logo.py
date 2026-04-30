"""
Generate the TIDE logo: ocean-gradient TIDE text with a sine-wave underline.

Color: navy → cyan vertical gradient on the letters; lighter sky-cyan wave
beneath them. Background white. The opening 30% of the GIF is a diffusion
noise reveal — both text and wave emerge from noise — matching the
framework's diffusion nature.

Outputs (next to this script):
  - logo.png : final clean frame (used in README)
  - logo.gif : full animation
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------- Configuration ----------
HERE = os.path.dirname(os.path.abspath(__file__))

W, H = 480, 210
TOTAL_DURATION = 3.0
FPS = 15
TEXT = "TIDE"

# Ocean palette
TOP_COLOR = (0, 61, 91)        # deep navy  #003D5B
BOT_COLOR = (0, 180, 216)      # bright cyan #00B4D8
WAVE_COLOR = (110, 200, 230)   # sky cyan
BG_COLOR = (255, 255, 255)

# Wave geometry
WAVE_Y = int(H * 0.86)         # baseline of the wave
WAVE_AMPLITUDE = 7
WAVE_PERIOD = 70               # pixels per cycle
WAVE_THICKNESS = 4

# Text geometry — letters live in the upper portion, leaving room for wave
TEXT_REGION_H = int(H * 0.78)

OUTPUT = os.path.join(HERE, "logo.gif")
LAST_FRAME_PNG = os.path.join(HERE, "logo.png")
DIFFUSION_PORTION = 0.3
SEED = 8

FONT_PATH = os.path.join(HERE, "JetBrainsMono-VariableFont_wght.ttf")
FONT_WEIGHT = "ExtraBold"


# ---------- Auto font size ----------
def load_font_auto_size(text, w, h, target_width_ratio=0.92, target_height_ratio=0.92):
    lo, hi = 10, 2000
    best_font = None
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(FONT_PATH, size=mid)
        try:
            font.set_variation_by_name(FONT_WEIGHT)
        except Exception:
            pass
        dummy = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(dummy)
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= w * target_width_ratio and th <= h * target_height_ratio:
            best_font = font
            lo = mid + 1
        else:
            hi = mid - 1
    return best_font if best_font is not None else font


def render_text_mask_in_region(W, H, region_h, text, font):
    """Render the text mask onto a full-size canvas (H x W), centered
    horizontally and centered within the upper `region_h` rows."""
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (W - tw) // 2 - bbox[0]
    y = (region_h - th) // 2 - bbox[1]
    d.text((x, y), text, font=font, fill=255)
    return np.asarray(img, np.float32) / 255.0


def render_wave_mask(W, H, baseline_y, amp, period, thickness, phase=0.0):
    """Draw an anti-aliased sine wave; return (H, W) float32 mask in [0, 1]."""
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    xs = np.arange(W)
    ys = baseline_y + amp * np.sin(2 * np.pi * xs / period + phase)
    pts = list(zip(xs.tolist(), ys.tolist()))
    for i in range(len(pts) - 1):
        d.line([pts[i], pts[i + 1]], fill=255, width=thickness)
    return np.asarray(img, np.float32) / 255.0


def build_gradient(H, W, top_rgb, bot_rgb):
    """Vertical gradient image (H, W, 3) in [0, 1]."""
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None, None]
    top = np.array(top_rgb, dtype=np.float32) / 255.0
    bot = np.array(bot_rgb, dtype=np.float32) / 255.0
    grad = top * (1.0 - y) + bot * y
    return np.broadcast_to(grad, (H, W, 3)).copy()


# ---------- Initialization ----------
font = load_font_auto_size(TEXT, W, TEXT_REGION_H)
text_mask = render_text_mask_in_region(W, H, TEXT_REGION_H, TEXT, font)
gradient = build_gradient(H, W, TOP_COLOR, BOT_COLOR)
wave_color_arr = np.array(WAVE_COLOR, dtype=np.float32) / 255.0
bg_arr = np.array(BG_COLOR, dtype=np.float32) / 255.0

num_frames = int(TOTAL_DURATION * FPS)
diffusion_frames = max(1, int(num_frames * DIFFUSION_PORTION))
hold_ms = int((TOTAL_DURATION - diffusion_frames / FPS) * 1000)

rng = np.random.default_rng(SEED)
frames = []


def composite_clean(text_mask, wave_mask):
    """Final clean image: gradient text + wave + white bg."""
    text_alpha = text_mask[..., None]
    wave_alpha = (wave_mask * (1.0 - text_mask))[..., None]  # text wins overlap
    bg_alpha = 1.0 - text_alpha - wave_alpha
    img = (
        bg_alpha * bg_arr
        + text_alpha * gradient
        + wave_alpha * wave_color_arr
    )
    return np.clip(img, 0.0, 1.0)


# ---------- Diffusion stage ----------
for i in range(diffusion_frames):
    t = i / max(1, diffusion_frames - 1)
    progress = t**0.9
    noise_sigma = (1.0 - progress) ** 2.2

    # Animate the wave: phase drifts left-to-right as we converge
    phase = 2.0 * np.pi * t * 0.6
    wave_mask = render_wave_mask(
        W, H, WAVE_Y, WAVE_AMPLITUDE, WAVE_PERIOD, WAVE_THICKNESS, phase=phase
    )

    # Background noise that clears toward white
    noise = rng.standard_normal((H, W, 1)).astype(np.float32)
    noise_img = 1.0 - noise_sigma * 0.5 * np.abs(noise)
    np.clip(noise_img, 0.0, 1.0, out=noise_img)

    # Reveal both the text and the wave from the noise
    reveal_mask = np.clip(text_mask + wave_mask, 0.0, 1.0)
    alpha = progress**2.0
    alpha_map = (reveal_mask * alpha).astype(np.float32)[..., None]

    target = composite_clean(text_mask, wave_mask)

    frame = (1.0 - alpha_map) * noise_img + alpha_map * target
    frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    frames.append(Image.fromarray(frame, mode="RGB"))

# ---------- Final clean frame (used for PNG and GIF hold) ----------
final_wave_mask = render_wave_mask(
    W, H, WAVE_Y, WAVE_AMPLITUDE, WAVE_PERIOD, WAVE_THICKNESS, phase=0.0
)
final_clean = composite_clean(text_mask, final_wave_mask)
final_frame = Image.fromarray((final_clean * 255).astype(np.uint8), mode="RGB")

final_frame.save(LAST_FRAME_PNG)
print(f"Last frame saved as: {LAST_FRAME_PNG}")

# ---------- Quantization (smaller GIF) ----------
pal_frames = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
pal_final = final_frame.convert("P", palette=Image.ADAPTIVE, colors=64)

# ---------- Save GIF ----------
normal_ms = int(1000 / FPS)
durations = [normal_ms] * len(pal_frames) + [hold_ms]

pal_frames[0].save(
    OUTPUT,
    save_all=True,
    append_images=pal_frames[1:] + [pal_final],
    duration=durations,
    loop=0,
    optimize=True,
)

print(f"GIF saved: {OUTPUT}")
print(
    f"Frames (diffusion only): {len(pal_frames)} at {FPS} FPS, "
    f"final hold {hold_ms} ms, resolution {W}x{H}"
)
