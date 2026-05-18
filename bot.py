import os
import io
import asyncio
import logging
import fitz
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("BOT_TOKEN")
PDF_DPI = 400
IMG_DPI = 600
MAX_FILE_SIZE = 45 * 1024 * 1024


def remove_watermark_from_page(page, mat_img, page_rect, pdf_scale):
    pix = page.get_pixmap(matrix=mat_img, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    data = np.array(img)

    r = data[:, :, 0].astype(np.int16)
    g = data[:, :, 1].astype(np.int16)
    b = data[:, :, 2].astype(np.int16)

    red_mask = (r > g + 15) & (r > b + 15)
    ghost_mask = (r > 200) & (g > 200) & (b > 200)
    data[red_mask | ghost_mask] = [255, 255, 255]

    clean_img = Image.fromarray(data)

    png_buf = io.BytesIO()
    clean_img.save(png_buf, format="PNG")
    png_bytes = png_buf.getvalue()

    pdf_w = int(page_rect.width * pdf_scale)
    pdf_h = int(page_rect.height * pdf_scale)
    pdf_img = clean_img.resize((pdf_w, pdf_h), Image.LANCZOS)

    jpg_buf = io.BytesIO()
    pdf_img.save(jpg_buf, format="JPEG", quality=95)

    return png_bytes, jpg_buf.getvalue()


def process_pdf_bytes(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = doc.page_count

    zoom_img = IMG_DPI / 72
    mat_img = fitz.Matrix(zoom_img, zoom_img)
    pdf_scale = PDF_DPI / 72

    page_rects = [doc[i].rect for i in range(total_pages)]
    pdf_pages = [None] * total_pages
    png_pages = [None] * total_pages

    workers = min(os.cpu_count() or 2, total_pages)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(remove_watermark_from_page, doc[i], mat_img, page_rects[i], pdf_scale): i
            for i in range(total_pages)
        }
        for f in as_completed(futures):
            i = futures[f]
            png_bytes, jpg_bytes = f.result()
            png_pages[i] = png_bytes
            pdf_pages[i] = jpg_bytes

    out_pdf = fitz.open()
    for i in range(total_pages):
        new_page = out_pdf.new_page(width=page_rects[i].width, height=page_rects[i].height)
        new_page.insert_image(new_page.rect, stream=pdf_pages[i])

    pdf_buf = io.BytesIO()
    out_pdf.save(pdf_buf, garbage=4, deflate=True)
    out_pdf.close()
    doc.close()

    return pdf_buf.getvalue(), png_pages


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send me a PDF file and I'll remove watermarks (red marks/ghost pixels) "
        "from it. I'll return the cleaned PDF plus each page as a high-quality image."
    )


async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg.document:
        return

    if not msg.document.file_name.lower().endswith(".pdf"):
        await msg.reply_text("Please send a PDF file.")
        return

    file_size = msg.document.file_size
    if file_size and file_size > MAX_FILE_SIZE:
        await msg.reply_text(f"File too large ({file_size / 1024 / 1024:.1f}MB). Max is 45MB.")
        return

    status = await msg.reply_text("Downloading file...")
    try:
        file = await context.bot.get_file(msg.document.file_id)
        pdf_bytes = await file.download_as_bytearray()
    except Exception as e:
        await status.edit_text(f"Failed to download file: {e}")
        return

    await status.edit_text(f"Processing {msg.document.file_name} ({len(pdf_bytes) / 1024 / 1024:.1f}MB)...")
    try:
        cleaned_pdf, page_images = await asyncio.get_running_loop().run_in_executor(
            None, process_pdf_bytes, bytes(pdf_bytes)
        )
    except Exception as e:
        logger.exception("Processing failed")
        await status.edit_text(f"Processing failed: {e}")
        return

    await status.edit_text("Sending cleaned PDF...")

    name = os.path.splitext(msg.document.file_name)[0]
    try:
        await msg.reply_document(
            document=cleaned_pdf,
            filename=f"{name}_cleaned.pdf",
        )
    except Exception as e:
        await msg.reply_text(f"Could not send PDF (may exceed Telegram's 50MB limit): {e}")

    await status.edit_text(f"Sending {len(page_images)} page images...")
    for i, png_bytes in enumerate(page_images):
        try:
            await msg.reply_document(
                document=png_bytes,
                filename=f"{name}_page_{i + 1:03d}.png",
            )
        except Exception as e:
            await msg.reply_text(f"Could not send page {i + 1}: {e}")

    await status.delete()


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled error: %s", context.error)


def main():
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN environment variable not set!")
        return

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
    app.add_error_handler(error_handler)

    port = int(os.environ.get("PORT", 8080))
    webhook_url = os.environ.get("WEBHOOK_URL")

    if webhook_url:
        logger.info(f"Starting webhook on port {port} at {webhook_url}")
        app.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=BOT_TOKEN,
            webhook_url=f"{webhook_url}/{BOT_TOKEN}",
        )
    else:
        logger.info("Starting polling...")
        app.run_polling()


if __name__ == "__main__":
    main()
