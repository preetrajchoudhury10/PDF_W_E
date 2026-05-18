import fitz  # PyMuPDF
import tkinter as tk
from tkinter import filedialog
import os
import io
import numpy as np
from PIL import Image
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


def select_pdfs():
    """Open file dialog to select MULTIPLE PDFs"""
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes('-topmost', True)
    
    file_paths = filedialog.askopenfilenames(
        title="Select PDF file(s)",
        filetypes=[("PDF files", "*.pdf")]
    )
    root.destroy()
    return file_paths


PDF_DPI = 400

def process_page(page_num, doc, mat_img, images_folder, page_rects):
    page = doc[page_num]
    page_rect = page_rects[page_num]

    pix = page.get_pixmap(matrix=mat_img, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    data = np.array(img)

    r = data[:,:,0].astype(np.int16)
    g = data[:,:,1].astype(np.int16)
    b = data[:,:,2].astype(np.int16)

    red_mask = (r > g + 15) & (r > b + 15)
    ghost_mask = (r > 200) & (g > 200) & (b > 200)
    data[red_mask | ghost_mask] = [255, 255, 255]

    clean_img = Image.fromarray(data)

    img_path = os.path.join(images_folder, f"page_{page_num + 1:03d}.png")
    clean_img.save(img_path, format='PNG')

    scale = PDF_DPI / 72
    pdf_w = int(page_rect.width * scale)
    pdf_h = int(page_rect.height * scale)
    pdf_img = clean_img.resize((pdf_w, pdf_h), Image.LANCZOS)

    img_byte_arr = io.BytesIO()
    pdf_img.save(img_byte_arr, format='JPEG', quality=95)
    return page_num, img_byte_arr.getvalue()


def process_pdf(pdf_path, output_folder):
    print(f"\n📄 Processing: {os.path.basename(pdf_path)}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ Error opening PDF: {e}")
        return

    total_pages = doc.page_count
    print(f"   Turbo: {total_pages} pages @ 600 DPI")

    base_name = os.path.basename(pdf_path)
    name_only, _ = os.path.splitext(base_name)

    images_folder = os.path.join(output_folder, f"{name_only}_images")
    os.makedirs(images_folder, exist_ok=True)

    zoom_img = 600 / 72
    mat_img = fitz.Matrix(zoom_img, zoom_img)

    page_rects = [doc[i].rect for i in range(total_pages)]
    page_data = [None] * total_pages

    workers = min(os.cpu_count() or 4, total_pages)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(process_page, i, doc, mat_img, images_folder, page_rects): i
            for i in range(total_pages)
        }
        for f in as_completed(futures):
            i, img_bytes = f.result()
            page_data[i] = img_bytes

    out_pdf = fitz.open()
    for i in range(total_pages):
        new_page = out_pdf.new_page(width=page_rects[i].width, height=page_rects[i].height)
        new_page.insert_image(new_page.rect, stream=page_data[i])

    output_path = os.path.join(output_folder, f"{name_only}_cleaned.pdf")
    counter = 1
    while os.path.exists(output_path):
        output_path = os.path.join(output_folder, f"{name_only}_cleaned_{counter}.pdf")
        counter += 1

    out_pdf.save(output_path, garbage=4, deflate=True)
    out_pdf.close()
    doc.close()

    print(f"   💾 Saved: {os.path.basename(output_path)}")


def check_dependencies():
    """Ensure required libraries exist before running."""
    try:
        import fitz
        import numpy
        from PIL import Image
        return True
    except ImportError:
        print("❌ Missing required libraries! Please run: pip install PyMuPDF Pillow numpy")
        return False


def main():
    """Main automated batch processing flow."""
    print("=" * 60)
    print("    PRO PDF WATERMARK REMOVER (BATCH & TIMESTAMPS)")
    print("=" * 60)
    
    if not check_dependencies():
        return
    
    print("\n📂 Waiting for file selection...")
    
    # Get tuple of all selected files
    pdf_paths = select_pdfs()
    
    if not pdf_paths:
        print("❌ No files selected. Exiting immediately.")
        return
        
    print(f"📂 You selected {len(pdf_paths)} file(s) to process.")
    
    # --- NEW FEATURE: Create a Timestamped Output Folder ---
    # Find the directory of the first selected file to base our new folder in
    base_dir = os.path.dirname(pdf_paths[0])
    
    # Generate a timestamp (e.g., 2026-05-14_14-30-00)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"Cleaned_PDFs_{timestamp}"
    output_folder = os.path.join(base_dir, folder_name)
    
    # Create the directory
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n📁 Created output folder: {output_folder}")
    
    # Process them one by one, sending them to the new folder
    for path in pdf_paths:
        process_pdf(path, output_folder)
        
    print("\n" + "=" * 60)
    print("🎉 ALL FILES PROCESSED SUCCESSFULLY! Exiting...")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")