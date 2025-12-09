import os
import time
import io
import zipfile
import gradio as gr
from ultralytics import YOLO
import numpy as np
import cv2
from datetime import datetime
import base64
from PIL import Image, UnidentifiedImageError
import torch

model = YOLO("runs/detect/train20/weights/best.pt")
# Prefer GPU if available; use half precision on CUDA for speed.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE == "cuda"
model.to(DEVICE)
if USE_HALF:
    model.model.half()
# Warmup to remove first-call latency (cheap dummy inference)
_ = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640, device=DEVICE, verbose=False)

# Enhanced color scheme and symbols for each defect class
DEFECT_SYMBOLS = {
    "missing_holes": "⚠️",
    "mouse_bite": "🟡",
    "open_circuit": "🔴",
    "short": "⚡",
    "spur": "🟠",
    "copper": "🟢"
}

DEFECT_COLORS = {
    "missing_holes": "#FFA500",  # Orange
    "mouse_bite": "#FFD700",     # Gold
    "open_circuit": "#FF0000",   # Red
    "short": "#FF00FF",          # Magenta
    "spur": "#FF8C00",           # Dark Orange
    "copper": "#00FF00"          # Green
}

MAX_SIDE = 1024  # Resize large uploads for faster inference; adjust if needed

def resize_with_aspect(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    """Resize while preserving aspect ratio, limiting longest side to max_side."""
    width, height = img.size
    longest = max(width, height)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_size = (int(width * scale), int(height * scale))
    return img.resize(new_size, Image.Resampling.LANCZOS)

def image_to_base64(img_array: np.ndarray) -> str:
    """Encode a numpy image array to base64 PNG."""
    _, buffer = cv2.imencode(".png", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")

def detect_single_image(img):
    """Process a single image and return results with pixel-level details"""
    frame = np.array(img)
    original_height, original_width = frame.shape[:2]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Use device/precision settings configured above
    results = model.predict(frame, conf=0.35, imgsz=640, device=DEVICE)
    
    # Annotated output image
    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    # Build detection summary with pixel details
    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = results[0].names[cls]
        symbol = DEFECT_SYMBOLS.get(name, "🔍")
        color = DEFECT_COLORS.get(name, "#808080")
        
        # Get bounding box coordinates in pixel format
        # YOLO's box.xyxy returns coordinates in pixels (already scaled)
        x1_pixel, y1_pixel, x2_pixel, y2_pixel = box.xyxy[0].cpu().numpy()
        x1_pixel = int(x1_pixel)
        y1_pixel = int(y1_pixel)
        x2_pixel = int(x2_pixel)
        y2_pixel = int(y2_pixel)
        
        # Calculate dimensions
        width_pixel = x2_pixel - x1_pixel
        height_pixel = y2_pixel - y1_pixel
        area_pixel = width_pixel * height_pixel
        
        # Center coordinates
        center_x = int((x1_pixel + x2_pixel) / 2)
        center_y = int((y1_pixel + y2_pixel) / 2)
        
        # Calculate MPBA (Mean Pixel Bounding Area) - percentage of image
        image_area = original_width * original_height
        mpba_percentage = (area_pixel / image_area) * 100
        
        detections.append({
            "name": name,
            "confidence": conf,
            "symbol": symbol,
            "color": color,
            "x1": x1_pixel,
            "y1": y1_pixel,
            "x2": x2_pixel,
            "y2": y2_pixel,
            "width": width_pixel,
            "height": height_pixel,
            "area": area_pixel,
            "center_x": center_x,
            "center_y": center_y,
            "mpba": mpba_percentage
        })
    
    return annotated, detections, original_width, original_height

def process_images(images):
    """Process multiple images and return results with pixel details"""
    if not images:
        return None, "Please upload at least one image.", None
    
    results_data = []
    all_images = []
    all_defects_tables = []
    
    for idx, img in enumerate(images, 1):
        t0 = time.perf_counter()
        annotated, detections, img_width, img_height = detect_single_image(img)
        duration_ms = (time.perf_counter() - t0) * 1000
        orig_w, orig_h = getattr(img, "orig_size", (img_width, img_height))
        print(f"[Timing] Image {idx}: original {orig_w}x{orig_h} -> used {img_width}x{img_height}, detect {duration_ms:.1f} ms")
        all_images.append(annotated)
        
        # Store results for this image
        image_result = {
            "image_number": idx,
            "defects": detections,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_width": img_width,
            "image_height": img_height,
            "duration_ms": duration_ms,
            "original_width": orig_w,
            "original_height": orig_h
        }
        results_data.append(image_result)
        
        # Create HTML block per image with its annotated image and table
        img_b64 = image_to_base64(annotated)
        table_html = f"""
        <div style='margin-bottom: 30px; padding: 20px; background: #F8F9FA; border-radius: 10px; border: 2px solid #E0E0E0;'>
            <h4 style='color: #2C3E50; margin-top: 0; margin-bottom: 15px; border-bottom: 2px solid #667eea; padding-bottom: 10px;'>
                 Image {idx} - Dimensions: {img_width} × {img_height} pixels (original: {orig_w} × {orig_h})
            </h4>
            <div style='text-align: center; margin-bottom: 15px;'>
                <img src="data:image/png;base64,{img_b64}" alt="Annotated Image {idx}" style="width: 360px; height: 260px; object-fit: contain; background: #fff; border-radius: 8px; border: 1px solid #ddd;" />
                <div style='margin-top: 8px;'>
                    <a href="data:image/png;base64,{img_b64}" download="annotated_image_{idx}.png" style='display: inline-block; padding: 8px 14px; background: #667eea; color: white; border-radius: 6px; text-decoration: none; font-weight: 600; box-shadow: 0 2px 4px rgba(0,0,0,0.15);'>
                        ⬇️ Download Image {idx}
                    </a>
                </div>
            </div>
            <p style='margin: 0 0 15px 0; color: #555; font-size: 0.95em;'>
                 Processing time: {duration_ms:.1f} ms
            </p>
        """
        
        if detections:
            table_html += """
            <div style='overflow-x: auto;'>
                <table style='width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <thead>
                        <tr style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>
                            <th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>#</th>
                            <th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>Defect Type</th>
                            <th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>Confidence</th>
                            <th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>X1 (px)</th>
                            <th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>Y1 (px)</th>
                            <th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>X2 (px)</th>
                            <th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>Y2 (px)</th>
                            <th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>Width (px)</th>
                            <th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>Height (px)</th>
                            <th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>Area (px²)</th>
                            <th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>Center X (px)</th>
                            <th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>Center Y (px)</th>
                            <th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>MPBA (%)</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for det_idx, det in enumerate(detections, 1):
                row_color = "#F8F9FA" if det_idx % 2 == 0 else "white"
                table_html += f"""
                        <tr style='background: {row_color};'>
                            <td style='padding: 10px; border: 1px solid #ddd; font-weight: bold;'>{det_idx}</td>
                            <td style='padding: 10px; border: 1px solid #ddd;'>
                                <span style='color: {det['color']}; font-weight: bold;'>{det['symbol']} {det['name']}</span>
                            </td>
                            <td style='padding: 10px; border: 1px solid #ddd;'>{det['confidence']:.2%}</td>
                            <td style='padding: 10px; border: 1px solid #ddd; font-family: monospace;'>{det['x1']}</td>
                            <td style='padding: 10px; border: 1px solid #ddd; font-family: monospace;'>{det['y1']}</td>
                            <td style='padding: 10px; border: 1px solid #ddd; font-family: monospace;'>{det['x2']}</td>
                            <td style='padding: 10px; border: 1px solid #ddd; font-family: monospace;'>{det['y2']}</td>
                            <td style='padding: 10px; border: 1px solid #ddd; font-family: monospace;'>{det['width']}</td>
                            <td style='padding: 10px; border: 1px solid #ddd; font-family: monospace;'>{det['height']}</td>
                            <td style='padding: 10px; border: 1px solid #ddd; font-family: monospace;'>{det['area']}</td>
                            <td style='padding: 10px; border: 1px solid #ddd; font-family: monospace;'>{det['center_x']}</td>
                            <td style='padding: 10px; border: 1px solid #ddd; font-family: monospace;'>{det['center_y']}</td>
                            <td style='padding: 10px; border: 1px solid #ddd; font-family: monospace;'>{det['mpba']:.4f}%</td>
                        </tr>
                """
            
            table_html += f"""
                    </tbody>
                </table>
            </div>
            <p style='margin-top: 15px; color: #555; font-size: 0.9em;'>
                <strong>Total Defects:</strong> {len(detections)} | 
                <strong>MPBA:</strong> Mean Pixel Bounding Area (% of total image)
            </p>
            <div style='margin-top: 15px; padding: 15px; background: #FFF3E0; border-radius: 8px; border: 2px dashed #FF9800; text-align: center;'>
                <p style='margin: 0; color: #E65100; font-weight: bold;'>
                    Download the complete defect report in text format below the table
                </p>
            </div>
            """
        else:
            table_html += """
            <div style='padding: 20px; text-align: center; background: #E8F5E9; border-radius: 8px;'>
                <p style='color: #27AE60; font-size: 1.2em; font-weight: bold; margin: 0;'>✨ No defects detected</p>
            </div>
            """
        
        table_html += "</div>"
        all_defects_tables.append(table_html)
    
    # Combine all tables
    combined_tables = "".join(all_defects_tables)
    
    # Generate text report
    report_text = generate_text_report(results_data)
    
    return all_images, combined_tables, report_text

def generate_text_report(results_data):
    """Generate annotated text file report with pixel details in tabular format"""
    text_content = "=" * 100 + "\n"
    text_content += " " * 35 + "PCB DEFECT DETECTION REPORT\n"
    text_content += "=" * 100 + "\n"
    text_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    text_content += "=" * 100 + "\n\n"
    
    for result in results_data:
        text_content += "-" * 100 + "\n"
        text_content += f"IMAGE #{result['image_number']}\n"
        text_content += "-" * 100 + "\n"
        text_content += f"Image Dimensions: {result['image_width']} × {result['image_height']} pixels\n"
        text_content += f"Processed at: {result['timestamp']}\n"
        text_content += f"Total Defects Detected: {len(result['defects'])}\n"
        text_content += "-" * 100 + "\n\n"
        
        if result['defects']:
            # Create table header
            text_content += f"{'#':<4} {'Defect Type':<20} {'Conf':<8} {'X1':<8} {'Y1':<8} {'X2':<8} {'Y2':<8} "
            text_content += f"{'Width':<8} {'Height':<8} {'Area':<10} {'Ctr X':<8} {'Ctr Y':<8} {'MPBA %':<10}\n"
            text_content += "-" * 100 + "\n"
            
            # Add each defect as a table row
            for det_idx, det in enumerate(result['defects'], 1):
                text_content += f"{det_idx:<4} "
                text_content += f"{det['symbol']} {det['name']:<17} "
                text_content += f"{det['confidence']:<7.2%} "
                text_content += f"{det['x1']:<8} {det['y1']:<8} {det['x2']:<8} {det['y2']:<8} "
                text_content += f"{det['width']:<8} {det['height']:<8} {det['area']:<10} "
                text_content += f"{det['center_x']:<8} {det['center_y']:<8} {det['mpba']:<10.4f}\n"
            
            text_content += "-" * 100 + "\n\n"
            
            # Detailed information for each defect
            text_content += "DETAILED DEFECT INFORMATION:\n"
            text_content += "-" * 100 + "\n"
            for det_idx, det in enumerate(result['defects'], 1):
                text_content += f"\nDEFECT #{det_idx}:\n"
                text_content += f"  Type: {det['name']} {det['symbol']}\n"
                text_content += f"  Confidence: {det['confidence']:.2%}\n"
                text_content += f"  Bounding Box:\n"
                text_content += f"    Top-Left Corner (X1, Y1): ({det['x1']}, {det['y1']}) pixels\n"
                text_content += f"    Bottom-Right Corner (X2, Y2): ({det['x2']}, {det['y2']}) pixels\n"
                text_content += f"  Dimensions:\n"
                text_content += f"    Width: {det['width']} pixels\n"
                text_content += f"    Height: {det['height']} pixels\n"
                text_content += f"  Area: {det['area']} square pixels\n"
                text_content += f"  Center Point: ({det['center_x']}, {det['center_y']}) pixels\n"
                text_content += f"  MPBA (Mean Pixel Bounding Area): {det['mpba']:.4f}% of total image\n"
        else:
            text_content += "STATUS: No defects detected\n"
        
        text_content += "\n" + "=" * 100 + "\n\n"
    
    text_content += "=" * 100 + "\n"
    text_content += " " * 40 + "END OF REPORT\n"
    text_content += "=" * 100 + "\n"
    text_content += "Report generated by PCB Defect Detection System\n"
    text_content += "Powered by YOLO Ultralytics\n"
    text_content += "=" * 100 + "\n"
    
    return text_content

# Custom CSS for better styling
custom_css = """
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .main-header p {
        margin: 10px 0 0 0;
        opacity: 0.9;
        font-size: 1.1em;
    }
    .gradio-container {
        max-width: 1600px !important;
        margin: 0 auto;
    }
    table {
        font-size: 0.95em;
    }
    th {
        font-weight: 600 !important;
    }
"""

with gr.Blocks(title="PCB Defect Detector", theme=gr.themes.Soft(primary_hue="purple"), css=custom_css) as demo:
    # Main header
    gr.HTML("""
        <div class="main-header">
            <h1>PCB Defect Detection System</h1>
            
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("###  Upload Images")
            input_images = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="Select Multiple PCB Images",
                height=200
            )
            
            detect_btn = gr.Button(
                "Run Detection",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("""
            ### ℹ️ Instructions
            - Upload one or multiple PCB images
            - Click "Run Detection" to analyze defects
            - View results in tabular format with pixel details
            - Download annotated text report
            
            ### Defect Types
            - ⚠️ Missing Holes
            - 🟡 Mouse Bite
            - 🔴 Open Circuit
            - ⚡ Short
            - 🟠 Spur
            - 🟢 Copper
            
            ### Table Columns
            - **X1, Y1, X2, Y2**: Bounding box coordinates
            - **Width, Height**: Defect dimensions in pixels
            - **Area**: Total area in square pixels
            - **Center X, Y**: Center point coordinates
            - **MPBA**: Mean Pixel Bounding Area (%)
            """)
        
        with gr.Column(scale=3):
            gr.Markdown("### Detection Results - Tabular Format")
            
            output_gallery = gr.Gallery(
                label="Annotated Images",
                show_label=True,
                elem_id="gallery",
                columns=2,
                rows=2,
                height="auto",
                object_fit="contain"
            )
            
            defects_table = gr.HTML(
                label="Detected Defects Table (Pixel-Level Details)",
                value="<div style='padding: 20px; text-align: center; color: #7F8C8D;'>Upload images and click 'Run Detection' to see results here.</div>"
            )
            
            with gr.Row():
                report_file_txt = gr.File(
                    label="Download Defects Report (Text Format)",
                    visible=False,
                    scale=4
                )
                zip_file_all = gr.File(
                    label="Download All Annotated Images (ZIP)",
                    visible=False,
                    scale=4
                )
                gr.Markdown("""
                <div style='margin-top: 25px; padding: 10px; background: #E3F2FD; border-radius: 5px; border-left: 4px solid #2196F3;'>
                    <p style='margin: 0; color: #1976D2; font-size: 0.9em;'>
                        <strong>💡 Tip:</strong> Click the download button above to save the complete defect report with all pixel coordinates in text format.
                    </p>
                </div>
                """, scale=1)
    
    def process_and_generate(images):
        """Wrapper function to process images and create downloadable reports"""
        if not images:
            return None, "<div style='padding: 20px; text-align: center; color: #E74C3C;'>⚠️ Please upload at least one image.</div>", gr.File.update(visible=False), gr.File.update(visible=False)
        
        # Extract PIL images from file objects
        pil_images = []
        for img_file in images:
            if hasattr(img_file, 'name'):
                # It's a file path - validate and load safely
                file_path = img_file.name
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    error_msg = f"<div style='padding: 20px; text-align: center; color: #E74C3C;'>⚠️ The file '{os.path.basename(file_path)}' is empty or inaccessible. Please upload a valid image file.</div>"
                    return None, error_msg, gr.File.update(visible=False)
                try:
                    with open(file_path, "rb") as f:
                        img = Image.open(f)
                        img.load()  # force read to catch errors early
                        img = img.convert("RGB")
                        img.orig_size = img.size  # keep original dimensions for logging
                        img = resize_with_aspect(img)
                        pil_images.append(img)
                except UnidentifiedImageError:
                    error_msg = f"<div style='padding: 20px; text-align: center; color: #E74C3C;'>⚠️ Unable to read '{os.path.basename(file_path)}' as an image. Please upload a supported image format (e.g., PNG, JPG).</div>"
                    return None, error_msg, gr.File.update(visible=False), gr.File.update(visible=False)
                except OSError as exc:
                    error_msg = f"<div style='padding: 20px; text-align: center; color: #E74C3C;'>⚠️ Error reading '{os.path.basename(file_path)}': {exc}</div>"
                    return None, error_msg, gr.File.update(visible=False), gr.File.update(visible=False)
            else:
                # It's already a PIL image
                pil_images.append(img_file)
        
        annotated_images, tables_html, report_text = process_images(pil_images)
        
        # Save text report to temporary file
        txt_path = None
        if report_text:
            txt_path = f"pcb_defects_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(report_text)

        # Save all annotated images into a zip archive
        zip_path = None
        if annotated_images:
            zip_path = f"pcb_annotated_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
                for idx, ann in enumerate(annotated_images, 1):
                    buffer = io.BytesIO()
                    Image.fromarray(ann).save(buffer, format="PNG")
                    zipf.writestr(f"annotated_image_{idx}.png", buffer.getvalue())
        
        # Return with download button visible if report exists
        txt_update = gr.File.update(visible=bool(txt_path), value=txt_path if txt_path else None)
        zip_update = gr.File.update(visible=bool(zip_path), value=zip_path if zip_path else None)
        return annotated_images, tables_html, txt_update, zip_update
    
    detect_btn.click(
        fn=process_and_generate,
        inputs=input_images,
        outputs=[output_gallery, defects_table, report_file_txt, zip_file_all]
    )
    
    # Footer
    gr.HTML("""
        <div style='text-align: center; padding: 20px; color: #7F8C8D; margin-top: 20px; border-top: 2px solid #E0E0E0;'>
            <p><strong>PCB Defect Detection System</strong> | Powered by YOLO Ultralytics</p>
            <p style='font-size: 0.9em;'>Quality Control • AI-Powered Analysis • Pixel-Level Tabular Reporting</p>
        </div>
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)