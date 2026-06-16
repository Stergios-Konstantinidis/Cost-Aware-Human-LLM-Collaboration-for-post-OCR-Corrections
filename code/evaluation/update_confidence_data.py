import json
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add parent dir to path to import from code
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Re-using the logic from the new script
def extract_ocr_confidence_data(img_path, engine="paddle", threshold=0.8, cached_reader=None) -> dict:
    results = []
    lines, confs = [], []
    if engine == "paddle":
        try:
            res = cached_reader.ocr(str(img_path), cls=True)
            if res and res[0]:
                lines = [l[1][0] for l in res[0]]
                confs = [float(l[1][1]) / 100.0 if float(l[1][1]) > 1.0 else float(l[1][1]) for l in res[0]]
        except: pass
    elif engine == "easyocr":
        try:
            res = cached_reader.readtext(str(img_path))
            lines = [r[1] for r in res]
            confs = [float(r[2]) for r in res]
        except: pass
    elif engine == "tesseract":
        try:
            import pytesseract
            import pandas as pd
            from io import StringIO
            data = pytesseract.image_to_data(str(img_path), lang="fra")
            df = pd.read_csv(StringIO(data), sep="\t", quoting=3)
            df = df[df["conf"] != -1]
            line_groups = df.groupby(["block_num", "par_num", "line_num"])
            for _, group in line_groups:
                text = " ".join([str(x) for x in group["text"].tolist() if str(x).strip()])
                if not text: continue
                avg_conf = group["conf"].mean() / 100.0
                lines.append(text); confs.append(avg_conf)
        except: pass

    avg_image_conf = float(np.mean(confs)) if confs else 1.0
    for i, (text, conf) in enumerate(zip(lines, confs)):
        if conf < threshold:
            results.append({"index": i, "text": text, "confidence": conf, "prev_context": lines[max(0, i-3):i], "next_context": lines[i+1:min(len(lines), i+4)]})
    return {"avg_confidence": avg_image_conf, "low_confidence_lines": results}

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    eval_dir = data_dir / "evaluation_dataset"
    
    with open(eval_dir / "groundtruth.json", "r") as f:
        groundtruth_data = json.load(f)

    # Initialize engines
    print("Initializing OCR engines...")
    engines = {}
    try:
        import easyocr
        engines["easyocr"] = easyocr.Reader(["fr"], gpu=False)
    except: pass
    
    try:
        import pytesseract
        # Tesseract doesn't need a reader object like EasyOCR, 1 is a dummy
        engines["tesseract"] = 1
    except: pass
    
    try:
        from paddleocr import PaddleOCR

        engines["paddle"] = PaddleOCR(use_textline_orientation=True, lang="fr")
    except: pass
    
    for eng_name, eng_reader in engines.items():
        for threshold in [0.8, 0.9]:
            out_file = results_dir / f"confidence_data/low_confidence_words_{int(threshold*100)}_{eng_name}.json"
            print(f"Updating {out_file}...")
            
            data = {}
            for item in tqdm(groundtruth_data):
                img_path = eval_dir / "images" / item["filename"]
                if img_path.exists():
                    data[item["filename"]] = extract_ocr_confidence_data(img_path, engine=eng_name, threshold=threshold, cached_reader=eng_reader)
            
            with open(out_file, "w") as f:
                json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
