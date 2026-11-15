# -*- coding: utf-8 -*-
"""Giao diện web – Nhóm 19: Phân loại 102 loài hoa (Oxford Flowers 102).

Chạy:  py app.py            (tự mở trình duyệt)
       py app.py --share    (tạo link công khai tạm thời)
"""

import argparse
import html
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
import gradio as gr

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "Dataset"
MODEL_CANDIDATES = [BASE_DIR / "flower_classifier.joblib", OUT_DIR / "flower_classifier.joblib"]
LOW_CONFIDENCE = 0.40
TOP_K = 5


# =================================================================
# 1. TẢI MÔ HÌNH
# =================================================================
def load_bundle():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return joblib.load(path), path
    sys.exit("Không tìm thấy flower_classifier.joblib.\n"
             "Hãy chạy `py main.py` để huấn luyện, hoặc đặt file mô hình cạnh app.py.")


bundle, MODEL_PATH = load_bundle()
scaler, clf, class_names = bundle["scaler"], bundle["model"], bundle["class_names"]
BACKBONE_NAME = bundle.get("backbone_name", "resnet50")
IMG_SIZE = bundle.get("img_size", 224)
MEAN = list(bundle.get("mean", (0.485, 0.456, 0.406)))
STD = list(bundle.get("std", (0.229, 0.224, 0.225)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_backbone():
    # Backbone phải giống hệt lúc huấn luyện. Bundle của nhom19.py có kèm
    # state_dict; bundle của main.py dùng ResNet50 IMAGENET1K_V2 mặc định.
    if BACKBONE_NAME == "resnet18":
        net = models.resnet18(weights=None if "backbone_state_dict" in bundle
                              else models.ResNet18_Weights.DEFAULT)
    else:
        net = models.resnet50(weights=None if "backbone_state_dict" in bundle
                              else models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = nn.Identity()
    if "backbone_state_dict" in bundle:
        net.load_state_dict(bundle["backbone_state_dict"])
    for p in net.parameters():
        p.requires_grad_(False)
    return net.to(DEVICE).eval()


backbone = build_backbone()
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# =================================================================
# 2. DỰ ĐOÁN
# =================================================================
def pretty(name):
    return str(name).strip().title()


@torch.inference_mode()
def predict(img):
    if img is None:
        return EMPTY_RESULT, None
    start = time.perf_counter()
    x = preprocess(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    feat = backbone(x).cpu().numpy()
    proba = clf.predict_proba(scaler.transform(feat))[0]
    elapsed = (time.perf_counter() - start) * 1000
    top = np.argsort(proba)[::-1][:TOP_K]
    label = {pretty(class_names[i]): float(proba[i]) for i in top}
    return render_result(top, proba, elapsed), label


def render_result(top, proba, elapsed_ms):
    best, conf = int(top[0]), float(proba[top[0]])
    if conf >= 0.75:
        badge, badge_cls = "Độ tin cậy cao", "ok"
    elif conf >= LOW_CONFIDENCE:
        badge, badge_cls = "Độ tin cậy trung bình", "mid"
    else:
        badge, badge_cls = "Không chắc chắn", "low"

    rows = []
    for rank, i in enumerate(top, start=1):
        p = float(proba[i]) * 100
        rows.append(f"""
        <div class="fc-row{' fc-first' if rank == 1 else ''}">
          <span class="fc-rank">{rank}</span>
          <span class="fc-name">{html.escape(pretty(class_names[i]))}</span>
          <span class="fc-bar"><span style="width:{max(p, 0.8):.1f}%"></span></span>
          <span class="fc-pct">{p:.1f}%</span>
        </div>""")

    hint = ""
    if conf < LOW_CONFIDENCE:
        hint = ('<p class="fc-hint">Mô hình phân vân giữa nhiều loài. Hãy thử ảnh chụp rõ bông hoa, '
                'cận cảnh, đủ sáng. Lưu ý mô hình chỉ nhận biết 102 loài trong bộ Oxford Flowers.</p>')

    return f"""
    <div class="fc-card">
      <div class="fc-top">
        <div>
          <div class="fc-kicker">Kết quả dự đoán</div>
          <div class="fc-title">{html.escape(pretty(class_names[best]))}</div>
        </div>
        <div class="fc-score">
          <div class="fc-score-num">{conf * 100:.1f}<small>%</small></div>
          <span class="fc-badge {badge_cls}">{badge}</span>
        </div>
      </div>
      <div class="fc-sub">Top-{len(top)} khả năng cao nhất</div>
      {''.join(rows)}
      {hint}
      <div class="fc-meta">Thời gian xử lý: {elapsed_ms:.0f} ms · Thiết bị: {DEVICE.upper()}</div>
    </div>"""


EMPTY_RESULT = """
<div class="fc-card fc-empty">
  <div class="fc-empty-icon">🌸</div>
  <div class="fc-empty-title">Chưa có ảnh</div>
  <div class="fc-empty-text">Tải ảnh lên, chụp từ webcam hoặc chọn một ảnh mẫu bên dưới để bắt đầu.</div>
</div>"""


# =================================================================
# 3. DỮ LIỆU PHỤ: ẢNH MẪU + KẾT QUẢ ĐÁNH GIÁ
# =================================================================
def example_images(n=12):
    folder = DATA_DIR / "test"
    if not folder.is_dir():
        return []
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        return []
    idx = np.linspace(0, len(files) - 1, min(n, len(files)), dtype=int)
    return [[str(files[i])] for i in idx]


def load_metrics():
    path = OUT_DIR / "ket_qua_test.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    for col in df.columns[1:]:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].max() <= 1.0:
            df[col] = (df[col] * 100).round(2)
    return df


def best_accuracy(df):
    if df is None:
        return None
    row = df[df["Mô hình"].str.contains("Logistic", case=False)]
    return float((row if len(row) else df).iloc[0]["Accuracy"])


METRICS = load_metrics()
ACC = best_accuracy(METRICS)
EXAMPLES = example_images()


def stat(value, label):
    return f'<div class="fc-stat"><div class="fc-stat-v">{value}</div><div class="fc-stat-l">{label}</div></div>'


HERO = f"""
<div class="fc-hero">
  <div class="fc-hero-text">
    <div class="fc-eyebrow">Nhóm 19 · Học máy</div>
    <h1>Nhận dạng 102 loài hoa</h1>
    <p>Trích xuất đặc trưng bằng {BACKBONE_NAME.upper()} (transfer learning từ ImageNet),
       phân loại bằng Logistic Regression trên bộ dữ liệu Oxford Flowers 102.</p>
  </div>
  <div class="fc-stats">
    {stat(len(class_names), "loài hoa")}
    {stat(f"{ACC:.1f}%" if ACC is not None else "—", "accuracy tập test")}
    {stat(BACKBONE_NAME.replace("resnet", "ResNet-"), "backbone")}
  </div>
</div>"""


# =================================================================
# 4. GIAO DIỆN
# =================================================================
CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
footer { display: none !important; }

.fc-hero { display:flex; gap:24px; align-items:center; justify-content:space-between; flex-wrap:wrap;
  padding:28px 32px; border-radius:18px; color:#fff;
  background: linear-gradient(135deg, #7c3aed 0%, #db2777 55%, #f59e0b 100%); }
.fc-hero h1 { margin:4px 0 8px; font-size:2rem; font-weight:800; color:#fff; letter-spacing:-.02em; }
.fc-hero p { margin:0; max-width:560px; opacity:.92; line-height:1.55; color:#fff; }
.fc-eyebrow { text-transform:uppercase; letter-spacing:.12em; font-size:.75rem; font-weight:700; opacity:.85; }
.fc-stats { display:flex; gap:12px; flex-wrap:wrap; }
.fc-stat { background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28); border-radius:14px;
  padding:12px 18px; min-width:110px; backdrop-filter: blur(6px); }
.fc-stat-v { font-size:1.45rem; font-weight:800; }
.fc-stat-l { font-size:.78rem; opacity:.9; }

.fc-card { border:1px solid var(--border-color-primary); border-radius:16px; padding:22px;
  background: var(--background-fill-primary); }
.fc-top { display:flex; justify-content:space-between; align-items:flex-start; gap:16px;
  padding-bottom:16px; margin-bottom:14px; border-bottom:1px solid var(--border-color-primary); }
.fc-kicker { font-size:.78rem; text-transform:uppercase; letter-spacing:.1em; color:var(--body-text-color-subdued); }
.fc-title { font-size:1.75rem; font-weight:800; line-height:1.2; color:var(--body-text-color); margin-top:4px; }
.fc-score { text-align:right; }
.fc-score-num { font-size:2rem; font-weight:800; color:#db2777; line-height:1; }
.fc-score-num small { font-size:1rem; }
.fc-badge { display:inline-block; margin-top:8px; padding:3px 10px; border-radius:999px; font-size:.75rem; font-weight:600; }
.fc-badge.ok  { background:#dcfce7; color:#166534; }
.fc-badge.mid { background:#fef3c7; color:#92400e; }
.fc-badge.low { background:#fee2e2; color:#991b1b; }
.fc-sub { font-size:.85rem; font-weight:600; color:var(--body-text-color-subdued); margin-bottom:8px; }
.fc-row { display:grid; grid-template-columns: 26px minmax(0,1.3fr) minmax(0,1fr) 58px; align-items:center;
  gap:10px; padding:7px 0; color:var(--body-text-color); }
.fc-rank { width:24px; height:24px; border-radius:50%; display:grid; place-items:center; font-size:.75rem;
  font-weight:700; background:var(--background-fill-secondary); }
.fc-first .fc-rank { background:#db2777; color:#fff; }
.fc-first .fc-name { font-weight:700; }
.fc-name { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.fc-bar { height:8px; border-radius:99px; background:var(--background-fill-secondary); overflow:hidden; }
.fc-bar span { display:block; height:100%; border-radius:99px; background:linear-gradient(90deg,#7c3aed,#db2777); }
.fc-pct { text-align:right; font-variant-numeric: tabular-nums; font-size:.9rem; }
.fc-hint { margin:14px 0 0; padding:10px 12px; border-radius:10px; font-size:.85rem;
  background:#fef3c7; color:#92400e; }
.fc-meta { margin-top:14px; font-size:.78rem; color:var(--body-text-color-subdued); }
.fc-empty { text-align:center; padding:48px 22px; }
.fc-empty-icon { font-size:2.6rem; }
.fc-empty-title { font-weight:700; font-size:1.1rem; margin-top:8px; color:var(--body-text-color); }
.fc-empty-text { color:var(--body-text-color-subdued); margin-top:6px; }

.fc-section { font-weight:700; font-size:1.05rem; margin:6px 0 2px; }
@media (max-width: 640px) {
  .fc-hero { padding:22px 18px; } .fc-hero h1 { font-size:1.5rem; }
  .fc-title { font-size:1.35rem; } .fc-row { grid-template-columns: 24px minmax(0,1fr) 54px; }
  .fc-bar { display:none; }
}
"""

THEME = gr.themes.Soft(primary_hue="pink", secondary_hue="violet", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])


def build_ui():
    with gr.Blocks(title="Nhóm 19 – Phân loại hoa") as demo:
        gr.HTML(HERO)

        with gr.Tabs():
            with gr.Tab("🔍 Nhận dạng"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        image = gr.Image(type="pil", label="Ảnh hoa", sources=["upload", "webcam", "clipboard"],
                                         height=380)
                        with gr.Row():
                            clear_btn = gr.Button("Xóa", variant="secondary")
                            run_btn = gr.Button("Nhận dạng", variant="primary")
                    with gr.Column(scale=6):
                        result_html = gr.HTML(EMPTY_RESULT)
                        label = gr.Label(num_top_classes=TOP_K, visible=False)

                if EXAMPLES:
                    gr.Markdown("Ảnh mẫu từ tập test (bấm để thử)", elem_classes="fc-section")
                    gr.Examples(examples=EXAMPLES, inputs=image, examples_per_page=12)

            with gr.Tab("📊 Đánh giá mô hình"):
                if METRICS is not None:
                    gr.Markdown("Kết quả trên tập test (%)", elem_classes="fc-section")
                    gr.Dataframe(METRICS, interactive=False, wrap=True)
                with gr.Row():
                    for fname, cap in [("so_sanh_mo_hinh.png", "So sánh các mô hình"),
                                       ("knn_chon_k_cv.png", "Chọn K cho KNN (5-fold CV)")]:
                        if (OUT_DIR / fname).exists():
                            gr.Image(str(OUT_DIR / fname), label=cap, interactive=False)
                if (OUT_DIR / "confusion_matrix.png").exists():
                    gr.Image(str(OUT_DIR / "confusion_matrix.png"), label="Ma trận nhầm lẫn (chuẩn hóa)",
                             interactive=False)

            with gr.Tab("ℹ️ Giới thiệu"):
                gr.Markdown(f"""
### Quy trình
1. **Tiền xử lý** – resize 256, cắt giữa {IMG_SIZE}×{IMG_SIZE}, chuẩn hóa theo mean/std ImageNet.
2. **Trích xuất đặc trưng** – {BACKBONE_NAME.upper()} đã huấn luyện trên ImageNet, bỏ lớp `fc`, thu vector đặc trưng.
3. **Chuẩn hóa** – StandardScaler fit trên tập train.
4. **Phân loại** – Logistic Regression cho xác suất của {len(class_names)} lớp; hiển thị Top-{TOP_K}.

### Lưu ý khi dùng
- Mô hình chỉ biết **{len(class_names)} loài** trong Oxford Flowers 102; ảnh loài khác vẫn bị gán vào loài gần nhất.
- Kết quả tốt nhất với ảnh chụp rõ một bông hoa, cận cảnh, đủ sáng.
- Độ tin cậy dưới {LOW_CONFIDENCE:.0%} được đánh dấu *không chắc chắn*.

<sub>Mô hình: `{MODEL_PATH.name}` · Thiết bị: {DEVICE.upper()}</sub>
""")

        run_btn.click(predict, inputs=image, outputs=[result_html, label])
        image.change(predict, inputs=image, outputs=[result_html, label])
        clear_btn.click(lambda: (None, EMPTY_RESULT, None), outputs=[image, result_html, label])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="tạo link công khai tạm thời")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    build_ui().launch(inbrowser=True, share=args.share, server_port=args.port, theme=THEME, css=CSS)
