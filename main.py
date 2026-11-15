import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, Normalizer, FunctionTransformer, normalize
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)
import joblib

# ======================= CẤU HÌNH =======================
IMG_SIZE = 224
BATCH_SIZE = 64
N_AUG = 2                      # số bản tăng cường thêm cho mỗi ảnh train
SEED = 42
K_VALUES = list(range(1, 30, 2))
KNN_METRICS = ['euclidean', 'manhattan', 'cosine']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXT = ('.jpg', '.jpeg', '.png')

SHOW_PLOTS = False             # True: hiện cửa sổ biểu đồ (phải đóng cửa sổ mới chạy tiếp)
OUT_DIR = 'outputs'


def save_fig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f'   → Đã lưu hình: {path}')


def header(title):
    print('\n' + '=' * 70 + f'\n{title}\n' + '=' * 70)


# =================================================================
# BƯỚC 1 – KHÁM PHÁ DỮ LIỆU (EDA)
# =================================================================
def list_images(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))


def step1_eda(data_dir):
    header('BƯỚC 1 – KHÁM PHÁ DỮ LIỆU (EDA)')

    # 1.1. Tạo bảng dữ liệu từ cấu trúc thư mục (valid/ có nhãn → dùng làm tập TEST)
    rows = []
    for split, sub in [('train', 'train'), ('test', 'valid')]:
        split_dir = os.path.join(data_dir, sub)
        for cls in os.listdir(split_dir):
            cdir = os.path.join(split_dir, cls)
            if os.path.isdir(cdir):
                for p in list_images(cdir):
                    rows.append({'path': p, 'class_folder': cls, 'split': split})
    df = pd.DataFrame(rows)
    classes_sorted = sorted(df['class_folder'].unique(), key=int)
    df['label'] = df['class_folder'].map({c: i for i, c in enumerate(classes_sorted)})
    num_classes = len(classes_sorted)
    print(f'Tổng số ảnh có nhãn: {len(df)} | Số lớp: {num_classes}')
    print(df['split'].value_counts().to_string())

    # 1.2. Tên các loài hoa
    cat_to_name = json.load(open(os.path.join(data_dir, 'cat_to_name.json'), encoding='utf-8'))
    class_names = [cat_to_name[c] for c in classes_sorted]
    df['class_name'] = df['label'].map(lambda i: class_names[i])
    print('10 loài đầu:', class_names[:10])

    # 1.3. Phân phối nhãn
    counts = df['class_name'].value_counts()
    print('Số ảnh/lớp: min = %d, max = %d, trung bình = %.1f, trung vị = %d'
          % (counts.min(), counts.max(), counts.mean(), counts.median()))
    print('Tỷ lệ mất cân bằng (max/min): %.1f lần' % (counts.max() / counts.min()))
    plt.figure(figsize=(18, 5))
    counts.plot(kind='bar', color='tab:purple')
    plt.axhline(counts.mean(), color='red', ls='--', label=f'Trung bình = {counts.mean():.0f}')
    plt.title('Số lượng ảnh theo từng loài hoa')
    plt.ylabel('Số ảnh'); plt.xticks(fontsize=6); plt.legend()
    save_fig('eda_phan_phoi_nhan.png')

    # 1.4. Kích thước ảnh và số kênh màu
    sizes = []
    for p in tqdm(df['path'], desc='Đọc kích thước ảnh'):
        with Image.open(p) as im:                 # chỉ đọc header, không giải mã cả ảnh
            sizes.append((im.width, im.height, len(im.getbands())))
    df[['width', 'height', 'channels']] = pd.DataFrame(sizes, index=df.index)
    print(df[['width', 'height']].describe().round(1).to_string())
    print('Số kênh màu:', df['channels'].value_counts().to_dict())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(df['width'], bins=40, color='tab:blue'); ax[0].set_title('Phân phối chiều rộng (px)')
    ax[1].hist(df['height'], bins=40, color='tab:orange'); ax[1].set_title('Phân phối chiều cao (px)')
    save_fig('eda_kich_thuoc_anh.png')

    # 1.5. Ảnh mẫu của một số lớp
    chosen = np.random.default_rng(SEED).choice(num_classes, 8, replace=False)
    fig, axes = plt.subplots(8, 5, figsize=(12, 19))
    for r, c in enumerate(chosen):
        paths = df[df['label'] == c]['path'].sample(5, random_state=SEED).tolist()
        for k, p in enumerate(paths):
            axes[r, k].imshow(Image.open(p).convert('RGB')); axes[r, k].axis('off')
        axes[r, 0].set_title(class_names[c], fontsize=9, loc='left')
    save_fig('eda_anh_mau.png')

    return df, class_names, num_classes


# =================================================================
# BƯỚC 2 – TIỀN XỬ LÝ ẢNH
# =================================================================
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),                                  # đưa pixel về [0, 1]
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_aug_tf = transforms.Compose([                          # chỉ dùng cho tập train
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class FlowerDataset(Dataset):
    def __init__(self, frame, transform):
        self.paths = frame['path'].tolist()
        self.labels = frame['label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        return self.transform(img), self.labels[i]


def denorm(t):
    return np.clip(t.permute(1, 2, 0).numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1)


def step2_preprocess(df):
    header('BƯỚC 2 – TIỀN XỬ LÝ ẢNH')

    # 2.1. Tách 15% train (phân tầng) làm tập validation để chọn tham số NB, LR, SVM
    tr = df[df['split'] == 'train'].index
    _, idx_val = train_test_split(tr, test_size=0.15, stratify=df.loc[tr, 'label'], random_state=SEED)
    df.loc[idx_val, 'split'] = 'val'
    df_train = df[df['split'] == 'train'].reset_index(drop=True)
    df_val = df[df['split'] == 'val'].reset_index(drop=True)
    df_test = df[df['split'] == 'test'].reset_index(drop=True)
    print(f'Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}')

    # 2.2 + 2.3. Resize, chuẩn hóa, tăng cường dữ liệu – minh họa trên 1 ảnh
    img = Image.open(df_train['path'].iloc[0]).convert('RGB')
    fig, ax = plt.subplots(1, 6, figsize=(18, 3.5))
    ax[0].imshow(img); ax[0].set_title(f'Gốc {img.size}')
    ax[1].imshow(denorm(eval_tf(img))); ax[1].set_title('Resize + crop 224')
    for k in range(2, 6):
        ax[k].imshow(denorm(train_aug_tf(img))); ax[k].set_title(f'Augment #{k-1}')
    for a in ax:
        a.axis('off')
    save_fig('tien_xu_ly_augment.png')

    return df_train, df_val, df_test


# =================================================================
# BƯỚC 3 – TRÍCH XUẤT ĐẶC TRƯNG: TRANSFER LEARNING (ResNet50)
# =================================================================
def load_backbone(device):
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Identity()                 # bỏ lớp phân loại → vector 2048 chiều
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


@torch.no_grad()
def extract_features(backbone, device, frame, transform, desc):
    loader = DataLoader(FlowerDataset(frame, transform), batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0)       # num_workers=0 để chạy ổn định trên Windows
    feats, labels = [], []
    for x, y in tqdm(loader, desc=desc):
        feats.append(backbone(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats).astype(np.float32), np.concatenate(labels)


def step3_features(backbone, device, df_train, df_val, df_test):
    header('BƯỚC 3 – TRÍCH XUẤT ĐẶC TRƯNG (ResNet50)')
    feat_file = os.path.join(OUT_DIR, f'features_resnet50_aug{N_AUG}.npz')
    if os.path.exists(feat_file):
        d = np.load(feat_file)
        print('Đã tải đặc trưng có sẵn:', feat_file)
        return tuple(d[k] for k in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'])

    if device == 'cpu':
        print('Không có GPU → trích xuất trên CPU sẽ khá lâu (có thể 30–60 phút). Chỉ cần chạy 1 lần.')
    t = time.time()
    Xs, ys = [], []
    X0, y0 = extract_features(backbone, device, df_train, eval_tf, 'Train (gốc)')
    Xs.append(X0); ys.append(y0)
    for k in range(N_AUG):
        Xa, ya = extract_features(backbone, device, df_train, train_aug_tf, f'Train (augment {k+1}/{N_AUG})')
        Xs.append(Xa); ys.append(ya)
    X_train, y_train = np.concatenate(Xs), np.concatenate(ys)
    X_val, y_val = extract_features(backbone, device, df_val, eval_tf, 'Validation')
    X_test, y_test = extract_features(backbone, device, df_test, eval_tf, 'Test')
    np.savez_compressed(feat_file, X_train=X_train, y_train=y_train, X_val=X_val,
                        y_val=y_val, X_test=X_test, y_test=y_test)
    print(f'Trích xuất xong sau {time.time()-t:.0f}s, đã lưu: {feat_file}')
    return X_train, y_train, X_val, y_val, X_test, y_test


# =================================================================
# BƯỚC 4 – HUẤN LUYỆN MÔ HÌNH
# =================================================================
def knn_vote(nb_labels, k, num_classes):
    return np.array([np.bincount(r[:k], minlength=num_classes).argmax() for r in nb_labels])


def make_knn(metric, k, scaler):
    # Mô hình nhận đầu vào là đặc trưng đã StandardScaler (giống các mô hình khác)
    if metric == 'cosine':
        return make_pipeline(FunctionTransformer(scaler.inverse_transform),   # quay về đặc trưng gốc
                             Normalizer(),                                     # chuẩn hóa L2 ⇔ cosine
                             KNeighborsClassifier(n_neighbors=k, n_jobs=-1))
    return KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)


def select_knn_cv(X_train, y_train, n0, num_classes):
    """Chọn K và độ đo khoảng cách cho KNN bằng Stratified 5-Fold CV trên tập train."""
    groups = np.tile(np.arange(n0), N_AUG + 1)     # ảnh gốc và các bản augment cùng 1 fold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = {m: [] for m in KNN_METRICS}

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train, groups), start=1):
        va_idx = va_idx[va_idx < n0]                # chỉ đánh giá trên ảnh gốc
        sc = StandardScaler().fit(X_train[tr_idx])  # fit scaler riêng trong từng fold
        inputs = {
            'euclidean': (sc.transform(X_train[tr_idx]), sc.transform(X_train[va_idx]), 'euclidean'),
            'manhattan': (sc.transform(X_train[tr_idx]), sc.transform(X_train[va_idx]), 'manhattan'),
            'cosine':    (normalize(X_train[tr_idx]), normalize(X_train[va_idx]), 'euclidean'),
        }
        for m in KNN_METRICS:
            Xtr, Xva, metric = inputs[m]
            nn_model = KNeighborsClassifier(n_neighbors=max(K_VALUES), metric=metric, n_jobs=-1).fit(Xtr, y_train[tr_idx])
            _, nb_idx = nn_model.kneighbors(Xva)    # tìm hàng xóm 1 lần cho mọi K
            nb_labels = y_train[tr_idx][nb_idx]
            cv_scores[m].append([accuracy_score(y_train[va_idx], knn_vote(nb_labels, k, num_classes))
                                 for k in K_VALUES])
        print(f'   Fold {fold}/5 xong')

    plt.figure(figsize=(8, 4))
    for m in KNN_METRICS:
        s = np.array(cv_scores[m]); mean, std = s.mean(0), s.std(0)
        plt.plot(K_VALUES, mean, marker='o', label=m)
        plt.fill_between(K_VALUES, mean - std, mean + std, alpha=0.2)
        print(f'   {m:10s}: K tốt nhất = {K_VALUES[int(mean.argmax())]:2d}, CV acc = {mean.max():.4f}')
    plt.xlabel('K'); plt.ylabel('Accuracy (5-fold CV)'); plt.title('Chọn K cho KNN bằng Stratified 5-fold CV')
    plt.legend()
    save_fig('knn_chon_k_cv.png')

    best_metric = max(KNN_METRICS, key=lambda m: np.array(cv_scores[m]).mean(0).max())
    best_k = K_VALUES[int(np.array(cv_scores[best_metric]).mean(0).argmax())]
    print(f'=> Chọn KNN: K = {best_k}, khoảng cách = {best_metric}')
    return best_k, best_metric


def tune(make_model, grid, name, X_train_s, y_train, X_val_s, y_val):
    best = (None, -1, None)
    for v in grid:
        m = make_model(v).fit(X_train_s, y_train)
        acc = accuracy_score(y_val, m.predict(X_val_s))
        print(f'   {name} param={v}: val acc = {acc:.4f}')
        if acc > best[1]:
            best = (v, acc, m)
    print(f'=> {name}: tham số tốt nhất = {best[0]}')
    return best


def step4_train(X_train, y_train, X_val, y_val, n0, num_classes):
    header('BƯỚC 4 – HUẤN LUYỆN MÔ HÌNH')

    # 4.1. Chuẩn hóa đặc trưng (fit trên train)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)

    # 4.2. KNN – Stratified 5-Fold CV
    print('\n[KNN] Chọn K bằng Stratified 5-Fold Cross-Validation')
    best_k, best_metric = select_knn_cv(X_train, y_train, n0, num_classes)

    # 4.3. Gaussian NB, Logistic Regression, SVM – chọn tham số trên tập validation
    args = (X_train_s, y_train, X_val_s, y_val)
    print('\n[Gaussian NB]')
    best_nb = tune(lambda v: GaussianNB(var_smoothing=v), [1e-9, 1e-7, 1e-5, 1e-3, 1e-1], 'GaussianNB', *args)
    print('\n[Logistic Regression]')
    best_lr = tune(lambda v: LogisticRegression(C=v, max_iter=3000), [0.01, 0.1, 1.0],
                   'LogisticRegression', *args)
    print('\n[SVM]')
    best_svm = tune(lambda v: SVC(C=v, kernel='rbf', gamma='scale'), [1, 10, 100], 'SVM-RBF', *args)

    final_models = {
        f'KNN (K={best_k}, {best_metric})': make_knn(best_metric, best_k, scaler).fit(X_train_s, y_train),
        'Gaussian Naive Bayes': best_nb[2],
        'Logistic Regression': best_lr[2],
        'SVM (RBF)': best_svm[2],
    }
    return scaler, final_models


# =================================================================
# BƯỚC 5 – ĐÁNH GIÁ
# =================================================================
def step5_evaluate(final_models, scaler, X_test, y_test, class_names, num_classes):
    header('BƯỚC 5 – ĐÁNH GIÁ TRÊN TẬP TEST')
    X_test_s = scaler.transform(X_test)

    # 5.1. Bảng so sánh các mô hình
    results, preds = [], {}
    for name, m in final_models.items():
        y_pred = m.predict(X_test_s)
        preds[name] = y_pred
        results.append({
            'Mô hình': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision (macro)': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'Recall (macro)': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'F1 (macro)': f1_score(y_test, y_pred, average='macro', zero_division=0),
        })
    res_df = pd.DataFrame(results).sort_values('F1 (macro)', ascending=False).reset_index(drop=True)
    res_df.to_csv(os.path.join(OUT_DIR, 'ket_qua_test.csv'), index=False, encoding='utf-8-sig')
    print(res_df.round(4).to_string())

    plot_df = res_df.melt(id_vars='Mô hình', var_name='Chỉ số', value_name='Giá trị')
    plt.figure(figsize=(11, 5))
    ax = sns.barplot(data=plot_df, x='Mô hình', y='Giá trị', hue='Chỉ số')
    for cont in ax.containers:
        ax.bar_label(cont, fmt='%.3f', fontsize=7)
    plt.ylim(0, 1.05); plt.title('So sánh các mô hình (tập test)')
    save_fig('so_sanh_mo_hinh.png')

    # 5.2. Báo cáo chi tiết mô hình tốt nhất
    best_name = res_df.loc[0, 'Mô hình']
    y_best = preds[best_name]
    print('\nMô hình tốt nhất:', best_name)
    rep = pd.DataFrame(classification_report(y_test, y_best, labels=list(range(num_classes)),
                                             target_names=class_names, output_dict=True, zero_division=0)).T
    rep.to_csv(os.path.join(OUT_DIR, 'classification_report_best.csv'), encoding='utf-8-sig')
    print('10 loài khó nhận dạng nhất:')
    print(rep.iloc[:num_classes].sort_values('f1-score').head(10).round(3).to_string())

    # 5.3. Ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_best, labels=range(num_classes))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.title(f'Confusion matrix chuẩn hóa – {best_name}')
    plt.xlabel('Nhãn dự đoán'); plt.ylabel('Nhãn thật')
    save_fig('confusion_matrix.png')

    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], i, j) for i in range(num_classes) for j in range(num_classes) if off[i, j] > 0),
                   reverse=True)
    print('\n10 cặp loài bị nhầm nhiều nhất:')
    print(pd.DataFrame([{'Nhãn thật': class_names[i], 'Bị đoán thành': class_names[j], 'Số lần': n}
                        for n, i, j in pairs[:10]]).to_string())


# =================================================================
# LƯU MÔ HÌNH + GIAO DIỆN WEB
# =================================================================
def save_model(final_models, scaler, class_names):
    path = os.path.join(OUT_DIR, 'flower_classifier.joblib')
    joblib.dump({'scaler': scaler, 'model': final_models['Logistic Regression'], 'class_names': class_names}, path)
    print(f'\nĐã lưu mô hình cho giao diện web: {path}')


def launch_gui(backbone, device, scaler, model, class_names):
    import gradio as gr

    @torch.no_grad()
    def predict_flower(img):
        if img is None:
            return {}
        x = eval_tf(img.convert('RGB')).unsqueeze(0).to(device)
        proba = model.predict_proba(scaler.transform(backbone(x).cpu().numpy()))[0]
        top5 = np.argsort(proba)[::-1][:5]
        return {class_names[i]: float(proba[i]) for i in top5}

    gr.Interface(
        fn=predict_flower,
        inputs=gr.Image(type='pil', label='Tải ảnh hoa lên'),
        outputs=gr.Label(num_top_classes=5, label='Kết quả dự đoán (Top-5)'),
        title='Nhóm 19 – Phân loại 102 loài hoa',
        description='ResNet50 (Transfer Learning) + Logistic Regression – Oxford Flowers 102',
    ).launch(inbrowser=True)


# =================================================================
def main():
    global OUT_DIR, SHOW_PLOTS
    parser = argparse.ArgumentParser(description='Nhóm 19 – Phân loại hoa Oxford Flowers 102')
    parser.add_argument('--data', default='Dataset', help='thư mục dataset (chứa train/, valid/, cat_to_name.json)')
    parser.add_argument('--out', default='outputs', help='thư mục lưu kết quả')
    parser.add_argument('--gui', action='store_true', help='mở giao diện web sau khi huấn luyện')
    parser.add_argument('--show', action='store_true', help='hiện cửa sổ biểu đồ khi chạy')
    args = parser.parse_args()

    OUT_DIR, SHOW_PLOTS = args.out, args.show
    if not SHOW_PLOTS:
        matplotlib.use('Agg')                    # chỉ lưu hình, không mở cửa sổ
    os.makedirs(OUT_DIR, exist_ok=True)
    sns.set_style('whitegrid')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Thiết bị:', device, '| Dataset:', os.path.abspath(args.data))

    df, class_names, num_classes = step1_eda(args.data)
    df_train, df_val, df_test = step2_preprocess(df)
    backbone = load_backbone(device)
    X_train, y_train, X_val, y_val, X_test, y_test = step3_features(backbone, device, df_train, df_val, df_test)
    scaler, final_models = step4_train(X_train, y_train, X_val, y_val, len(df_train), num_classes)
    step5_evaluate(final_models, scaler, X_test, y_test, class_names, num_classes)
    save_model(final_models, scaler, class_names)

    print(f'\nHOÀN TẤT. Toàn bộ hình và bảng kết quả nằm trong thư mục: {os.path.abspath(OUT_DIR)}')
    if args.gui:
        launch_gui(backbone, device, scaler, final_models['Logistic Regression'], class_names)
    else:
        print('Muốn mở giao diện web: python main.py --gui   (hoặc python app.py)')


if __name__ == '__main__':
    main()
