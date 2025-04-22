import os
import glob
import cv2

def augment_dataset(dataset_name: str):
    base_dir = os.path.join("Datasets", dataset_name)
    img_dir  = os.path.join(base_dir, "images")
    lbl_dir  = os.path.join(base_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # ── 이미 증강된 데이터가 하나라도 있으면 스킵 ──
    already = any(
        glob.glob(os.path.join(img_dir, f"*{suffix}{ext}"))
        for suffix in ("_aug",)
        for ext in exts
    )
    if already:
        print("✅ 데이터셋이 이미 증강되어 있습니다. 작업을 건너뜁니다.")
        return

    # ── train split 파일만 선택 ──
    split_files = glob.glob(os.path.join(base_dir, "train_iter_*.txt"))

    # ── 전 파일에 대해 한 번씩만 증강 수행 ──
    for img_path in glob.glob(os.path.join(img_dir, "*")):
        stem, ext = os.path.splitext(os.path.basename(img_path))
        if ext.lower() not in exts:
            continue

        lbl_path  = os.path.join(lbl_dir, f"{stem}.txt")
        if not os.path.isfile(lbl_path):
            print(f"⚠️ 레이블 없음, 스킵: {lbl_path}")
            continue

        # 원본 불러와 좌우 뒤집기
        img      = cv2.imread(img_path)
        flip_img = cv2.flip(img, 1)

        # 레이블 읽어 x_center 뒤집기
        new_lines = []
        with open(lbl_path, "r") as f:
            for line in f:
                cls, xc, yc, bw, bh = line.strip().split()
                xc = 1.0 - float(xc)
                new_lines.append(f"{cls} {xc:.6f} {yc} {bw} {bh}")

        # 파일명 _aug
        out_img   = os.path.join(img_dir, f"{stem}_aug{ext}")
        out_lbl   = os.path.join(lbl_dir, f"{stem}_aug.txt")
        orig_line = os.path.join(base_dir, "images", f"{stem}{ext}")
        aug_line  = os.path.join(base_dir, "images", f"{stem}_aug{ext}")

        # 저장
        cv2.imwrite(out_img, flip_img)
        with open(out_lbl, "w") as wf:
            wf.write("\n".join(new_lines) + "\n")
        print(f"➕ 생성됨: {stem}_aug")

        # train split 파일에만 추가
        for split_file in split_files:
            with open(split_file, "r") as rf:
                lines = {l.strip() for l in rf if l.strip()}
            if orig_line in lines:
                with open(split_file, "a") as af:
                    af.write(aug_line + "\n")

    # ── 요약 ──
    originals = len([p for p in glob.glob(os.path.join(img_dir, "*"))
                     if os.path.splitext(p)[1].lower() in exts and not p.endswith("_aug" + os.path.splitext(p)[1])])
    flips = len([p for p in glob.glob(os.path.join(img_dir, "*_aug*"))
                 if os.path.splitext(p)[1].lower() in exts])
    print(f"\n완료: 원본 {originals}장 → flip {flips}장 (총 {originals + flips}장)")
