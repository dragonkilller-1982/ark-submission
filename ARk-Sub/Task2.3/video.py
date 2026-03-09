import cv2    #video
import numpy as np
import math
import os

# --- Setup ---
folder = r"C:/Users/soumy/Downloads/drive-download-20260306T184609Z-3-001"
videos = [os.path.join(folder, f) for f in ["1.mp4", "2.mp4", "3.mp4"]]
out_frames = r"C:/Users/soumy/Downloads/output_frames"
os.makedirs(out_frames, exist_ok=True)

# Precompute hough lookup
thetas = np.arange(0, np.pi, np.pi / 180)
cos_t, sin_t = np.cos(thetas), np.sin(thetas)


def hough_lines(edges, diag, thresh=100):
    acc = np.zeros((2 * diag, len(thetas)), dtype=np.int32)
    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        return []
    rhos = np.round(xs[:, None] * cos_t + ys[:, None] * sin_t).astype(np.int32) + diag
    np.clip(rhos, 0, 2 * diag - 1, out=rhos)
    np.add.at(acc, (rhos, np.broadcast_to(np.arange(len(thetas)), rhos.shape)), 1)
    ri, ti = np.where(acc >= thresh)
    return list(zip(ri - diag, thetas[ti]))


def draw_line(img, rho, theta, color, thick, ox=0, oy=0):
    a, b = math.cos(theta), math.sin(theta)
    x0, y0 = a * rho, b * rho
    p1 = (int(x0 - 3000 * b) + ox, int(y0 + 3000 * a) + oy)
    p2 = (int(x0 + 3000 * b) + ox, int(y0 - 3000 * a) + oy)
    cv2.line(img, p1, p2, color, thick)


def medial_axis(lines):
    if len(lines) < 2:
        return None
    lines = sorted(lines, key=lambda l: l[1])
    # group by similar angle
    groups, used = [], set()
    for i in range(len(lines)):
        if i in used:
            continue
        g = [lines[i]]
        used.add(i)
        for j in range(i + 1, len(lines)):
            if j not in used and abs(lines[j][1] - lines[i][1]) < np.radians(15):
                g.append(lines[j])
                used.add(j)
        groups.append(g)
    groups.sort(key=len, reverse=True)
    if len(groups[0]) < 2:
        return None
    g = sorted(groups[0], key=lambda l: l[0])
    return ((g[0][0] + g[-1][0]) / 2, (g[0][1] + g[-1][1]) / 2)


def best_tool(fg):
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, score = None, 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 300:
            continue
        rw, rh = cv2.minAreaRect(c)[1]
        if rw == 0 or rh == 0:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if aspect < 1.5:
            continue
        s = aspect * math.sqrt(area)
        if s > score:
            score, best = s, c
    if best is None:
        return None, None
    return best, cv2.boundingRect(best)


# Morphology kernels
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# --- Process each video ---
for vpath in videos:
    name = os.path.splitext(os.path.basename(vpath))[0]
    print(f"\n--- {name}.mp4 ---")

    cap = cv2.VideoCapture(vpath)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(os.path.join(folder, f"{name}_medial_axis.mp4"),
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))

    # Build background from sampled frames
    samples = []
    for i in np.linspace(0, total - 1, min(60, total), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            samples.append(f)
    bg = np.median(samples, axis=0).astype(np.uint8)
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

    # Train MOG2
    mog = cv2.createBackgroundSubtractorMOG2(300, 40, False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(min(60, total)):
        ok, f = cap.read()
        if ok:
            mog.apply(f)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_gray, last_box, lost = None, None, 0
    count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        count += 1
        orig = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3 masks: static bg, frame diff, MOG2
        m1 = cv2.threshold(cv2.absdiff(gray, bg_gray), 20, 255, cv2.THRESH_BINARY)[1]
        m3 = cv2.threshold(mog.apply(frame), 200, 255, cv2.THRESH_BINARY)[1]
        if prev_gray is not None:
            m2 = cv2.dilate(cv2.threshold(cv2.absdiff(gray, prev_gray), 10, 255,
                            cv2.THRESH_BINARY)[1], k_dil, iterations=3)
        else:
            m2 = np.zeros_like(m1)

        fg = cv2.bitwise_or(m1, cv2.bitwise_or(m2, m3))

        # Filter by low saturation (metallic tool)
        sat = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
        fg = cv2.bitwise_and(fg, cv2.threshold(sat, 80, 255, cv2.THRESH_BINARY_INV)[1])

        # Cleanup
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open)
        fg = cv2.dilate(fg, k_dil)

        # Boost near last known position if tool was recently seen
        if last_box is not None and lost < 15:
            bx, by, bw, bh = last_box
            pad = 60
            y1, y2 = max(0, by - pad), min(fh, by + bh + pad)
            x1, x2 = max(0, bx - pad), min(fw, bx + bw + pad)
            boost = np.zeros_like(fg)
            boost[y1:y2, x1:x2] = cv2.threshold(
                cv2.absdiff(gray[y1:y2, x1:x2], bg_gray[y1:y2, x1:x2]),
                15, 255, cv2.THRESH_BINARY)[1]
            fg = cv2.morphologyEx(cv2.bitwise_or(fg, boost), cv2.MORPH_CLOSE, k_close)

        # Find tool and draw
        contour, bbox = best_tool(fg)
        if contour is not None:
            bx, by, bw, bh = bbox
            last_box, lost = bbox, 0

            # Edge detect on tool mask
            mask = np.zeros(fg.shape, np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            crop = mask[by:by + bh, bx:bx + bw]
            sx = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sx ** 2 + sy ** 2)
            edges = np.uint8(edges / max(edges.max(), 1) * 255)
            edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)[1]

            # Hough + medial axis
            diag = int(math.sqrt(bw ** 2 + bh ** 2))
            n = cv2.countNonZero(edges)
            lines = hough_lines(edges, diag, max(15, n // 40)) if n > 5 else []

            for rho, theta in lines:
                draw_line(orig, rho, theta, (0, 255, 0), 1, bx, by)

            med = medial_axis(lines) if lines else None
            if med:
                draw_line(orig, med[0], med[1], (0, 0, 255), 3, bx, by)

            cv2.rectangle(orig, (bx, by), (bx + bw, by + bh), (255, 255, 0), 1)
        else:
            lost += 1

        prev_gray = gray.copy()
        cv2.putText(orig, f"{name} - Frame {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(orig)
        cv2.imwrite(os.path.join(out_frames, f"{name}_frame_{count:04d}.jpg"), orig)
        cv2.imshow("result", orig)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count % 30 == 0:
            print(f"  {count}/{total}")

    cap.release()
    out.release()
    print(f"  done - {count} frames")

cv2.destroyAllWindows()
print("\nall done!")