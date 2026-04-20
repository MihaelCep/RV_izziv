"""
Nine-Hole Peg Test (9HPT) – Robotski vid izziv
Določanje kinematičnih parametrov gibanja roke iz enega pogleda.

Rešitev temelji na MediaPipe Hands za zaznavo roke in prstov,
sledenje skozi čas ter izračun d/v/a parametrov.

Uporaba:
    python src/main.py --input data/video.mp4 --output data/results/
    python src/main.py --input 0   (webcam)
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import json
import csv
from collections import deque
import time

# ── Konstante ─────────────────────────────────────────────────────────────────

# MediaPipe indeksi landmarks
WRIST       = 0
THUMB_TIP   = 4
INDEX_TIP   = 8
MIDDLE_TIP  = 12
RING_TIP    = 16
PINKY_TIP   = 20

# Gladilno okno za hitrost/pospešek (zmanjša šum)
SMOOTH_WINDOW = 5

# Barve (BGR)
COLOR_HAND   = (0, 255, 0)
COLOR_THUMB  = (255, 100,   0)
COLOR_INDEX  = (0,   100, 255)
COLOR_TRAIL  = (200, 200,   0)
COLOR_TEXT   = (255, 255, 255)

# ── Pomožne funkcije ───────────────────────────────────────────────────────────

def pixel_distance(p1, p2):
    """Evklidska razdalja med dvema točkama (px)."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def smooth(values, window=SMOOTH_WINDOW):
    """Drseče povprečje za glajenje signala."""
    if len(values) < window:
        return values[-1] if values else 0.0
    return float(np.mean(list(values)[-window:]))


def draw_trail(frame, trail, color, max_len=60):
    """Nariše sled gibanja točke."""
    pts = list(trail)[-max_len:]
    for i in range(1, len(pts)):
        if pts[i-1] is not None and pts[i] is not None:
            alpha = i / len(pts)
            c = tuple(int(v * alpha) for v in color)
            cv2.line(frame, pts[i-1], pts[i], c, 2)


def overlay_text(frame, lines, x=10, y=20, dy=22):
    """Izpiše seznam vrstic besedila na frame."""
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)


# ── Razred za sledenje kinematike ─────────────────────────────────────────────

class KinematicsTracker:
    """Sledenje pozicije, poti, hitrosti in pospeška za eno točko."""

    def __init__(self, name: str, fps: float):
        self.name = name
        self.fps = fps
        self.dt = 1.0 / fps

        # Časovne vrste (indeks = številka okvirja)
        self.positions   = []   # (x, y) v px
        self.path_length = []   # kumulativna dolžina poti [px]
        self.velocity    = []   # trenutna hitrost [px/s]
        self.accel       = []   # pospešek [px/s²]
        self.timestamps  = []   # čas [s]

        self._vel_buf  = deque(maxlen=SMOOTH_WINDOW)
        self._acc_buf  = deque(maxlen=SMOOTH_WINDOW)
        self._trail    = deque(maxlen=120)
        self._cum_dist = 0.0
        self._frame    = 0

    def update(self, pos):
        """Posodobi tracker z novo pozicijo (x, y) v pikslih."""
        t = self._frame * self.dt
        self.timestamps.append(t)
        self.positions.append(pos)
        self._trail.append(pos)
        self._frame += 1

        if len(self.positions) >= 2:
            d = pixel_distance(self.positions[-1], self.positions[-2])
            self._cum_dist += d

            # Hitrost
            v = d / self.dt
            self._vel_buf.append(v)
            v_smooth = smooth(self._vel_buf)
            self.velocity.append(v_smooth)

            # Pospešek
            if len(self.velocity) >= 2:
                a = (self.velocity[-1] - self.velocity[-2]) / self.dt
                self._acc_buf.append(a)
                self.accel.append(smooth(self._acc_buf))
            else:
                self.accel.append(0.0)
        else:
            self.velocity.append(0.0)
            self.accel.append(0.0)

        self.path_length.append(self._cum_dist)

    def current_stats(self):
        """Vrne trenutne vrednosti za prikaz."""
        v = self.velocity[-1] if self.velocity else 0.0
        a = self.accel[-1]    if self.accel    else 0.0
        d = self.path_length[-1] if self.path_length else 0.0
        return d, v, a

    @property
    def trail(self):
        return self._trail

    def to_dict(self):
        return {
            "name":        self.name,
            "fps":         self.fps,
            "timestamps":  self.timestamps,
            "positions":   self.positions,
            "path_length": self.path_length,
            "velocity":    self.velocity,
            "acceleration":self.accel,
        }


# ── Glavna funkcija ───────────────────────────────────────────────────────────

def process_video(input_source, output_dir: str, px_per_mm: float = 1.0,
                  show: bool = True):
    """
    Obdela video, zaznava roko in izračuna kinematične parametre.

    Args:
        input_source: pot do video datoteke ali indeks kamere (int)
        output_dir:   mapa za shranjevanje rezultatov
        px_per_mm:    umeritveni faktor (privzeto: 1.0 → rezultati v px)
        show:         ali prikazovati okno med obdelavo
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Odpri vhodni video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        raise RuntimeError(f"Ne morem odpreti vira: {input_source}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Vir: {input_source}  |  {W}×{H} @ {fps:.1f} fps  |  {total} okvirjev")

    # ── Video writer ──────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, "annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # ── MediaPipe inicializacija ──────────────────────────────────────────────
    mp_hands   = mp.solutions.hands
    mp_draw    = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    # ── Trackerji ─────────────────────────────────────────────────────────────
    wrist_t = KinematicsTracker("wrist",  fps)
    thumb_t = KinematicsTracker("thumb",  fps)
    index_t = KinematicsTracker("index",  fps)

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        detected = False

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                detected = True

                # Nariši skelet roke
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                # Izvleči koordinate v pikslih
                def lm_px(idx):
                    lm = hand_lm.landmark[idx]
                    return (int(lm.x * W), int(lm.y * H))

                wrist_px = lm_px(WRIST)
                thumb_px = lm_px(THUMB_TIP)
                index_px = lm_px(INDEX_TIP)

                # Posodobi trackerje
                wrist_t.update(wrist_px)
                thumb_t.update(thumb_px)
                index_t.update(index_px)

                # Nariši sledi
                draw_trail(frame, wrist_t.trail, COLOR_TRAIL)
                draw_trail(frame, thumb_t.trail, COLOR_THUMB)
                draw_trail(frame, index_t.trail, COLOR_INDEX)

                # Označi konice prstov
                cv2.circle(frame, thumb_px, 8, COLOR_THUMB, -1)
                cv2.circle(frame, index_px, 8, COLOR_INDEX, -1)
                cv2.circle(frame, wrist_px, 6, COLOR_HAND,  -1)

                # Prikaži trenutne statistike
                wd, wv, wa = wrist_t.current_stats()
                td, tv, ta = thumb_t.current_stats()
                id_, iv, ia = index_t.current_stats()

                lines = [
                    f"Frame: {frame_idx:5d}   t={frame_idx/fps:.2f}s",
                    f"WRIST  d={wd/px_per_mm:6.1f}mm  v={wv/px_per_mm:6.1f}mm/s  a={wa/px_per_mm:7.1f}mm/s2",
                    f"THUMB  d={td/px_per_mm:6.1f}mm  v={tv/px_per_mm:6.1f}mm/s  a={ta/px_per_mm:7.1f}mm/s2",
                    f"INDEX  d={id_/px_per_mm:6.1f}mm  v={iv/px_per_mm:6.1f}mm/s  a={ia/px_per_mm:7.1f}mm/s2",
                ]
                overlay_text(frame, lines)
                break  # samo ena roka

        if not detected:
            overlay_text(frame, [f"Frame: {frame_idx:5d} – roka ni zaznana"])

        writer.write(frame)

        if show:
            cv2.imshow("9HPT – Kinematika roke", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Zaustavljeno s tipko 'q'.")
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"[INFO] Obdelano {frame_idx}/{total} okvirjev  ({elapsed:.1f}s)")

    # ── Zaključek ─────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    hands.close()
    if show:
        cv2.destroyAllWindows()

    print(f"[INFO] Anotiran video shranjen: {out_path}")

    # ── Shrani rezultate ──────────────────────────────────────────────────────
    results = {
        "source":      str(input_source),
        "fps":         fps,
        "px_per_mm":   px_per_mm,
        "total_frames":frame_idx,
        "trackers": {
            "wrist": wrist_t.to_dict(),
            "thumb": thumb_t.to_dict(),
            "index": index_t.to_dict(),
        }
    }

    json_path = os.path.join(output_dir, "kinematics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Kinematični podatki (JSON): {json_path}")

    # CSV za wrist (za Excel / vrednotenje)
    csv_path = os.path.join(output_dir, "wrist_kinematics.csv")
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["frame", "time_s", "x_px", "y_px",
                              "path_mm", "velocity_mm_s", "accel_mm_s2"])
        for i, (t, pos, d, v, a) in enumerate(zip(
                wrist_t.timestamps, wrist_t.positions,
                wrist_t.path_length, wrist_t.velocity, wrist_t.accel)):
            writer_csv.writerow([
                i, f"{t:.4f}", pos[0], pos[1],
                f"{d/px_per_mm:.3f}",
                f"{v/px_per_mm:.3f}",
                f"{a/px_per_mm:.3f}",
            ])
    print(f"[INFO] Wrist CSV: {csv_path}")

    # Izpiši povzetek
    print("\n── POVZETEK ──────────────────────────────────────────────────")
    for name, tracker in [("Wrist", wrist_t), ("Thumb", thumb_t), ("Index", index_t)]:
        if tracker.path_length:
            max_v = max(tracker.velocity) / px_per_mm
            avg_v = np.mean(tracker.velocity) / px_per_mm
            total_d = tracker.path_length[-1] / px_per_mm
            print(f"  {name:6s}: skupna pot={total_d:.1f} mm  "
                  f"avg hitrost={avg_v:.1f} mm/s  max hitrost={max_v:.1f} mm/s")
    print("──────────────────────────────────────────────────────────────\n")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="9HPT Robotski vid – kinematika roke"
    )
    parser.add_argument("--input",  default="0",
                        help="Pot do video datoteke ali indeks kamere (privzeto: 0)")
    parser.add_argument("--output", default="data/results",
                        help="Mapa za shranjevanje rezultatov")
    parser.add_argument("--px_per_mm", type=float, default=1.0,
                        help="Umeritveni faktor px/mm (privzeto: 1.0)")
    parser.add_argument("--no-show", action="store_true",
                        help="Ne prikazuj okna med obdelavo")
    args = parser.parse_args()

    # Pretvori "0" v int za webcam
    source = int(args.input) if args.input.isdigit() else args.input

    process_video(
        input_source=source,
        output_dir=args.output,
        px_per_mm=args.px_per_mm,
        show=not args.no_show,
    )
