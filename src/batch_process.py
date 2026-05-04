"""
Batch procesiranje vseh pacientov za RV izziv.
Uporaba:
    python src/batch_process.py --data /data --output /results
"""

import os
import argparse
import subprocess
from pathlib import Path

def process_all(data_dir, output_dir):
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Najdi vse mape pacientov
    patients = sorted([p for p in data_path.iterdir() if p.is_dir()])
    print(f"[INFO] Najdenih {len(patients)} map pacientov")

    ok, skipped, failed = 0, 0, 0

    for patient in patients:
        # Najdi vse videe v mapi pacienta
        videos = sorted(patient.glob("*.mp4"))

        if not videos:
            print(f"[SKIP] {patient.name} – ni videov")
            skipped += 1
            continue

        for video in videos:
            # Mapa za rezultate tega videa
            result_dir = output_path / patient.name / video.stem
            result_dir.mkdir(parents=True, exist_ok=True)

            # Preskoči če že obdelano
            if (result_dir / "kinematics.json").exists():
                print(f"[SKIP] {video.name} – že obdelano")
                continue

            print(f"[INFO] Obdelujem: {video.name}")
            try:
                subprocess.run([
                    "python", "src/main.py",
                    "--input", str(video),
                    "--output", str(result_dir),
                    "--no-show"
                ], check=True)
                ok += 1
            except subprocess.CalledProcessError:
                print(f"[FAIL] {video.name}")
                failed += 1

    print(f"\n── ZAKLJUČEK ──────────────────────────────")
    print(f"  Uspešno:    {ok}")
    print(f"  Preskočeno: {skipped}")
    print(f"  Napaka:     {failed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    process_all(args.data, args.output)