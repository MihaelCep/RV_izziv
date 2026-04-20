# RV – Izziv: Nine-Hole Peg Test (9HPT)

Robotski vid – Seminarska naloga – FE Ljubljana 2025/26

## Opis rešitve

Rešitev temelji na **MediaPipe Hands** za zaznavo 21 točk roke v realnem času iz enega pogleda (top-down kamera). Za vsako ključno točko (zapestje, palec, kazalec) se skozi čas izračunajo:

- **d** – skupna dolžina poti [mm]
- **v** – trenutna hitrost [mm/s]
- **a** – pospešek [mm/s²]

### Arhitektura

```
src/main.py
├── KinematicsTracker     – sledenje pozicije + d/v/a za eno točko
├── process_video()       – glavna zanka: zaznava → sledenje → zapis
└── CLI                   – argumenti za vhod/izhod/umerjanje
```

## Namestitev in zagon

### Z Dockerjem (priporočeno)

```bash
# Postavi video v data/
cp /pot/do/video.mp4 data/video.mp4

# Zgraditi Docker sliko
docker build -f docker/Dockerfile -t rv_izziv .

# Zaženi (brez GUI)
docker run --rm \
  -v $(pwd)/data:/app/data \
  rv_izziv

# Z umestitvenim faktorjem (primer: 5.2 px = 1 mm)
docker run --rm \
  -v $(pwd)/data:/app/data \
  rv_izziv python src/main.py --input data/video.mp4 --output data/results --px_per_mm 5.2 --no-show
```

### Lokalno (Python 3.10+)

```bash
pip install -r docker/requirements.txt
python src/main.py --input data/video.mp4 --output data/results --px_per_mm 5.2
# Za webcam:
python src/main.py --input 0
```

## Argumenti

| Argument | Privzeto | Opis |
|---|---|---|
| `--input` | `0` | Pot do videa ali indeks kamere |
| `--output` | `data/results` | Mapa za rezultate |
| `--px_per_mm` | `1.0` | Umeritveni faktor (px → mm) |
| `--no-show` | — | Ne prikazuj okna (za Docker) |

## Rezultati

Po obdelavi v `data/results/`:
- `annotated.mp4` – anotiran video z vizualizacijo
- `kinematics.json` – vsi kinematični podatki (wrist + thumb + index)
- `wrist_kinematics.csv` – zapestje v CSV formatu za Excel

## Umerjanje (px_per_mm)

Izmeri znano razdaljo na sliki (npr. razdalja med luknjami na plošči 9HPT je 32 mm). Preštej pike in izračunaj:

```
px_per_mm = razdalja_v_pikslih / razdalja_v_mm
```

## Podproblemi (razširitve)

- ✅ Kinematika ločeno za palec in kazalec (že implementirano)
- 🔲 Zaznavanje prijema/odlaganja zatiča (prag zapiranja med palcem in kazalcem)
- 🔲 Korelacija z EDSS/CogEval ocenami
- 🔲 Vrednotenje z referenčnimi meritvami (IMU)

## Mejniki

| Datum | Mejnik |
|---|---|
| 1.4.2026 | Metodologija |
| 8.4.2026 | Delovno okolje (Docker + Git) ✅ |
| 22.4.2026 | Vmesna rešitev |
| 20.5.2026 | Končna oddaja (do 8:00) |
| 3.6.2026 | Javna predstavitev |
