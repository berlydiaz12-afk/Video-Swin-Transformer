# vswin: Video Swin Transformer para Colab

Este paquete carga los checkpoints oficiales de
[Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) con PyTorch 2.x,
sin mmcv ni mmaction, y los afina con los miniclips del paquete de preprocesamiento (`preproc`).

El repositorio oficial exige `mmcv` entre 1.3.1 y 1.4.0. Esa versión necesita PyTorch 1.x y no se puede
instalar en Colab. El modelo en sí solo depende de `torch`, `timm` y `einops`. Por eso
`vswin/swin3d.py` es una copia del backbone oficial (licencia Apache-2.0, ver `LICENSE_VIDEO_SWIN`)
sin esas dependencias. Los nombres de los parámetros no cambian, así que los `.pth` oficiales cargan
directamente.

## Instalación en Colab

1. Sube esta carpeta a Drive.
2. Elige **Entorno de ejecución → Cambiar tipo → GPU**.
3. Ejecuta:

```python
from google.colab import drive
drive.mount('/content/drive')
!pip install -q /content/drive/MyDrive/Video-Swin-Transformer-Colab
```

Colab ya trae `torch` y `ffmpeg`. Para el resto de los pasos, abre `colab_quickstart.ipynb`.

## Módulos

| Módulo | Contenido |
|---|---|
| `variants.py` | Catálogo de checkpoints oficiales (T/S/B, K400, K600, SSv2) con hiperparámetros y URL |
| `checkpoints.py` | Descarga con caché (`MyDrive/vswin_cache`), lectura de `.pth` y carga con informe de claves |
| `model.py` | `VideoSwin` = `SwinTransformer3D` + `I3DHead`; `build_model(...)` |
| `swin3d.py` | Backbone oficial sin mmcv |
| `swin_input.py` | Formato de entrada: `to_model_input`, `prepare_clips`, `normalize`, `to_cthw`, `check_input` |
| `miniclip_reading.py` | Lectura de los `.mp4` de `miniclips/<id>/` con ffmpeg a RGB uint8 |
| `dataset.py` | `index_dataset`, `split_records`, `MiniclipDataset` (modos clip y video), `make_loaders` |
| `inference.py` | `predict_clips`, `predict_video`, `extract_features` |
| `fine_tuning.py` | `FineTuneConfig`, `FineTuner` (AMP, acumulación, warmup + coseno, reanudación), `evaluate`, métricas |
| `feature_cache.py` | Solo cabeza: extrae y guarda los embeddings (original y volteado), `HeadTrainer` y `evaluate_features` |
| `colab_setup.py` | Montaje de Drive, información de la GPU, tipo de AMP y carpeta de caché |

## Cargar el modelo

```python
import vswin

# Kinetics-400 completo (400 clases)
model = vswin.build_model('swin_b_k400_22k', device='cuda')

# Tesis: backbone preentrenado + cabeza nueva de 2 clases
model = vswin.build_model('swin_b_k400_22k', num_classes=2, use_checkpoint=True, device='cuda')

# Cabeza binaria: 1 salida + sigmoide (BCEWithLogitsLoss) en lugar de 2 salidas + softmax
model = vswin.build_model('swin_b_k400_22k', num_classes=2, binary=True, device='cuda')

# Modelo ya afinado (binary debe coincidir con el usado al entrenar)
model = vswin.build_model('swin_b_k400_22k', num_classes=2, pretrained='.../best.pth', device='cuda')
```

Con `binary=True`, `FineTuner` usa `BCEWithLogitsLoss` (con `pos_weight` opcional), y
`logits_to_probs` convierte σ(z) en `[1 − p, p]`. Así las métricas y las predicciones son las mismas que
con softmax.

`build_model` imprime cuántos tensores cargó, si la cabeza se dejó sin cargar y si el backbone quedó
completo. Si falta alguna capa del backbone, lanza un error.

## Entrada del modelo

| | Valor |
|---|---|
| Forma | `(B, C, T, H, W) = (B, 3, 32, 224, 224)` |
| Tipo | `float32` (AMP: bf16 en A100/L4, fp16 en T4) |
| Canales | RGB |
| Normalización | `(x/255 − (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)` |
| Muestreo en el preentrenamiento | 32 cuadros, 1 de cada 2 (igual que `stride: 2` en `preproc`) |

```python
clips = vswin.read_miniclips('salida/pornography2k/miniclips/<id>')  # (K, 32, 224, 224, 3) uint8
x = vswin.prepare_clips(clips, device='cuda')                         # (K, 3, 32, 224, 224) float32
```

Así cambian las dimensiones dentro de Swin-B: (B, 3, 32, 224, 224) → patch 2×4×4 →
(B, 128, 16, 56, 56) → etapas 1 a 4 (espacio 56→28→14→7) → (B, 1024, 16, 7, 7) →
promedio → (B, 1024) → lineal → (B, num_classes).

Los Datasets entregan uint8. La conversión a float y la normalización se hacen en la GPU con
`to_model_input`, porque así se transfieren 4 veces menos bytes.

## Ajuste fino

```python
records = vswin.index_dataset('/content/data/pornography2k',
                              labels=vswin.label_from_path({'NonPorn': 0, 'Porn': 1}))
train_rec, val_rec = vswin.split_records(records, val_fraction=0.2)
train_loader, val_loader = vswin.make_loaders(train_rec, val_rec, batch_size=2)

cfg = vswin.FineTuneConfig(output_dir='/content/drive/MyDrive/vswin_runs/run1', epochs=30)
vswin.FineTuner(model, train_loader, val_loader, cfg).fit()
```

- **Partición fija para varios entrenamientos.** `get_or_create_split(records, split_dir)` divide la primera
  vez en train, val y test (por video y estratificado) y guarda `train.txt`, `val.txt`, `test.txt` y
  `split_info.json`. Las corridas siguientes solo leen esos archivos (`load_split`). Si falta un video
  de la lista, se detiene con un error. `make_eval_loader(test_rec)` crea el loader de test.
- **Entrenamiento por miniclip.** El muestreador da el mismo peso a cada video, sin importar su K, y a
  cada clase.
- **Evaluación por video.** La probabilidad de un video es el promedio de las de sus miniclips (como
  `average_clips='prob'` en mmaction). Se informan métricas por miniclip (`clip_*`) y por video
  (`video_*`): accuracy, balanced accuracy y, con 2 clases, precision, recall, F1 y AUC.
- **Archivos en `output_dir`:** `last.pth` (cada época, para reanudar), `best.pth` (mejor `monitor`),
  `history.json`, `predictions_best.json` y `config.json`.
- **Reanudación.** Si Colab se desconecta, crea otra vez el modelo, los loaders y el `FineTuner` con el
  mismo `output_dir`, y el entrenamiento continúa desde `last.pth`.

### Memoria (Swin-B, 32×224×224)

| GPU | Recomendación |
|---|---|
| T4 16 GB | `batch_size=2`, `use_checkpoint=True`, `accumulation_steps=8` |
| L4 24 GB | `batch_size=4`, `use_checkpoint=True` |
| A100 40 GB | `batch_size=8` (sin checkpointing si cabe) |

`use_checkpoint=True` recalcula activaciones en el paso hacia atrás: usa mucha menos memoria y tarda
alrededor de un 30 % más. Si solo hay poca GPU disponible, `freeze_backbone_epochs` entrena solo la
cabeza durante las primeras épocas.

### Diferencias con la receta oficial

- Las tasas de aprendizaje oficiales suponen 8 GPU × 8 videos × acumulación de 8. Con un lote efectivo
  mucho menor, conviene probar `lr` entre 1e-4 y 3e-4.
- En mmcv solo se aplica la primera clave de `paramwise_cfg` que coincide. Por eso, en las corridas
  oficiales, las normas del backbone conservaron weight decay. Aquí ningún parámetro 1-D ni las tablas
  de posición llevan weight decay, y todo el backbone usa `lr × 0.1`.
- El pipeline oficial hace `RandomResizedCrop` sobre el video original. Aquí los miniclips ya están en
  224×224, así que solo se aplica volteo horizontal.
