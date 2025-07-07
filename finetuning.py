import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import cv2
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# === Configuración adaptativa (CUDA/CPU) ===
BASE_DIR = "./training"
CLIP_DURATION = 2.56  # segundos
STRIDE = 1.0  # segundos
EPOCHS = 10
LEARNING_RATE = 1e-4
OUTPUT_MODEL = "i3d_finetuned_violencia_robo_normal.pth"
PRETRAINED_MODEL = "I3D_8x8_R50_pytorchvideo.pyth"  # Formato PyTorchVideo
LABELS = ["Normal", "Robo", "Violencia"]
TARGET_SIZE = (224, 224)
NUM_FRAMES = 64

# Configuración que se ajusta automáticamente según el hardware disponible
def get_hardware_config():
    if torch.cuda.is_available():
        return {
            'batch_size': 2,  # RTX 3060 puede manejar batch size 2
            'num_workers': 2,
            'pin_memory': True,
            'use_mixed_precision': True,
            'cuda_device': 0
        }
    else:
        return {
            'batch_size': 1,  # CPU usa batch size más pequeño
            'num_workers': 0,
            'pin_memory': False,
            'use_mixed_precision': False,
            'cuda_device': None
        }

# === Dataset con sliding window ===
class SlidingWindowVideoDataset(Dataset):
    def __init__(self, video_paths, clip_duration, stride, num_frames=64):
        self.clips = []
        self.clip_duration = clip_duration
        self.stride = stride
        self.num_frames = num_frames
        
        # Generar todos los clips con sliding window
        for video_path, label in video_paths:
            clips_from_video = self._generate_clips(video_path, label)
            self.clips.extend(clips_from_video)
        
        print(f"📊 Total de clips generados: {len(self.clips)}")
    
    def _generate_clips(self, video_path, label):
        """Genera clips usando sliding window para un video"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if fps <= 0:
                return []
            
            video_duration = total_frames / fps
            clips = []
            
            start_time = 0.0
            while start_time + self.clip_duration <= video_duration:
                start_frame = int(start_time * fps)
                end_frame = int((start_time + self.clip_duration) * fps)
                
                clips.append({
                    'video_path': video_path,
                    'label': label,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'fps': fps
                })
                
                start_time += self.stride
            
            return clips
            
        except Exception as e:
            print(f"❌ Error procesando {video_path}: {e}")
            return []
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        
        # Cargar frames del clip
        frames = self._load_video_clip(
            clip_info['video_path'],
            clip_info['start_frame'],
            clip_info['end_frame']
        )
        
        # Convertir a tensor y aplicar transformaciones
        video_tensor = self._preprocess_frames(frames)
        
        return {
            'video': video_tensor,
            'label': torch.tensor(clip_info['label'], dtype=torch.long)
        }
    
    def _load_video_clip(self, video_path, start_frame, end_frame):
        """Carga frames específicos de un video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Calcular índices de frames a extraer
        total_clip_frames = end_frame - start_frame
        if total_clip_frames <= 0:
            cap.release()
            return [np.zeros((*TARGET_SIZE, 3), dtype=np.uint8)] * self.num_frames
        
        # Submuestreo uniforme para obtener NUM_FRAMES
        frame_indices = np.linspace(start_frame, end_frame-1, self.num_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, TARGET_SIZE)
                frames.append(frame)
            else:
                # Usar último frame válido si hay error
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((*TARGET_SIZE, 3), dtype=np.uint8))
        
        cap.release()
        
        # Asegurar que tenemos exactamente NUM_FRAMES
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((*TARGET_SIZE, 3), dtype=np.uint8))
        
        return frames[:self.num_frames]
    
    def _preprocess_frames(self, frames):
        """Convierte frames a tensor y aplica transformaciones"""
        # Convertir a tensor [T, H, W, C]
        video_tensor = torch.from_numpy(np.array(frames)).float()
        
        # Permutar a [C, T, H, W]
        video_tensor = video_tensor.permute(3, 0, 1, 2)
        
        # Normalizar a [0,1]
        video_tensor = video_tensor / 255.0
        
        # Normalización estándar
        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
        video_tensor = (video_tensor - mean) / std
        
        return video_tensor

# === Funciones de utilidad ===
def recolectar_videos(base_dir):
    """Recolecta todas las rutas de videos con sus etiquetas"""
    video_paths = []
    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    
    for label in LABELS:
        label_dir = Path(base_dir) / label
        if not label_dir.exists():
            print(f"⚠️ Directorio no encontrado: {label_dir}")
            continue
        
        for video_file in label_dir.glob("*.mp4"):
            video_paths.append((str(video_file), label_to_idx[label]))
    
    return video_paths

def contar_videos_por_etiqueta(video_paths):
    """Cuenta videos por cada etiqueta"""
    conteo = defaultdict(int)
    for _, label_idx in video_paths:
        label_name = LABELS[label_idx]
        conteo[label_name] += 1
    
    print("\n📊 [Resumen de videos por clase]")
    for label, count in conteo.items():
        print(f"   {label}: {count} videos")
    print()

def estimar_clips_totales(video_paths):
    """Estima el número total de clips que se generarán"""
    total_clips = 0
    clips_por_clase = defaultdict(int)
    
    for video_path, label_idx in video_paths:
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if fps > 0:
                video_duration = total_frames / fps
                num_clips = max(0, int((video_duration - CLIP_DURATION) / STRIDE) + 1)
                total_clips += num_clips
                clips_por_clase[LABELS[label_idx]] += num_clips
                
        except Exception as e:
            print(f"❌ Error estimando clips para {video_path}: {e}")
    
    print("📊 [Estimación de clips por sliding window]")
    for label, count in clips_por_clase.items():
        print(f"   {label}: {count} clips")
    print(f"   Total estimado: {total_clips} clips\n")

# === Modelo I3D ===
def crear_modelo_i3d():
    """Crea modelo I3D-ResNet50 usando PyTorchVideo Hub"""
    try:
        from pytorchvideo.models.hub import i3d_r50
        
        # Crear modelo I3D desde PyTorchVideo Hub
        print("🔄 Creando modelo I3D-ResNet50 desde PyTorchVideo Hub...")
        model = i3d_r50(pretrained=False)  # Sin pesos preentrenados por ahora
        
        print("✅ Modelo I3D-ResNet50 creado exitosamente")
        return model
        
    except ImportError:
        print("❌ PyTorchVideo no encontrado. Instalando...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorchvideo"])
        print("✅ PyTorchVideo instalado. Reintentando...")
        return crear_modelo_i3d()
        
    except Exception as e:
        print(f"❌ Error creando modelo I3D: {e}")
        print("🔄 Intentando método alternativo...")
        try:
            # Fallback al método anterior
            from pytorchvideo.models.resnet import create_resnet
            model = create_resnet(
                input_channel=3,
                model_depth=50,
                model_num_class=400
            )
            print("✅ Modelo creado con método alternativo")
            return model
        except Exception as e2:
            print(f"❌ Error con método alternativo: {e2}")
            exit(1)


def descargar_modelo_preentrenado():
    """Descarga el modelo I3D preentrenado si no existe"""
    modelo_url = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/I3D_8x8_R50.pyth"
    modelo_local = PRETRAINED_MODEL
    
    if not os.path.exists(modelo_local):
        print(f"📥 Descargando modelo preentrenado I3D...")
        print(f"🔗 URL: {modelo_url}")
        try:
            import urllib.request
            urllib.request.urlretrieve(modelo_url, modelo_local)
            print(f"✅ Modelo descargado: {modelo_local}")
            return True
        except Exception as e:
            print(f"❌ Error descargando modelo: {e}")
            print(f"💡 Puedes descargarlo manualmente desde: {modelo_url}")
            return False
    else:
        print(f"✅ Modelo preentrenado encontrado: {modelo_local}")
        return True

def cargar_pesos_preentrenados(model, pretrained_path):
    """Carga pesos preentrenados desde archivos .pth o .pyth"""
    if os.path.exists(pretrained_path):
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Detectar formato del checkpoint
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                # Formato PyTorchVideo (.pyth)
                state_dict = checkpoint['model_state']
                print(f"🔍 Detectado formato PyTorchVideo (.pyth)")
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # Formato estándar con state_dict
                state_dict = checkpoint['state_dict']
                print(f"🔍 Detectado formato con state_dict")
            else:
                # Formato directo (solo state_dict)
                state_dict = checkpoint
                print(f"🔍 Detectado formato directo")
            
            # Intentar cargar con strict=False para ignorar incompatibilidades
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            # Contar cuántas capas se cargaron exitosamente
            total_params = len(model.state_dict())
            loaded_params = total_params - len(missing_keys)
            
            if len(missing_keys) > total_params * 0.5:  # Si más del 50% no coincide
                print(f"⚠️ Arquitectura incompatible: {len(missing_keys)}/{total_params} capas no coinciden")
                print(f"💡 Entrenando desde cero (recomendado para esta arquitectura)")
                return False
            else:
                print(f"✅ Pesos preentrenados cargados: {loaded_params}/{total_params} capas")
                if missing_keys:
                    print(f"⚠️ {len(missing_keys)} capas se inicializarán aleatoriamente")
                return True
                
        except Exception as e:
            print(f"⚠️ Error cargando pesos: {e}")
            print(f"💡 Continuando entrenamiento desde cero")
            return False
    else:
        print(f"⚠️ Archivo {pretrained_path} no encontrado")
        print(f"💡 Entrenando desde cero")
        return False

def modificar_capa_final(model, num_clases):
    """Modifica la capa final del modelo para el número de clases objetivo"""
    # Buscar y reemplazar la capa final
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == 400:
            # Encontramos la capa de 400 clases
            parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else None
            child_name = name.split('.')[-1]
            
            new_layer = nn.Linear(module.in_features, num_clases)
            
            if parent_name:
                parent_module = dict(model.named_modules())[parent_name]
                setattr(parent_module, child_name, new_layer)
            else:
                setattr(model, child_name, new_layer)
            
            print(f"✅ Capa final '{name}' modificada: {module.in_features} -> {num_clases}")
            return True
    
    # Fallback para modelos de torchvision
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_clases)
        print(f"✅ Capa 'fc' modificada para {num_clases} clases")
        return True
    
    print("❌ No se pudo encontrar la capa final para modificar")
    return False

# === Entrenamiento adaptativo (CUDA/CPU) ===
def entrenar_modelo(model, dataloader, device, num_epochs, config):
    """Función principal de entrenamiento adaptativa para CUDA/CPU"""
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Configurar mixed precision si está disponible
    scaler = None
    if config['use_mixed_precision'] and device.type == 'cuda':
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        print("✅ Mixed Precision habilitado (AMP)")
    
    # Optimizaciones CUDA
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # Optimizar para tamaños fijos
        torch.backends.cudnn.deterministic = False  # Mejor rendimiento
        print(f"🚀 Optimizaciones CUDA habilitadas")
        print(f"📊 VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"\n🚀 [Entrenamiento iniciado en {device}]")
    print(f"📋 Configuración: {num_epochs} épocas, batch_size={config['batch_size']}, lr={LEARNING_RATE}")
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        epoch_start_time = time.time()
        
        print(f"\n📅 --- Época {epoch+1}/{num_epochs} ---")
        
        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()
            
            # Mover datos al device (optimizado)
            videos = batch['video'].to(device, non_blocking=config['pin_memory'])
            labels = batch['label'].to(device, non_blocking=config['pin_memory'])
            
            optimizer.zero_grad()
            
            # Forward pass con mixed precision si está disponible
            if scaler is not None:
                with autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                
                # Backward pass con scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass normal
                outputs = model(videos)
                loss = criterion(outputs, labels)
                
                # Backward pass normal
                loss.backward()
                optimizer.step()
            
            # Estadísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Calcular ETA
            batch_time = time.time() - batch_start_time
            remaining_batches = len(dataloader) - batch_idx - 1
            eta_seconds = batch_time * remaining_batches
            
            # Log del lote con info de GPU
            accuracy = 100 * correct_predictions / total_samples
            gpu_memory = torch.cuda.memory_allocated(0) / 1e9 if device.type == 'cuda' else 0
            
            if device.type == 'cuda':
                print(f"   [{batch_idx+1:4d}/{len(dataloader):4d}] "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {accuracy:.2f}% | "
                      f"GPU: {gpu_memory:.1f}GB | "
                      f"ETA: {int(eta_seconds):3d}s")
            else:
                print(f"   [{batch_idx+1:4d}/{len(dataloader):4d}] "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {accuracy:.2f}% | "
                      f"ETA: {int(eta_seconds):3d}s")
            
            # Liberar memoria cache cada cierto número de batches
            if batch_idx % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Resumen de la época
        epoch_time = time.time() - epoch_start_time
        avg_loss = running_loss / len(dataloader)
        epoch_accuracy = 100 * correct_predictions / total_samples
        
        print(f"   ✅ Época {epoch+1} completada en {int(epoch_time)}s")
        print(f"      Loss promedio: {avg_loss:.4f}")
        print(f"      Accuracy final: {epoch_accuracy:.2f}%")
        
        if device.type == 'cuda':
            print(f"      VRAM máxima usada: {torch.cuda.max_memory_allocated(0) / 1e9:.1f} GB")
            torch.cuda.reset_peak_memory_stats()
    
    # Guardar modelo
    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"\n💾 Modelo guardado como '{OUTPUT_MODEL}'")

# === MAIN adaptativo (CUDA/CPU) ===
def main():
    print("🎬 Iniciando entrenamiento I3D con Sliding Window")
    print("=" * 50)
    
    # Obtener configuración según hardware disponible
    config = get_hardware_config()
    
    # Verificar CUDA y configurar device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{config["cuda_device"]}')
        gpu_name = torch.cuda.get_device_name(config["cuda_device"])
        gpu_memory = torch.cuda.get_device_properties(config["cuda_device"]).total_memory / 1e9
        print(f"🚀 GPU detectada: {gpu_name}")
        print(f"💾 VRAM total: {gpu_memory:.1f} GB")
        print(f"🖥️  Device: {device}")
        
        # Limpiar cache de GPU
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print(f"⚠️  CUDA no disponible, usando CPU")
        print(f"🖥️  Device: {device}")
        print(f"💡 Para usar GPU, ejecuta: setup_cuda.bat")
    
    # Recolectar videos
    print(f"\n📂 Buscando videos en: {BASE_DIR}")
    video_paths = recolectar_videos(BASE_DIR)
    
    if not video_paths:
        print("❌ No se encontraron videos. Verifica la estructura de carpetas.")
        return
    
    print(f"✅ Total de videos encontrados: {len(video_paths)}")
    
    # Mostrar estadísticas
    contar_videos_por_etiqueta(video_paths)
    estimar_clips_totales(video_paths)
    
    # Crear dataset con sliding window
    print("🔄 Creando dataset con sliding window...")
    dataset = SlidingWindowVideoDataset(
        video_paths=video_paths,
        clip_duration=CLIP_DURATION,
        stride=STRIDE,
        num_frames=NUM_FRAMES
    )
    
    # Crear dataloader con configuración adaptativa
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=config['num_workers'] > 0
    )
    
    print(f"✅ DataLoader creado: {len(dataset)} clips, {len(dataloader)} batches")
    if device.type == 'cuda':
        print(f"🔧 Optimizaciones: pin_memory={config['pin_memory']}, num_workers={config['num_workers']}")
    
    # Crear modelo
    print("\n🧠 Configurando modelo I3D...")
    model = crear_modelo_i3d()
    
    # Descargar modelo preentrenado si no existe
    print("\n📦 Verificando modelo preentrenado...")
    descargar_modelo_preentrenado()
    
    # Cargar pesos preentrenados
    cargar_pesos_preentrenados(model, PRETRAINED_MODEL)
    
    # Modificar capa final
    modificar_capa_final(model, len(LABELS))
    
    # Mostrar información del modelo
    model_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parámetros del modelo: {model_params:,}")
    
    # Entrenar
    entrenar_modelo(model, dataloader, device, EPOCHS, config)
    
    print("\n🎉 ¡Entrenamiento completado!")
    if device.type == 'cuda':
        print(f"🏁 VRAM final usada: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")

if __name__ == "__main__":
    main()
