import os
import glob
import shutil
import gc
import sys

import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio

# (Opcional) se rodar headless e tiver erro de SDL:
# os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
# os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")

import gfootball.env as football_env

# =========================
# Configurações principais
# =========================
SCENARIO_NAME = "three_vs_three"

OUTPUT_DIR = f"frames_{SCENARIO_NAME}"                  # PNGs
VIDEO_LOGDIR = os.path.join(OUTPUT_DIR, "video_logs")    # engine video (se suportado)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_LOGDIR, exist_ok=True)

# Limpa frames e vídeos antigos (sobrescrever sempre)
for old_png in glob.glob(os.path.join(OUTPUT_DIR, "frame_*.png")):
    try:
        os.remove(old_png)
    except Exception:
        pass
for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
    for old_vid in glob.glob(os.path.join(VIDEO_LOGDIR, ext)):
        try:
            os.remove(old_vid)
        except Exception:
            pass

# Limpa artefatos antigos específicos deste run
for old_file in [
    os.path.join(OUTPUT_DIR, f"{SCENARIO_NAME}_fallback.mp4"),
]:
    try:
        if os.path.isdir(old_file):
            shutil.rmtree(old_file, ignore_errors=True)
        else:
            os.remove(old_file)
    except Exception:
        pass

# Parâmetros de execução
MAX_STEPS = 2000
CHANNEL_W, CHANNEL_H = 640, 480   # (largura, altura) -> obs vem (H, W, 3)

# Fallback MP4 (você já instalou o encoder)
MAKE_MP4_FROM_PNGS = True
FPS_FALLBACK = 30

print("=" * 80)
print(f"Iniciando execução - cenário: {SCENARIO_NAME}")
print(f"Saída de frames em: {OUTPUT_DIR}")
print(f"Log de vídeo (engine): {VIDEO_LOGDIR}")
print(f"Gerar MP4 de fallback pelos PNGs: {MAKE_MP4_FROM_PNGS} (FPS={FPS_FALLBACK})")
print("=" * 80)

if not os.environ.get("DISPLAY"):
    print("No DISPLAY defined, doing off-screen rendering")

# =========================
# Ambiente único (PIXELS) — força vídeo do motor se disponível
# =========================
def create_env_pixels():
    base_kwargs = dict(
        env_name=SCENARIO_NAME,
        representation="pixels",
        channel_dimensions=(CHANNEL_W, CHANNEL_H),
        render=True,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,
        write_video=True,
        logdir=VIDEO_LOGDIR,
    )
    try:
        # Algumas builds não suportam estes flags; tentamos e caímos no except se necessário
        return football_env.create_environment(
            **base_kwargs,
            dump_full_episodes=True,
            write_full_episode_dumps=True,
        )
    except TypeError:
        return football_env.create_environment(**base_kwargs)

env = create_env_pixels()
print("Ambiente criado com sucesso e render ativo.")

# =========================
# Utils — frames
# =========================
def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[2] > 3:
        frame = frame[:, :, :3]
    return frame

def extract_frame(obs):
    # Observação do GRF em pixels pode vir como ndarray, lista/tupla ou dict com "frame"
    if isinstance(obs, np.ndarray):
        return _to_uint8_rgb(obs)
    if isinstance(obs, (list, tuple)) and len(obs) > 0:
        return extract_frame(obs[0])
    if isinstance(obs, dict) and "frame" in obs:
        return _to_uint8_rgb(np.array(obs["frame"]))
    # Se chegar aqui é porque o formato não é esperado; levantamos um erro explícito
    raise TypeError(f"Formato de observação inesperado para pixels: {type(obs)}; exemplo={str(obs)[:200]}")

# =========================
# Execução principal (apenas frames/vídeo, sem métricas)
# =========================
obs = env.reset()
try:
    print(f"Verificando action_space: {env.action_space}")
except Exception:
    pass

print("Ambiente resetado. Coleta de frames iniciada.")

# primeiro frame
try:
    frame0 = extract_frame(obs)
    imageio.imwrite(os.path.join(OUTPUT_DIR, f"frame_{0:04d}.png"), frame0)
    print(f"🔎 Frame inicial salvo. shape={frame0.shape}, dtype={frame0.dtype}")
except Exception as e:
    print("❌ Falha ao salvar frame inicial:", repr(e))
    env.close()
    sys.exit(1)

done = False
step = 1  # já salvamos o frame 0

with tqdm(total=MAX_STEPS - 1, desc="Frames (1 env)", ncols=92) as pbar:
    while not done and step < MAX_STEPS:
        # ação aleatória apenas para gerar movimentação (substitua pelo seu agente quando quiser)
        action = env.action_space.sample()
        obs, reward, done, info = env.step([action])

        try:
            frame = extract_frame(obs)
            image_path = os.path.join(OUTPUT_DIR, f"frame_{step:04d}.png")
            imageio.imwrite(image_path, frame)
        except Exception as e:
            print(f"⚠️ Falha ao salvar frame {step}: {repr(e)}")

        if step % 100 == 0:
            print(f"Progresso: {step}/{MAX_STEPS} steps")

        step += 1
        pbar.update(1)

# fecha e força flush do vídeo do motor
env.close()
del env
gc.collect()

print(f"🏁 Episódio terminou no step {step}")
print(f"✅ PNGs salvos em: {OUTPUT_DIR}/")

# =========================
# Checagem do vídeo do motor (substitui antigos pois limpamos antes)
# =========================
def find_engine_videos(root):
    vids = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        vids.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return vids

engine_videos = find_engine_videos(VIDEO_LOGDIR)
if engine_videos:
    print("🎬 Vídeo(s) do motor encontrado(s):")
    for v in engine_videos:
        print(" -", v)
else:
    print("⚠️ Nenhum vídeo do motor encontrado em", VIDEO_LOGDIR)

# =========================
# Fallback MP4 a partir dos PNGs
# =========================
if MAKE_MP4_FROM_PNGS:
    try:
        mp4_path = os.path.join(OUTPUT_DIR, f"{SCENARIO_NAME}_fallback.mp4")
        pngs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "frame_*.png")))
        if pngs:
            print(f"🎯 Gerando vídeo de fallback a partir dos PNGs: {mp4_path}")
            with imageio.get_writer(mp4_path, mode="I", fps=FPS_FALLBACK, codec="libx264") as writer:
                for p in pngs:
                    writer.append_data(imageio.imread(p))
            print("✅ Vídeo de fallback salvo em:", mp4_path)
        else:
            print("⚠️ Sem PNGs para montar vídeo de fallback.")
    except Exception as e:
        print("❌ Falha ao gerar MP4 de fallback:", repr(e))
        print("💡 Verifique se o encoder está disponível:  pip install 'imageio[ffmpeg]'  ou apt-get install -y ffmpeg")

print("📦 Finalizado: frames + (opcional) vídeo gerado(s).")
