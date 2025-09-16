#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from tqdm import tqdm

import numpy as np

# Import correto: o create_environment está em gfootball.env
import gfootball.env as grf

# ===== Configurações =====
# Troque para "three_vs_three_with_keeper" se você já colocou o cenário com goleiro fixo.
SCENARIO_NAME = "three_vs_three"
FRAMES_DIR = Path(f"frames_{SCENARIO_NAME}")
ENGINE_LOG_DIR = FRAMES_DIR / "video_logs"
MAKE_FALLBACK_MP4 = True
FALLBACK_FPS = 30
MAX_STEPS = 2000
RENDER = True  # off-screen ok

def create_environment():
    env = grf.create_environment(
        env_name=SCENARIO_NAME,
        representation='pixels',
        render=RENDER,
        logdir=str(ENGINE_LOG_DIR),
        write_video=False,  # geramos PNGs + MP4 de fallback
        number_of_left_players_agent_controls=3,   # apenas linha; GK, se existir, fica não-controlável
        number_of_right_players_agent_controls=3,
        channel_dimensions=(480, 640)  # (height, width)
    )
    return env

def save_frame_png(frame_idx, frame):
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio
    png_path = FRAMES_DIR / f"frame_{frame_idx:04d}.png"
    # frame esperado como HxWxC (uint8)
    if not isinstance(frame, np.ndarray):
        frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    imageio.imwrite(png_path, frame)
    return png_path

def fallback_pngs_to_mp4():
    if not MAKE_FALLBACK_MP4:
        return None
    try:
        try:
            import imageio.v2 as imageio
        except Exception:
            import imageio
        pngs = sorted(FRAMES_DIR.glob("frame_*.png"))
        if not pngs:
            print("Sem PNGs para compor MP4 de fallback.")
            return None
        mp4_path = FRAMES_DIR / f"{SCENARIO_NAME}_fallback.mp4"
        with imageio.get_writer(mp4_path, fps=FALLBACK_FPS, codec='libx264') as w:
            for p in tqdm(pngs, desc="Compondo MP4 (fallback)"):
                w.append_data(imageio.imread(p))
        print(f"MP4 de fallback gerado: {mp4_path}")
        return mp4_path
    except Exception as e:
        print("Falha ao compor MP4 de fallback: ", repr(e))
        return None

def main():
    print("="*80)
    print(f"Iniciando execução - cenário: {SCENARIO_NAME}")
    print(f"Saída de frames em: {FRAMES_DIR}")
    print(f"Log de vídeo (engine): {ENGINE_LOG_DIR}")
    print(f"Gerar MP4 de fallback pelos PNGs: {MAKE_FALLBACK_MP4} (FPS={FALLBACK_FPS})")
    print("="*80)

    if not os.environ.get("DISPLAY"):
        print("No DISPLAY defined, doing off-screen rendering")

    env = create_environment()
    print("Ambiente criado com sucesso e render ativo.")

    print("Ambiente resetado. Coleta de frames iniciada (sem métricas pesadas).")
    obs = env.reset()

    # Observação: no GRF, quando representation='pixels', obs costuma ser np.ndarray (HxWxC).
    # Em alguns wrappers pode vir como lista/tupla; tratamos ambos.
    frame0 = obs[0] if isinstance(obs, (list, tuple)) else obs
    save_frame_png(0, frame0)

    pbar = tqdm(range(1, MAX_STEPS), desc="Steps")
    done = False

    for step_i in pbar:
        # Amostra ação (o espaço pode ser composto; o sample() já retorna no formato correto)
        try:
            action = env.action_space.sample()
        except Exception:
            action = 0

        obs, reward, done, info = env.step(action)

        frame = obs[0] if isinstance(obs, (list, tuple)) else obs
        save_frame_png(step_i, frame)

        if step_i % 50 == 0:
            pbar.set_postfix_str(f"{step_i}/{MAX_STEPS} frames")

        if done:
            break

    env.close()
    print("Execução encerrada, compondo vídeo de fallback (se habilitado)...")
    fallback_pngs_to_mp4()
    print("Finalizado.")

if __name__ == "__main__":
    main()
