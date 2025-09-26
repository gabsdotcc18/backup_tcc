# treino_ppo_3x3_gymnasium.py
import os
import glob
import time
from typing import Callable
import numpy as np

import gymnasium as gym
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
import gfootball.env as football_env

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecMonitor,
    VecVideoRecorder,
    VecNormalize,
)
from stable_baselines3.common.callbacks import BaseCallback

# --- helper para achar VecNormalize em qualquer pilha de wrappers (SB3 2.x) ---
def _find_vecnormalize(env):
    from stable_baselines3.common.vec_env import VecNormalize
    e = env
    visited = set()
    while e is not None and id(e) not in visited:
        visited.add(id(e))
        if isinstance(e, VecNormalize):
            return e
        # desce pelos encadeamentos comuns
        e = getattr(e, "venv", None) or getattr(e, "env", None)
    return None



# =============================================================================
# Configurações principais
# =============================================================================
SCENARIO_NAME = "3x3"  # seu cenário 3x3 + GK na defesa; ataque sem GK
FRAME_DIR = "frames_three_vs_three"
VIDEO_DIR = os.path.join(FRAME_DIR, "video_logs")
CHECKPOINT_DIR = "checkpoints_ppo"
BEST_DIR = os.path.join(CHECKPOINT_DIR, "best_model")
BEST_MODEL_FILE = os.path.join(BEST_DIR, "best_model.zip")
NORM_FILE = os.path.join(CHECKPOINT_DIR, "vecnormalize.pkl")  # stats do VecNormalize

os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)


# =============================================================================
# Funções de criação de ambiente
# =============================================================================
def make_single_env(seed: int) -> Callable[[], gym.Env]:
    """
    Cria função-fábrica de um env GRF (API Gym v21) e envolve com shimmy -> Gymnasium.
    """
    def _init():
        # 1) Env GRF nativo (Gym v21)
        old_env = football_env.create_environment(
            env_name=SCENARIO_NAME,
            representation="pixels",         # vizualização/validação melhor
            render=True,  # <<< NECESSÁRIO para 'frame' no pixels                  # render off-screen; salvaremos vídeos
            channel_dimensions=(64, 64),     # downscale para acelerar PPO em pixels
            number_of_left_players_agent_controls=1,  # 1 atacante treinando
            number_of_right_players_agent_controls=0, # defesa + GK heurístico
        )
        # 2) Convertido para Gymnasium
        env = GymV21CompatibilityV0(env=old_env)
        # 3) Seed
        env.reset(seed=seed)
        return env
    return _init


def build_vec_env(n_cpu: int = 4, base_seed: int = 1234) -> gym.Env:
    """
    Constrói o VecEnv (Subproc) + VecMonitor.
    O VecNormalize será aplicado depois (para permitir load dos stats).
    """
    set_random_seed(base_seed)
    env_fns = [make_single_env(base_seed + i) for i in range(n_cpu)]
    venv = DummyVecEnv(env_fns) if n_cpu == 1 else SubprocVecEnv(env_fns)
    venv = VecMonitor(venv)
    return venv


# =============================================================================
# Callback de checkpoints + avaliação + save de VecNormalize
# =============================================================================
class CheckpointAndEvalCallback(BaseCallback):
    def __init__(
        self,
        save_freq_steps: int,
        ckpt_dir: str,
        best_file: str,
        norm_file: str,
        n_eval_episodes: int = 5,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq_steps = save_freq_steps
        self.ckpt_dir = ckpt_dir
        self.best_file = best_file
        self.norm_file = norm_file
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # salva/avalia a cada save_freq_steps
        if self.num_timesteps % self.save_freq_steps != 0:
            return True

        # 1) Salvar checkpoint incremental
        ckpt_path = os.path.join(self.ckpt_dir, f"ppo_{self.num_timesteps}_steps.zip")
        self.model.save(ckpt_path)
        if self.verbose:
            print(f"[checkpoint] Modelo salvo em: {ckpt_path}")

        # 2) Salvar stats do VecNormalize (se existir)
        vec_norm = _find_vecnormalize(self.model.get_env())
        try:
            if vec_norm is not None:
                vec_norm.save(self.norm_file)
                if self.verbose:
                    print(f"[normalize] Stats VecNormalize salvas em: {self.norm_file}")
        except Exception as e:
            print(f"[normalize] Aviso: falha ao salvar VecNormalize: {e}")

        # 3) Avaliação rápida (usa o próprio env vetorizado; suficiente p/ ranking)
        mean_reward = self._quick_eval_mean_reward(episodes=self.n_eval_episodes)
        if self.verbose:
            print(f"[eval] mean_reward (n={self.n_eval_episodes}) = {float(mean_reward):.3f}")

        # 4) Atualizar best_model se melhorar
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = float(mean_reward)
            self.model.save(self.best_file)
            if self.verbose:
                print(f"[best] Novo melhor modelo salvo em: {self.best_file} (reward={self.best_mean_reward:.3f})")

        return True

    def _quick_eval_mean_reward(self, episodes: int = 5) -> float:
        """
        Rollout curto determinístico no próprio env de treino (ok para ranking rápido).
        Em VecEnv, done vem agregando terminated|truncated pela camada vec.
        """
        env = self.model.get_env()
        ep_rewards = []
        for _ in range(episodes):
            obs = env.reset()
            # reset() em VecEnv retorna obs; infos tratados internamente
            done = [False] * env.num_envs
            ep_r = np.zeros(env.num_envs, dtype=np.float32)
            while not all(done):
                actions, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(actions)
                ep_r += rewards
                # marca finalizados
                done = np.logical_or(done, dones)
            ep_rewards.extend(ep_r.tolist())
        return float(np.mean(ep_rewards)) if ep_rewards else 0.0


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Iniciando treinamento - cenário: three_vs_three (1 atacante treinando com PPO)")
    print(f"Saída de frames em: {FRAME_DIR}")
    print(f"Log de vídeo (engine): {VIDEO_DIR}")
    print("Gerar MP4 de fallback via imageio/ffmpeg: HABILITADO (VecVideoRecorder)")
    print("=" * 80)

    # 1) Construir VecEnv base
    N_CPU = int(os.environ.get("PPO_WORKERS", "4"))
    BASE_SEED = 42
    base_env = build_vec_env(n_cpu=N_CPU, base_seed=BASE_SEED)

    # 2) Aplicar VecNormalize — carregar stats se já existirem (resume)
    if os.path.isfile(NORM_FILE):
        print(f"[resume] Carregando stats VecNormalize de: {NORM_FILE}")
        env = VecNormalize.load(NORM_FILE, base_env)
        env.norm_obs = False
        env.norm_reward = True

    else:
        env = VecNormalize(base_env, norm_obs=False, norm_reward=True, clip_obs=10.0)
    # Importante: modo treino (True) para atualizar stats durante o learning
    env.training = True

    # 3) Gravação de vídeo periódica (para validação visual)
   # env = VecVideoRecorder(
  #      env,
#        video_folder=VIDEO_DIR,
#        record_video_trigger=lambda step: step % 10_000 == 0,  # a cada 10k steps
#        video_length=800,  # ~duração de cada vídeo
#        name_prefix="ppo-football",
#    )

    # 4) Retomar do melhor modelo se existir; senão, iniciar do zero
    if os.path.isfile(BEST_MODEL_FILE):
        print(f"[resume] Carregando best_model: {BEST_MODEL_FILE}")
        model = PPO.load(BEST_MODEL_FILE, env=env, print_system_info=True)
        # Reduzir LR para fine-tuning estável
        try:
            # em SB3 2.x, learning_rate pode ser float ou scheduler; tratar genericamente
            if isinstance(model.learning_rate, float):
                new_lr = model.learning_rate * 0.1
                model.learning_rate = new_lr
            else:
                # se for scheduler/callable, substitua pelo float reduzido
                base_lr = 2.5e-4
                model.learning_rate = base_lr * 0.1
            print(f"[resume] LR reduzida para {model.learning_rate}")
        except Exception as e:
            print(f"[resume] Aviso: não foi possível ajustar LR automaticamente: {e}")
    else:
        print("[new] Treinamento do zero com PPO (CnnPolicy)")
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=2.5e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log="./ppo_tensorboard/",
            verbose=1,
        )

    # 5) Callback: checkpoints + avaliação + salvar VecNormalize
    callback = CheckpointAndEvalCallback(
        save_freq_steps=10_000,
        ckpt_dir=CHECKPOINT_DIR,
        best_file=BEST_MODEL_FILE,
        norm_file=NORM_FILE,
        n_eval_episodes=5,
        verbose=1,
    )

    # 6) Treinamento
    TIMESTEPS = int(os.environ.get("PPO_TOTAL_STEPS", "1000000"))
    print("=" * 80)
    print(f"Iniciando learn(): total_timesteps={TIMESTEPS} | workers={N_CPU}")
    print("Avaliação periódica + checkpoints a cada 10k steps; vídeos gravados em video_logs/")
    print("=" * 80)

    model.learn(total_timesteps=TIMESTEPS, callback=callback, reset_num_timesteps=False)

    # 7) Salvar estado final e stats
    final_ckpt = os.path.join(CHECKPOINT_DIR, f"ppo_final_{int(time.time())}.zip")
    model.save(final_ckpt)
    vec_norm = VecNormalize.get_vec_normalize_env(model.get_env())
    if vec_norm is not None:
        vec_norm.save(NORM_FILE)

    print("=" * 80)
    print("✅ Treinamento finalizado.")
    print(f"Último checkpoint: {final_ckpt}")
    print(f"Best model:        {BEST_MODEL_FILE}")
    print(f"VecNormalize:      {NORM_FILE}")
    print(f"Vídeos:            {VIDEO_DIR}")
    print("=" * 80)

    # 8) Fechamento
    env.close()
