# PPO33.py  —  Treino incremental até 1 milhão de steps em blocos de 10k
# (SB3 2.x + Gymnasium + shimmy + GRF) — mantém feedbacks constantes e best_model a cada checkpoint
#
# Requisitos (pip):
#   torch==2.2.2 torchvision==0.17.2
#   stable_baselines3==2.3.2 gymnasium==0.29.1 shimmy==1.3.0
#   opencv-python tensorboard moviepy imageio[ffmpeg]
#
# Observações:
# - Pixels exigem render=True no GRF (senão "Missing 'frame'").
# - VecNormalize: NUNCA normalize obs de imagem (norm_obs=False).
# - Este script treina incrementalmente até TARGET_TOTAL_STEPS (padrão: 1_000_000),
#   em blocos de TIMESTEPS_PER_BLOCK (padrão: 10_000), salvando checkpoint, best_model
#   e VecNormalize ao final de CADA bloco.
# - Feedbacks constantes: logs do SB3 (log_interval=1) + prints de progresso por bloco.

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
    VecNormalize,
)
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure


# =========================
# Configurações principais
# =========================
SCENARIO_NAME = "three_vs_three"  # seu 3x3 com GK na defesa
OUT_DIR = "."
FRAME_DIR = os.path.join(OUT_DIR, "frames_three_vs_three")
CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints_ppo")
BEST_DIR = os.path.join(CHECKPOINT_DIR, "best_model")
BEST_MODEL_FILE = os.path.join(BEST_DIR, "best_model.zip")
NORM_FILE = os.path.join(CHECKPOINT_DIR, "vecnormalize.pkl")

os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

# Hiperparâmetros/execução
N_CPU = int(os.environ.get("PPO_WORKERS", "1"))  # use 1 para validar; depois 2-4 se quiser
BASE_SEED = int(os.environ.get("PPO_SEED", "42"))
TIMESTEPS_PER_BLOCK = int(os.environ.get("PPO_STEPS_PER_BLOCK", "10000"))  # bloco de 10k
SAVE_FREQ_STEPS = TIMESTEPS_PER_BLOCK  # salvar/avaliar no fim de cada bloco
TARGET_TOTAL_STEPS = int(os.environ.get("PPO_TARGET_TOTAL_STEPS", "1000000"))  # alvo global: 1M
REDUCE_LR_ON_RESUME = os.environ.get("REDUCE_LR_ON_RESUME", "0") == "1"  # por padrão, NÃO reduzir LR a cada retomada


# =========================
# Helpers
# =========================
def _find_vecnormalize(env):
    """Encontra VecNormalize em uma pilha de wrappers (compatível SB3 2.x)."""
    e, seen = env, set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, VecNormalize):
            return e
        e = getattr(e, "venv", None) or getattr(e, "env", None)
    return None


def make_single_env(seed: int) -> Callable[[], gym.Env]:
    """Fábrica de env: GRF (Gym v21) -> shimmy -> Gymnasium."""
    def _init():
        old_env = football_env.create_environment(
            env_name=SCENARIO_NAME,
            representation="pixels",
            render=True,                     # NECESSÁRIO para pixels
            channel_dimensions=(64, 64),     # pode reduzir para 56x56 ou 48x48 para ganhar FPS
            number_of_left_players_agent_controls=1,   # 1 atacante treinando
            number_of_right_players_agent_controls=0,  # defesa + GK heurístico
        )
        env = GymV21CompatibilityV0(env=old_env)
        env.reset(seed=seed)
        return env
    return _init


def build_vec_env(n_cpu: int = 1, base_seed: int = 1234) -> gym.Env:
    set_random_seed(base_seed)
    env_fns = [make_single_env(base_seed + i) for i in range(n_cpu)]
    venv = DummyVecEnv(env_fns) if n_cpu == 1 else SubprocVecEnv(env_fns)
    venv = VecMonitor(venv)
    return venv


# =========================
# Callback: avaliação + best + checkpoint + VecNormalize
# =========================
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
        self.eval_env = None  # criado lazy

    def _make_eval_env(self):
        if self.eval_env is not None:
            return self.eval_env

        def _f():
            old_env = football_env.create_environment(
                env_name=SCENARIO_NAME,
                representation="pixels",
                render=True,
                channel_dimensions=(64, 64),
                number_of_left_players_agent_controls=1,
                number_of_right_players_agent_controls=0,
            )
            return GymV21CompatibilityV0(env=old_env)

        base = DummyVecEnv([_f])

        # copiar stats do VecNormalize de treino, mas em modo avaliação
        train_vec = _find_vecnormalize(self.model.get_env())
        if train_vec is not None:
            self.eval_env = VecNormalize(base, training=False)
            self.eval_env.obs_rms = train_vec.obs_rms
            self.eval_env.ret_rms = train_vec.ret_rms
            self.eval_env.clip_obs = train_vec.clip_obs
            self.eval_env.clip_reward = train_vec.clip_reward
            self.eval_env.gamma = train_vec.gamma
        else:
            self.eval_env = base
        return self.eval_env

    def _evaluate_deterministic(self, episodes: int):
        env = self._make_eval_env()
        ep_rewards = []
        for _ in range(episodes):
            obs = env.reset()
            done = [False]
            ep_r = 0.0
            while not all(done):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, done, infos = env.step(action)
                ep_r += float(rewards[0])
            ep_rewards.append(ep_r)
        return float(np.mean(ep_rewards)) if ep_rewards else 0.0

    def _on_step(self) -> bool:
        # executa no fim de cada bloco (save_freq_steps)
        if self.num_timesteps % self.save_freq_steps != 0:
            return True

        # 1) Avaliar primeiro (garante best mesmo se salvar stats falhar)
        try:
            mean_reward = self._evaluate_deterministic(self.n_eval_episodes)
            if self.verbose:
                print(f"[eval] mean_reward(n={self.n_eval_episodes}) = {mean_reward:.3f}")
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.best_file)
                if self.verbose:
                    print(f"[best] Novo melhor modelo salvo em: {self.best_file} (reward={self.best_mean_reward:.3f})")
        except Exception as e:
            print(f"[eval] Aviso: falha na avaliação: {e}")

        # 2) Salvar checkpoint incremental (sempre)
        ckpt_path = os.path.join(self.ckpt_dir, f"ppo_{self.num_timesteps}_steps.zip")
        self.model.save(ckpt_path)
        if self.verbose:
            print(f"[checkpoint] Modelo salvo em: {ckpt_path}")

        # 3) Salvar stats do VecNormalize (não travar o best se falhar)
        try:
            vec_norm = _find_vecnormalize(self.model.get_env())
            if vec_norm is not None:
                vec_norm.save(self.norm_file)
                if self.verbose:
                    print(f"[normalize] Stats VecNormalize salvas em: {self.norm_file}")
        except Exception as e:
            print(f"[normalize] Aviso: falha ao salvar VecNormalize: {e}")

        return True


# =========================
# Main
# =========================
if __name__ == "__main__":
    print("=" * 80)
    print("Iniciando treinamento (incremental) — cenário: three_vs_three (1 atacante com PPO)")
    print(f"Bloco de steps: {TIMESTEPS_PER_BLOCK} | Workers: {N_CPU}")
    print(f"Alvo global (TARGET_TOTAL_STEPS): {TARGET_TOTAL_STEPS}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print(f"Best model:  {BEST_MODEL_FILE}")
    print(f"VecNorm:     {NORM_FILE}")
    print("=" * 80)

    # 1) VecEnv base
    base_env = build_vec_env(n_cpu=N_CPU, base_seed=BASE_SEED)

    # 2) Aplicar VecNormalize (norm_obs=False para imagens)
    if os.path.isfile(NORM_FILE):
        print(f"[resume] Carregando VecNormalize de: {NORM_FILE}")
        env = VecNormalize.load(NORM_FILE, base_env)
        # garantir flags corretas
        env.training = True
        env.norm_obs = False
        env.norm_reward = True
    else:
        env = VecNormalize(base_env, norm_obs=False, norm_reward=True, clip_obs=10.0)
        env.training = True

    # 3) Carregar best_model se existir, senão fallback para último checkpoint; caso contrário, começar do zero
    latest_ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "ppo_*_steps.zip")), key=os.path.getmtime)
    resume_path = BEST_MODEL_FILE if os.path.isfile(BEST_MODEL_FILE) else (latest_ckpts[-1] if latest_ckpts else None)

    if resume_path:
        print(f"[resume] Carregando: {resume_path}")
        model = PPO.load(resume_path, env=env, print_system_info=True)
        # logger para stdout
        model.set_logger(configure(OUT_DIR, ["stdout"]))
        if REDUCE_LR_ON_RESUME:
            try:
                base_lr = 2.5e-4
                new_lr = (model.learning_rate if isinstance(model.learning_rate, float) else base_lr) * 0.1
                model.learning_rate = new_lr
                print(f"[resume] LR reduzida para {model.learning_rate}")
            except Exception as e:
                print(f"[resume] Aviso: não foi possível ajustar LR: {e}")
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
            tensorboard_log=os.path.join(OUT_DIR, "ppo_tensorboard"),
            verbose=1,
        )
        model.set_logger(configure(OUT_DIR, ["stdout"]))

    # 4) Callback (avaliar → best → checkpoint → salvar VecNormalize)
    callback = CheckpointAndEvalCallback(
        save_freq_steps=SAVE_FREQ_STEPS,
        ckpt_dir=CHECKPOINT_DIR,
        best_file=BEST_MODEL_FILE,
        norm_file=NORM_FILE,
        n_eval_episodes=5,
        verbose=1,
    )

    # Estado inicial
    start_steps = int(model.num_timesteps or 0)
    print("=" * 80)
    print(f"Estado inicial do contador SB3: num_timesteps={start_steps}")
    print("Treino incremental: múltiplos blocos até atingir TARGET_TOTAL_STEPS")
    print("Avaliação e salvamento ao fim de CADA bloco (best + checkpoint + VecNormalize)")
    print("=" * 80)

    # 5) Loop incremental até TARGET_TOTAL_STEPS
    block_idx = 0
    while (model.num_timesteps or 0) < TARGET_TOTAL_STEPS:
        current_steps = int(model.num_timesteps or 0)
        remaining = TARGET_TOTAL_STEPS - current_steps
        this_block = min(TIMESTEPS_PER_BLOCK, remaining)
        block_idx += 1

        pct = 100.0 * (current_steps / max(1, TARGET_TOTAL_STEPS))
        print("-" * 80)
        print(f"🟦 Bloco #{block_idx} — iniciando learn(): {this_block} steps")
        print(f"Progresso atual: {current_steps:,}/{TARGET_TOTAL_STEPS:,}  ({pct:.2f}% concluído)")
        print("-" * 80)

        model.learn(
            total_timesteps=this_block,
            callback=callback,          # faz eval + best + checkpoint + VecNormalize
            reset_num_timesteps=False,  # mantém contador cumulativo
            log_interval=1,             # imprime métricas a cada rollout
        )

        # 6) Salvar estado final do bloco (redundante ao callback, mas útil se interromper pós-callback)
        final_ckpt = os.path.join(CHECKPOINT_DIR, f"ppo_final_{int(time.time())}.zip")
        model.save(final_ckpt)
        vec_norm = _find_vecnormalize(model.get_env())
        if vec_norm is not None:
            vec_norm.save(NORM_FILE)

        # Feedback pós-bloco
        new_steps = int(model.num_timesteps or 0)
        pct_after = 100.0 * (new_steps / max(1, TARGET_TOTAL_STEPS))
        print("=" * 80)
        print("✅ Bloco finalizado.")
        print(f"Checkpoint final do bloco: {final_ckpt}")
        print(f"Best model (até agora):    {BEST_MODEL_FILE}  (atualiza quando melhora na avaliação)")
        print(f"VecNormalize:              {NORM_FILE}")
        print(f"Progresso: {new_steps:,}/{TARGET_TOTAL_STEPS:,}  ({pct_after:.2f}%)")
        print("=" * 80)

    # 7) Resumo final
    total_done = int(model.num_timesteps or 0)
    print("#" * 80)
    print("🏁 Treinamento incremental concluído (alvo atingido).")
    print(f"Steps executados (SB3): {total_done:,}")
    print(f"Best model final:        {BEST_MODEL_FILE}")
    print(f"Checkpoints em:          {CHECKPOINT_DIR}")
    print(f"VecNormalize final:      {NORM_FILE}")
    print("#" * 80)
