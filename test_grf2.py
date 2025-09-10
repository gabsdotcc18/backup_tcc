import os
import glob
import csv
import json
import shutil
import math
import gc

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

OUTPUT_DIR = f"frames_{SCENARIO_NAME}"                  # PNGs + métricas
VIDEO_LOGDIR = os.path.join(OUTPUT_DIR, "video_logs")    # engine video (se suportado)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_LOGDIR, exist_ok=True)

# Limpa frames e vídeos antigos (sobrescrever sempre)
for old_png in glob.glob(os.path.join(OUTPUT_DIR, "frame_*.png")):
    try: os.remove(old_png)
    except: pass
for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
    for old_vid in glob.glob(os.path.join(VIDEO_LOGDIR, ext)):
        try: os.remove(old_vid)
        except: pass

# Limpa métricas antigas
for old_file in [
    os.path.join(OUTPUT_DIR, "events_legacy.csv"),
    os.path.join(OUTPUT_DIR, "events_user.csv"),
    os.path.join(OUTPUT_DIR, "metrics.json"),
    os.path.join(OUTPUT_DIR, f"{SCENARIO_NAME}_fallback.mp4"),
    os.path.join(OUTPUT_DIR, "info_debug.jsonl"),
]:
    try:
        if os.path.isdir(old_file): shutil.rmtree(old_file, ignore_errors=True)
        else: os.remove(old_file)
    except: pass

MAX_STEPS = 500
CHANNEL_W, CHANNEL_H = 640, 480   # (largura, altura) -> obs vem (H, W, 3)

# Fallback MP4 (você já instalou o encoder)
MAKE_MP4_FROM_PNGS = True
FPS_FALLBACK = 30

# Saídas
METRICS_JSON       = os.path.join(OUTPUT_DIR, "metrics.json")
EVENTS_LEGACY_CSV  = os.path.join(OUTPUT_DIR, "events_legacy.csv")
EVENTS_USER_CSV    = os.path.join(OUTPUT_DIR, "events_user.csv")
INFO_DBG           = os.path.join(OUTPUT_DIR, "info_debug.jsonl")

print(f"▶ Rodando cenário: {SCENARIO_NAME}")
print(f"⏱ Máximo de steps: {MAX_STEPS}")
print(f"🖼 Obs (pixels) = ({CHANNEL_H}, {CHANNEL_W}, 3)")
print(f"🎥 Vídeo do motor (substitui antigos) em: {VIDEO_LOGDIR}/")

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
    # tenta forçar dumps (nem toda versão aceita)
    try:
        return football_env.create_environment(
            **base_kwargs,
            dump_full_episodes=True,
            write_full_episode_dumps=True,
        )
    except TypeError:
        return football_env.create_environment(**base_kwargs)

env = create_env_pixels()

# =========================
# Utils — frames e estado interno
# =========================
def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[2] > 3:
        frame = frame[:, :, :3]
    return frame

def extract_frame(obs):
    if isinstance(obs, np.ndarray):
        return _to_uint8_rgb(obs)
    if isinstance(obs, (list, tuple)) and len(obs) > 0:
        return extract_frame(obs[0])
    if isinstance(obs, dict) and "frame" in obs:
        return _to_uint8_rgb(np.array(obs["frame"]))
    raise TypeError(f"Formato de observação inesperado para pixels: {type(obs)}; exemplo={str(obs)[:200]}")

def _get_internal_observation(env_obj):
    for attr in ["_env", "env"]:
        inner = getattr(env_obj.unwrapped, attr, None)
        if inner is None: continue
        for fn_name in ["_observation", "observation"]:
            fn = getattr(inner, fn_name, None)
            if callable(fn):
                try:
                    raw = fn()
                    if isinstance(raw, dict):
                        return raw
                except Exception:
                    pass
    return None

def collect_state(info, env_obj, dbg_writer=None, step=None):
    """Extrai estado padronizado do mesmo env (info + raw)."""
    out = {
        "score": info.get("score", None),
        "ball_owned_team": info.get("ball_owned_team", None),
        "ball_owned_player": info.get("ball_owned_player", None),
        "ball_position": info.get("ball_position", None),
        "left_team": info.get("left_team", None),
        "right_team": info.get("right_team", None),
    }
    need_raw = any(out[k] is None for k in ["score", "ball_owned_team", "ball_owned_player", "ball_position"]) \
               or out["left_team"] is None or out["right_team"] is None
    raw = None
    if need_raw:
        raw = _get_internal_observation(env_obj)
        if isinstance(raw, dict):
            if out["score"] is None:
                sc = raw.get("score")
                if isinstance(sc, (list, tuple)) and len(sc) >= 2:
                    out["score"] = (int(sc[0]), int(sc[1]))
            if out["ball_owned_team"] is None:
                bot = raw.get("ball_owned_team")
                if isinstance(bot, (int, np.integer)):
                    out["ball_owned_team"] = int(bot)
            if out["ball_owned_player"] is None:
                bop = raw.get("ball_owned_player")
                if isinstance(bop, (int, np.integer)):
                    out["ball_owned_player"] = int(bop)
            if out["ball_position"] is None:
                bp = None
                for key in ("ball_position", "ball", "ball_coords"):
                    val = raw.get(key)
                    if isinstance(val, (list, tuple, np.ndarray)):
                        bp = val; break
                if isinstance(bp, (list, tuple, np.ndarray)) and len(bp) >= 2:
                    out["ball_position"] = [float(bp[0]), float(bp[1])]
                else:
                    pr = raw.get("players_raw")
                    if isinstance(pr, (list, tuple)) and len(pr) > 0 and isinstance(pr[0], dict):
                        b = pr[0].get("ball")
                        if isinstance(b, (list, tuple, np.ndarray)) and len(b) >= 2:
                            out["ball_position"] = [float(b[0]), float(b[1])]
            if out["left_team"] is None:
                lt = raw.get("left_team")
                if isinstance(lt, (list, tuple, np.ndarray)) and len(lt) > 0:
                    out["left_team"] = np.array(lt, dtype=float).tolist()
            if out["right_team"] is None:
                rt = raw.get("right_team")
                if isinstance(rt, (list, tuple, np.ndarray)) and len(rt) > 0:
                    out["right_team"] = np.array(rt, dtype=float).tolist()

    # padrões
    out.setdefault("score", (0, 0))
    out.setdefault("ball_owned_team", -1)
    out.setdefault("ball_owned_player", -1)
    if out.get("ball_position") is None:
        out["ball_position"] = None
    if out.get("left_team") is None:
        out["left_team"] = None
    if out.get("right_team") is None:
        out["right_team"] = None

    if dbg_writer is not None:
        try:
            dbg_writer.write(json.dumps({
                "step": step,
                "keys_info": sorted(list(info.keys())),
                "sample": {
                    "score": out["score"],
                    "bot": out["ball_owned_team"],
                    "bop": out["ball_owned_player"],
                    "ball": out["ball_position"],
                    "lt_len": len(out["left_team"]) if isinstance(out["left_team"], list) else None,
                    "rt_len": len(out["right_team"]) if isinstance(out["right_team"], list) else None,
                }
            }, ensure_ascii=False) + "\n")
        except Exception:
            pass
    return out

# =========================
# Geometria e helpers
# =========================
def vsub(a, b): return (a[0]-b[0], a[1]-b[1])
def vlen(v): return math.hypot(v[0], v[1])
def vdot(a, b): return a[0]*b[0] + a[1]*b[1]
def angle_between(a, b):
    la, lb = vlen(a), vlen(b)
    if la == 0 or lb == 0: return math.pi
    cos = max(-1.0, min(1.0, vdot(a, b)/(la*lb)))
    return math.acos(cos)

GOAL_CENTER = {0: ( 1.0, 0.0), 1: (-1.0, 0.0)}  # time 0 chuta à direita; time 1 chuta à esquerda
GOAL_CONE_DEG = 12.0   # “direção ao gol” se ângulo <= 12°
PASS_CONE_DEG = 18.0   # “direção ao companheiro” se ângulo <= 18°
DEFENSE_RADIUS = 0.18  # “perto do gol” p/ considerar defesa
BALL_MIN_SPEED = 0.015 # velocidade mínima para considerar “chute/passe”
RECEIVE_RADIUS = 0.06  # “bola no pé” ao receber
WINDOW_PASS = 25       # janela de conclusão do passe
WINDOW_SHOT = 45       # janela de desfecho do chute

# =========================
# Trackers (duas famílias no MESMO estado)
# =========================
class BaseTracker:
    def __init__(self):
        self.events = []
        self.metrics = {
            0: {"passes_tentados": 0, "passes_certos": 0,
                "finalizacoes": 0, "finalizacoes_certas": 0, "finalizacoes_erradas": 0, "gols": 0},
            1: {"passes_tentados": 0, "passes_certos": 0,
                "finalizacoes": 0, "finalizacoes_certas": 0, "finalizacoes_erradas": 0, "gols": 0},
        }
        self.prev_score = (0, 0)
        self.prev_owner_team = -1
        self.prev_owner_player = -1
        self.prev_ball = None
        self.pending_pass = None  # {"team","from_idx","step","towards_idx"}
        self.pending_shot = None  # {"team","step"}

    def log(self, step, name, team=None, **kw):
        self.events.append((step, name, team if team is not None else -1, kw))

    def nearest_idx(self, ball, team_positions):
        if ball is None or team_positions is None: return None
        bx, by = ball
        dmin, idx = 1e9, None
        for i, p in enumerate(team_positions):
            d = (p[0]-bx)**2 + (p[1]-by)**2
            if d < dmin: dmin, idx = d, i
        return idx

    def crossed_goal_line(self, prev_ball, ball, team):
        if prev_ball is None or ball is None: return False
        x0, y0 = prev_ball; x1, y1 = ball
        GOAL_Y = 0.065
        def y_at_x(x_target):
            if x1 == x0: return None
            t = (x_target - x0)/(x1 - x0)
            if t < 0 or t > 1: return None
            return y0 + t*(y1 - y0)
        if team == 0:
            y_cross = y_at_x(1.0)
            return (y_cross is not None) and (abs(y_cross) <= GOAL_Y + 0.02)
        else:
            y_cross = y_at_x(-1.0)
            return (y_cross is not None) and (abs(y_cross) <= GOAL_Y + 0.02)

class LegacyTracker(BaseTracker):
    """Heurística anterior (para comparação)."""
    def update(self, step, s):
        score = s["score"]; owner_t = s["ball_owned_team"]; owner_p = s["ball_owned_player"]
        ball = s["ball_position"]; lt, rt = s["left_team"], s["right_team"]

        # gols
        if score != self.prev_score:
            dL = score[0]-self.prev_score[0]; dR = score[1]-self.prev_score[1]
            if dL>0: self.metrics[0]["gols"]+=dL; self.metrics[0]["finalizacoes"]+=dL; self.metrics[0]["finalizacoes_certas"]+=dL; self.log(step,"gol",0,delta=dL,score=score)
            if dR>0: self.metrics[1]["gols"]+=dR; self.metrics[1]["finalizacoes"]+=dR; self.metrics[1]["finalizacoes_certas"]+=dR; self.log(step,"gol",1,delta=dR,score=score)

        # passes simples: troca de owner_p no mesmo time
        if owner_t in (0,1) and self.prev_owner_team == owner_t:
            if self.prev_owner_player != -1 and owner_p != -1 and owner_p != self.prev_owner_player:
                self.metrics[owner_t]["passes_certos"] += 1
                self.metrics[owner_t]["passes_tentados"] += 1
                self.log(step, "passe_certo_legacy", owner_t, from_player=self.prev_owner_player, to_player=owner_p)

        # perda de posse direta -> passe tentado
        if self.prev_owner_team in (0,1) and owner_t in (0,1,-1):
            if owner_t != self.prev_owner_team and self.prev_owner_player != -1:
                losing = self.prev_owner_team
                self.metrics[losing]["passes_tentados"] += 1
                if owner_t in (0,1) and owner_t != losing:
                    self.log(step,"passe_errado_legacy",losing,from_player=self.prev_owner_player,to_team=owner_t)

        # finalizações (zona + janela + linha do gol)
        if owner_t in (0,1):
            last_touch = owner_t
        else:
            last_touch = self.prev_owner_team

        # abre janela se bola entra na faixa central perto do gol adversário
        def in_zone(b, team):
            if b is None: return False
            x,y = b; central = abs(y) < 0.15
            return (central and ((team==0 and x>0.7) or (team==1 and x<-0.7)))

        if in_zone(ball, last_touch) and self.pending_shot is None:
            self.pending_shot = {"team": last_touch, "step": step}
            self.metrics[last_touch]["finalizacoes"] += 1
            self.log(step, "finalizacao_tentada_legacy", last_touch)

        if self.pending_shot is not None:
            t = self.pending_shot["team"]
            if step - self.pending_shot["step"] > 30:
                on_target = self.crossed_goal_line(self.prev_ball, ball, t)
                if on_target:
                    self.metrics[t]["finalizacoes_certas"] += 1
                    self.log(step, "finalizacao_no_alvo_legacy", t)
                else:
                    self.metrics[t]["finalizacoes_erradas"] += 1
                    self.log(step, "finalizacao_fora_legacy", t)
                self.pending_shot = None
            else:
                if score != self.prev_score:
                    self.pending_shot = None

        # estado
        self.prev_score = score
        self.prev_owner_team = owner_t
        self.prev_owner_player = owner_p
        self.prev_ball = ball[:] if ball is not None else None

class UserDefsTracker(BaseTracker):
    """
    Suas definições:
      - Passe tentado: bola sai do pé de um jogador A e vai na direção de um companheiro (cone PASS_CONE_DEG),
        e NÃO está indo para o gol (cone GOAL_CONE_DEG).
      - Passe certo: mesmo que acima, mas a bola chega ao pé de um companheiro (recepção) na janela.
      - Finalização certa: bola é chutada na direção do gol (cone GOAL_CONE_DEG) e vira GOL ou é DEFENDIDA
        (adversário ganha posse nas redondezas do gol) dentro da janela.
    """
    def update(self, step, s):
        score = s["score"]; owner_t = s["ball_owned_team"]; owner_p = s["ball_owned_player"]
        ball = s["ball_position"]; lt, rt = s["left_team"], s["right_team"]

        # gols
        if score != self.prev_score:
            dL = score[0]-self.prev_score[0]; dR = score[1]-self.prev_score[1]
            if dL>0: self.metrics[0]["gols"]+=dL; self.metrics[0]["finalizacoes"]+=dL; self.metrics[0]["finalizacoes_certas"]+=dL; self.log(step,"gol",0,delta=dL,score=score)
            if dR>0: self.metrics[1]["gols"]+=dR; self.metrics[1]["finalizacoes"]+=dR; self.metrics[1]["finalizacoes_certas"]+=dR; self.log(step,"gol",1,delta=dR,score=score)

        # detectar "bola saiu do pé" (kick)
        prev_ball = self.prev_ball
        v = None
        speed = 0.0
        if ball is not None and prev_ball is not None:
            v = vsub(ball, prev_ball)
            speed = vlen(v)

        # quem chutou? se owner_t == -1 e antes tínhamos dono (prev_owner_player != -1), assumimos chute de prev_owner
        kick_detected = False
        kicker_team, kicker_player = None, None
        if speed > BALL_MIN_SPEED:
            if self.prev_owner_team in (0,1) and owner_t == -1:
                kick_detected = True
                kicker_team = self.prev_owner_team
                kicker_player = self.prev_owner_player
            # fallback: mesmo mantendo posse, se a bola acelerou bastante, consideramos toque/chute do atual dono
            elif owner_t in (0,1) and self.prev_owner_team == owner_t and owner_p == self.prev_owner_player:
                kick_detected = True
                kicker_team = owner_t
                kicker_player = owner_p

        # alvo gol?
        is_towards_goal = False
        goal_vec = None
        if kick_detected and v is not None:
            gcx, gcy = GOAL_CENTER[kicker_team]
            goal_vec = vsub((gcx, gcy), prev_ball if prev_ball is not None else ball)
            if goal_vec is not None:
                ang_goal = math.degrees(angle_between(v, goal_vec))
                is_towards_goal = (ang_goal <= GOAL_CONE_DEG)

        # alvo companheiro?
        towards_teammate = False
        target_idx = None
        if kick_detected and not is_towards_goal:
            team_positions = lt if kicker_team == 0 else rt
            best_ang, best_idx = 999.0, None
            if team_positions is not None and prev_ball is not None:
                for i, p in enumerate(team_positions):
                    if kicker_player is not None and i == kicker_player:
                        continue
                    to_teammate = vsub(p, prev_ball)
                    ang = math.degrees(angle_between(v, to_teammate))
                    # preferimos quem está na frente do vetor e relativamente perto do cone
                    if ang < best_ang:
                        best_ang, best_idx = ang, i
            if best_idx is not None and best_ang <= PASS_CONE_DEG:
                towards_teammate, target_idx = True, best_idx

        # marca passe tentado / chute tentado
        # (não contam dupla vez: prioridade para shot se for ao gol)
        if kick_detected and is_towards_goal:
            if self.pending_shot is None:
                self.pending_shot = {"team": kicker_team, "step": step}
                self.metrics[kicker_team]["finalizacoes"] += 1
                self.log(step, "finalizacao_tentada_user", kicker_team,
                         kicker_player=kicker_player, speed=round(speed,4))
        elif kick_detected and towards_teammate:
            # abre janela do passe
            self.pending_pass = {"team": kicker_team, "from_idx": kicker_player,
                                 "towards_idx": target_idx, "step": step}
            self.metrics[kicker_team]["passes_tentados"] += 1
            self.log(step, "passe_tentado_user", kicker_team,
                     from_idx=kicker_player, to_idx=target_idx, speed=round(speed,4))

        # desfecho do passe
        if self.pending_pass is not None:
            pt = self.pending_pass
            # recepção certa: posse volta para o mesmo time, com jogador diferente (ou bola tocando o target)
            if owner_t == pt["team"] and owner_p != -1 and owner_p != pt["from_idx"]:
                # distância bola-receptor confirma “no pé”
                team_positions = lt if owner_t == 0 else rt
                recv_pos = team_positions[owner_p] if (team_positions and owner_p < len(team_positions)) else None
                if ball is not None and recv_pos is not None and vlen(vsub(ball, recv_pos)) <= RECEIVE_RADIUS:
                    self.metrics[pt["team"]]["passes_certos"] += 1
                    self.log(step, "passe_certo_user", pt["team"],
                             from_idx=pt["from_idx"], to_idx=owner_p)
                    self.pending_pass = None
            # janela expira sem recepção
            elif step - pt["step"] > WINDOW_PASS:
                self.pending_pass = None

        # desfecho do chute
        if self.pending_shot is not None:
            st = self.pending_shot
            # gol já conta acima e encerra
            if score != self.prev_score:
                self.pending_shot = None
            elif step - st["step"] > WINDOW_SHOT:
                # defesa: adversário ganha posse “perto do gol”
                adv = 1 - st["team"]
                # posição da bola próxima ao gol adversário?
                gx, gy = GOAL_CENTER[st["team"]]
                defended = False
                if ball is not None:
                    if abs(ball[0]-gx) < 0.15 and abs(ball[1]-gy) < DEFENSE_RADIUS and owner_t == adv:
                        defended = True
                if defended:
                    self.metrics[st["team"]]["finalizacoes_certas"] += 1
                    self.log(step, "finalizacao_defendida_user", st["team"])
                else:
                    self.metrics[st["team"]]["finalizacoes_erradas"] += 1
                    self.log(step, "finalizacao_fora_user", st["team"])
                self.pending_shot = None

        # estado
        self.prev_score = score
        self.prev_owner_team = owner_t
        self.prev_owner_player = owner_p
        self.prev_ball = ball[:] if ball is not None else None

# =========================
# Execução principal (um env; duas famílias no mesmo estado)
# =========================
obs = env.reset()
done = False
step = 0

dbg = open(INFO_DBG, "w", encoding="utf-8")

legacy = LegacyTracker()
user = UserDefsTracker()

# primeiro frame
frame0 = extract_frame(obs)
print(f"🔎 shape inicial do frame: {frame0.shape}, dtype={frame0.dtype}")
imageio.imwrite(os.path.join(OUTPUT_DIR, f"frame_{step:04d}.png"), frame0)

# estado inicial
s0 = collect_state({}, env, dbg_writer=dbg, step=step)
legacy.update(step, s0)
user.update(step, s0)
step += 1

with tqdm(total=MAX_STEPS - 1, desc="Frames + métricas (1 env, 2 famílias)", ncols=92) as pbar:
    while not done and step < MAX_STEPS:
        action = env.action_space.sample()
        obs, reward, done, info = env.step([action])

        # salva PNG
        frame = extract_frame(obs)
        imageio.imwrite(os.path.join(OUTPUT_DIR, f"frame_{step:04d}.png"), frame)

        # mesmo estado para ambos
        state = collect_state(info or {}, env, dbg_writer=dbg, step=step)

        legacy.update(step, state)
        user.update(step, state)

        step += 1
        pbar.update(1)

dbg.close()

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
# Fallback MP4 a partir dos PNGs (substitui antigo pois apagamos no início)
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
        print("💡 Já que você instalou o render MP4, confirme também no container:  pip install 'imageio[ffmpeg]'  ou apt-get install -y ffmpeg")

# =========================
# Dump de eventos e métricas
# =========================
def dump_events(csv_path, events):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["step", "evento", "time", "extra_json"])
        for (s, name, team, extra) in events:
            w.writerow([s, name, team, json.dumps(extra, ensure_ascii=False)])

dump_events(EVENTS_LEGACY_CSV, legacy.events)
dump_events(EVENTS_USER_CSV, user.events)

totals = {
    "legacy": legacy.metrics,
    "user": user.metrics
}
with open(METRICS_JSON, "w", encoding="utf-8") as f:
    json.dump(totals, f, ensure_ascii=False, indent=2)

print("📊 Métricas finais (resumo):")
print(json.dumps(totals, ensure_ascii=False, indent=2))
print(f"📝 Eventos (legacy) em: {EVENTS_LEGACY_CSV}")
print(f"📝 Eventos (user)   em: {EVENTS_USER_CSV}")
print(f"🧾 Resumo em: {METRICS_JSON}")
print(f"🔍 Log de depuração das infos em: {INFO_DBG}")
