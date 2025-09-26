from . import *

def build_scenario(builder):
    cfg = builder.config()
    cfg.game_duration = 3000
    cfg.right_team_difficulty = 1
    cfg.left_team_difficulty = 0.5
    cfg.deterministic = False

    # Fixar lados: LEFT = atacante (AGENTE), RIGHT = defesa + GK
    # (Pixels só funcionam para agente no LEFT na sua build)

    # LEFT (AGENTE atacante) -----------------------------------------
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-0.9,  0.0, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(-0.2,  0.1, e_PlayerRole_CM, controllable=True)   # ⬅️ AGENTE (único controlável)
    builder.AddPlayer(-0.2, -0.1, e_PlayerRole_CF, controllable=False)
    builder.AddPlayer(-0.4, -0.1, e_PlayerRole_CF, controllable=False)

    # RIGHT (defesa + GK heurístico) ---------------------------------
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer( 0.9,  0.0, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer( 0.4,  0.1, e_PlayerRole_CB, controllable=False)
    builder.AddPlayer( 0.4, -0.1, e_PlayerRole_CB, controllable=False)
    builder.AddPlayer( 0.5, -0.1, e_PlayerRole_CB, controllable=False)

    # Log simples da configuração e quem é controlável
    try:
        print(f"[scenario] LEFT(diff={cfg.left_team_difficulty}) AGENTE(atk) vs RIGHT(diff={cfg.right_team_difficulty}) def+GK")
        print("[scenario] Controláveis LEFT: GK(False), CM(True), CF(False), CF(False)")
        print("[scenario] Controláveis RIGHT: todos False")
    except Exception:
        pass
