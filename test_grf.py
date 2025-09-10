import gfootball.env as football_env
import os
from PIL import Image

# Parâmetros
scenario = 'academy_single_goal_versus_lazy'
render = False
write_frames = True
output_dir = 'frames_single_goal_lazy'
max_steps = 3000

# ⚠️ Número de jogadores controlados (mude para 1, 4 ou 11)
n_controlled_players = 1  # escolha um número adequado

print(f'▶ Rodando cenário: {scenario}')
print(f'⏱ Máximo de steps: {max_steps}')

# Cria o ambiente
env = football_env.create_environment(
    env_name=scenario,
    render=render,
    write_full_episode_dumps=False,
    write_video=False,
    logdir='logs',
    representation='simple115',
    number_of_left_players_agent_controls=n_controlled_players,
)

obs = env.reset()

if write_frames:
    os.makedirs(output_dir, exist_ok=True)

for step in range(max_steps):
    # Gera ações válidas
    if n_controlled_players == 1:
        action = env.action_space.sample()  # só uma ação
    else:
        action = [env.action_space.sample() for _ in range(n_controlled_players)]

    obs, reward, done, info = env.step(action)

    if write_frames:
        frame = env.render(mode='rgb_array')
        if frame is not None:
            img = Image.fromarray(frame)
            img.save(f'{output_dir}/frame_{step:04d}.png')

    if done:
        print(f"🏁 Episódio terminou no step {step+1}")
        break

env.close()
print(f'✅ Finalizado. Frames salvos em: {output_dir}/')
