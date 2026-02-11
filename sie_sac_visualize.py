"""
SIE-SAC 학습 결과 시각화
- 학습된 모델 로드
- 드론 궤적 실시간 시각화
- NIS, Reward 분석 그래프
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 사용할 환경 선택 (둘 중 하나)
USE_SB3 = False  # True: Stable Baselines3 사용, False: 커스텀 구현 사용
USE_PAPER_ENV = True  # True: VectorizedSIEEnvPaper 사용 (SIE_SAC_paper.py 학습용)

if USE_SB3:
    from stable_baselines3 import SAC
    from SIE_SAC_env import VirtualSpoofingEnvV2 as SpoofingEnv
elif USE_PAPER_ENV:
    # Import configuration and environment from SIE_SAC_paper.py (SINGLE SOURCE)
    from SIE_SAC_paper import (
        VectorizedSIEEnvPaper,
        TRUE_DEST,
        FAKE_DEST,
        DEFAULT_ENV_CONFIG,
        DEFAULT_TRAIN_CONFIG
    )

    class SpoofingEnvWrapper:
        """
        VectorizedSIEEnvPaper를 단일 환경처럼 사용하기 위한 래퍼.
        시각화를 위해 n_envs=1로 생성하고, 인터페이스를 맞춤.
        """
        def __init__(self, config=None):
            self.env = VectorizedSIEEnvPaper(n_envs=1, config=config)
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
            self.true_dest = self.env.true_dest
            self.fake_dest = self.env.fake_dest
            self._last_radar_est = None

        def reset(self, seed=None):
            obs, radar_est, infos = self.env.reset(seed=seed)
            self._last_radar_est = radar_est[0]
            return obs[0], infos[0]

        def step(self, action):
            actions = action.reshape(1, -1)
            obs, rewards, terminateds, truncateds, radar_est_t, next_radar_est, infos = self.env.step(actions)
            self._last_radar_est = next_radar_est[0]

            # 추가 정보 제공
            info = infos[0]
            info['radar_est'] = radar_est_t[0]  # 현재 액션에 사용된 radar estimate

            return obs[0], rewards[0], terminateds[0], truncateds[0], info

        @property
        def true_pos(self):
            """드론의 실제 위치"""
            return self.env.true_pos[0]

        @property
        def radar_est(self):
            """마지막 radar estimate (x^e)"""
            return self._last_radar_est

    SpoofingEnv = SpoofingEnvWrapper
else:
    from SIE_SAC_env import SIESACEnv as SpoofingEnv


def load_model(model_path: str, env, env_config: dict = None, entropy_type: str = None):
    """모델 로드

    Args:
        model_path: 모델 파일 경로
        env: 환경 인스턴스
        env_config: 환경 설정 딕셔너리
        entropy_type: 'sie' 또는 'action' (None이면 경로에서 자동 추론)
    """
    if USE_SB3:
        try:
            model = SAC.load(model_path, env=env)
            print(f">>> SB3 모델 로드 성공: {model_path}")
            return model
        except Exception as e:
            print(f"!!! 모델 로드 실패: {e}")
            return None
    else:
        import torch
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # SIE_SAC_paper 모델인지 확인
        if 'paper' in model_path.lower():
            try:
                from SIE_SAC_paper import SIESACAgentPaper
                import numpy as np

                # env_config에서 필요한 파라미터 가져오기
                if env_config is None:
                    env_config = {}

                # entropy_type 자동 추론 (경로에서)
                if entropy_type is None:
                    if '_action' in model_path.lower() or 'action' in model_path.lower():
                        entropy_type = 'action'
                    else:
                        entropy_type = 'sie'

                print(f">>> Entropy Type: {entropy_type}")

                # Use global constants (imported from SIE_SAC_paper.py)
                agent = SIESACAgentPaper(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    action_low=env.action_space.low,
                    action_high=env.action_space.high,
                    fake_dest=FAKE_DEST,
                    true_dest=TRUE_DEST,
                    hidden_dim=env_config.get('hidden_dim', DEFAULT_TRAIN_CONFIG['hidden_dim']),
                    lr=env_config.get('lr', DEFAULT_TRAIN_CONFIG['lr']),
                    gamma=env_config.get('gamma', DEFAULT_TRAIN_CONFIG['gamma']),
                    tau=env_config.get('tau', DEFAULT_TRAIN_CONFIG['tau']),
                    alpha_init=env_config.get('alpha_init', DEFAULT_TRAIN_CONFIG['alpha_init']),
                    H_0=env_config.get('H_0', DEFAULT_TRAIN_CONFIG['H_0']),
                    lambda_sie=env_config.get('lambda_sie', DEFAULT_ENV_CONFIG['lambda_sie']),
                    rho_e=env_config.get('rho_e', DEFAULT_ENV_CONFIG['rho_e']),
                    omega_1=env_config.get('omega_1', DEFAULT_ENV_CONFIG['omega_1']),
                    entropy_type=entropy_type,
                )
                agent.load(model_path)
                print(f">>> SIE_SAC_paper 모델 로드 성공: {model_path}")
                return agent
            except Exception as e:
                print(f"!!! SIE_SAC_paper 모델 로드 실패: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            # 기존 SIE_SAC_train 모델
            try:
                from SIE_SAC_train import SIESACAgent

                agent = SIESACAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    action_low=env.action_space.low,
                    action_high=env.action_space.high,
                )
                agent.load(model_path)
                print(f">>> 커스텀 모델 로드 성공: {model_path}")
                return agent
            except Exception as e:
                print(f"!!! 모델 로드 실패: {e}")
                return None


def predict_action(model, obs, deterministic=True):
    """모델에서 액션 예측"""
    import numpy as np

    if USE_SB3:
        action, _ = model.predict(obs, deterministic=deterministic)
        return action
    else:
        # SIESACAgentPaper는 select_actions_batch 사용 (batch 입력 필요)
        # SIESACAgent는 select_action 사용 (단일 입력)
        if hasattr(model, 'select_actions_batch'):
            # SIESACAgentPaper: 배치 입력 필요
            obs_batch = obs[np.newaxis, :] if obs.ndim == 1 else obs
            action = model.select_actions_batch(obs_batch, evaluate=deterministic)
            return action[0] if obs.ndim == 1 else action
        else:
            # SIESACAgent: 단일 입력
            return model.select_action(obs, evaluate=deterministic)


def run_episode(env, model, max_steps=10000):
    """에피소드 실행 및 데이터 수집"""
    obs, _ = env.reset()

    # 실제 시작 위치 저장 (랜덤 노이즈 포함)
    if USE_PAPER_ENV:
        actual_start_pos = env.true_pos.copy()
    elif hasattr(env, 'simulator'):
        actual_start_pos = env.simulator.true_pos.copy()
    else:
        actual_start_pos = np.array([0.0, 0.0, -20.0])  # fallback

    # 데이터 기록용
    data = {
        'start_position': actual_start_pos,  # 실제 시작 위치 (랜덤 노이즈 포함)
        'drone_positions': [],      # 드론 실제 위치
        'spoof_positions': [],      # 기만 신호 위치 (x^s = x^e + Δx^s)
        'radar_estimates': [],      # 레이더 추정 위치 (x^e)
        'nis_values': [],           # NIS 값
        'gamma_s_values': [],       # 예측 NIS (γ^s)
        'rewards': [],              # 보상
        'actions': [],              # 액션
        'distances_to_fake': [],    # 기만 목적지까지 거리
        'distances_to_true': [],    # 실제 목적지까지 거리
        # === DEBUG: Z-axis bias investigation ===
        'spoof_offset_xyz': [],     # Spoofing offset in Cartesian (dx, dy, dz)
        'M_radar_diag': [],         # M_radar diagonal (xx, yy, zz)
        'reward_components': [],    # Individual reward components (r_x, r_v, r_gamma)
    }

    done = False
    step = 0

    while not done and step < max_steps:
        # 액션 예측
        if model is not None:
            action = predict_action(model, obs, deterministic=True)
        else:
            # 모델 없으면 랜덤 액션
            action = env.action_space.sample()

        # 환경 진행
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 데이터 기록 - info에서 가져오기 (auto-reset 전 데이터)
        # env.true_pos는 auto-reset 후 값이므로 사용하면 안됨!
        if USE_PAPER_ENV:
            # VectorizedSIEEnvPaper: info에 pre-reset 데이터가 있음
            true_pos = info.get('true_pos', env.true_pos.copy())
            radar_est = info.get('radar_est', true_pos)
        elif hasattr(env, 'simulator'):
            # 기존 SIESACEnv 사용
            true_pos = env.simulator.true_pos.copy()
            radar_est = true_pos  # 기존 환경은 radar_est가 없을 수 있음
        else:
            true_pos = info.get('true_pos', np.zeros(3))
            radar_est = info.get('radar_est', true_pos)

        # 기만 신호 위치 - ACTUAL position sent to UAV (1-step delay)
        # Use info['deceptive_pos'] which is what UAV actually received this step
        # NOT action-based calculation (that's for NEXT step due to 1-step delay!)
        if 'deceptive_pos' in info:
            spoof_pos = np.array(info['deceptive_pos'])
        else:
            # Fallback: use applied offset from info (spoof_offset_x/y/z are applied offset)
            dx = info.get('spoof_offset_x', 0.0)
            dy = info.get('spoof_offset_y', 0.0)
            dz = info.get('spoof_offset_z', 0.0)
            spoof_pos = radar_est + np.array([dx, dy, dz])

        data['drone_positions'].append(true_pos)
        data['spoof_positions'].append(spoof_pos)
        data['radar_estimates'].append(radar_est)
        data['nis_values'].append(info.get('drone_nis', 0))
        data['gamma_s_values'].append(info.get('gamma_s', 0))
        data['rewards'].append(reward)
        data['actions'].append(action)
        data['distances_to_fake'].append(info.get('dist_to_fake', 0))
        data['distances_to_true'].append(info.get('dist_to_true', 0))

        # === DEBUG: Collect z-axis bias investigation data ===
        data['spoof_offset_xyz'].append([
            info.get('spoof_offset_x', 0),
            info.get('spoof_offset_y', 0),
            info.get('spoof_offset_z', 0)
        ])
        data['M_radar_diag'].append([
            info.get('M_radar_xx', 0),
            info.get('M_radar_yy', 0),
            info.get('M_radar_zz', 0)
        ])
        data['reward_components'].append([
            info.get('r_x', 0),
            info.get('r_v', 0),
            info.get('r_gamma', 0)
        ])

        step += 1

    # numpy 배열로 변환
    for key in data:
        if key != 'start_position':  # start_position은 이미 numpy array
            data[key] = np.array(data[key])

    return data, step


def save_trajectory_data(data, env, filename='trajectory_data.txt'):
    """
    궤적 데이터를 텍스트 파일로 저장.

    저장 내용:
    - Episode 요약 (시작점, 목적지, 최종 위치)
    - 모든 스텝의 드론 위치
    - 모든 스텝의 기만 신호 위치
    - 모든 스텝의 액션
    - 모든 스텝의 보상
    """
    start_pos = data['start_position']  # 실제 시작 위치 (랜덤 노이즈 포함)
    drone_pos = data['drone_positions']
    spoof_pos = data['spoof_positions']
    radar_est = data['radar_estimates']
    actions = data['actions']
    rewards = data['rewards']

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SIE-SAC Trajectory Data\n")
        f.write("="*80 + "\n\n")

        # Episode 요약
        f.write("[Episode Summary]\n")
        f.write(f"Total steps: {len(drone_pos)}\n\n")

        # 시작 및 목적지 (실제 시작 위치 사용)
        f.write("[Positions]\n")
        f.write(f"Start position:        ( {start_pos[0]:8.2f}, {start_pos[1]:8.2f}, {start_pos[2]:8.2f} )\n")
        f.write(f"True destination:      ( {env.true_dest[0]:8.2f}, {env.true_dest[1]:8.2f}, {env.true_dest[2]:8.2f} )\n")
        f.write(f"Fake destination:      ( {env.fake_dest[0]:8.2f}, {env.fake_dest[1]:8.2f}, {env.fake_dest[2]:8.2f} )\n\n")

        # 최종 위치
        f.write("[Final Positions]\n")
        f.write(f"Final drone position:  ( {drone_pos[-1, 0]:8.2f}, {drone_pos[-1, 1]:8.2f}, {drone_pos[-1, 2]:8.2f} )\n")
        f.write(f"Final radar estimate:  ( {radar_est[-1, 0]:8.2f}, {radar_est[-1, 1]:8.2f}, {radar_est[-1, 2]:8.2f} )\n")
        f.write(f"Final spoof position:  ( {spoof_pos[-1, 0]:8.2f}, {spoof_pos[-1, 1]:8.2f}, {spoof_pos[-1, 2]:8.2f} )\n\n")

        # 최종 거리
        dist_to_true = np.linalg.norm(drone_pos[-1] - env.true_dest)
        dist_to_fake = np.linalg.norm(drone_pos[-1] - env.fake_dest)
        f.write(f"Distance to true dest: {dist_to_true:8.2f} m\n")
        f.write(f"Distance to fake dest: {dist_to_fake:8.2f} m\n\n")

        # 최종 액션
        f.write("[Final Action]\n")
        f.write(f"ρ (offset magnitude): {actions[-1, 0]:8.2f} m\n")
        f.write(f"θ (azimuth):          {np.degrees(actions[-1, 1]):8.2f}°\n")
        f.write(f"ψ (elevation):        {np.degrees(actions[-1, 2]):8.2f}°\n\n")

        # 전체 궤적 데이터
        f.write("="*80 + "\n")
        f.write("Full Trajectory Data\n")
        f.write("="*80 + "\n\n")

        f.write(f"{'Step':>5} | {'Drone Position (x, y, z)':>30} | {'Spoof Position (x, y, z)':>30} | {'ρ':>8} | {'θ(°)':>8} | {'ψ(°)':>8} | {'Reward':>10}\n")
        f.write("-"*80 + "\n")

        for i in range(len(drone_pos)):
            f.write(f"{i:5d} | "
                   f"({drone_pos[i, 0]:8.2f}, {drone_pos[i, 1]:8.2f}, {drone_pos[i, 2]:8.2f}) | "
                   f"({spoof_pos[i, 0]:8.2f}, {spoof_pos[i, 1]:8.2f}, {spoof_pos[i, 2]:8.2f}) | "
                   f"{actions[i, 0]:8.2f} | "
                   f"{np.degrees(actions[i, 1]):8.2f} | "
                   f"{np.degrees(actions[i, 2]):8.2f} | "
                   f"{rewards[i]:10.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Additional Debug Data\n")
        f.write("="*80 + "\n\n")

        # Spoof offset 상세 정보
        if 'spoof_offset_xyz' in data:
            spoof_offset_xyz = data['spoof_offset_xyz']
            f.write("[Spoofing Offset Statistics]\n")
            f.write(f"Mean offset: dx={spoof_offset_xyz[:, 0].mean():8.2f}, dy={spoof_offset_xyz[:, 1].mean():8.2f}, dz={spoof_offset_xyz[:, 2].mean():8.2f}\n")
            f.write(f"Std  offset: dx={spoof_offset_xyz[:, 0].std():8.2f}, dy={spoof_offset_xyz[:, 1].std():8.2f}, dz={spoof_offset_xyz[:, 2].std():8.2f}\n\n")

        # Reward 상세 정보
        if 'reward_components' in data:
            reward_comp = data['reward_components']
            f.write("[Reward Components Statistics]\n")
            f.write(f"Mean r_x:     {reward_comp[:, 0].mean():10.4f}\n")
            f.write(f"Mean r_v:     {reward_comp[:, 1].mean():10.4f}\n")
            f.write(f"Mean r_gamma: {reward_comp[:, 2].mean():10.4f}\n\n")

        f.write("="*80 + "\n")
        f.write("End of File\n")
        f.write("="*80 + "\n")

    print(f">>> 궤적 데이터 저장: {filename}")


def visualize_realtime(env, model, max_steps=1000, update_interval=5):
    """실시간 시각화 - 드론 실제 위치와 기만 신호 위치 모두 표시"""
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 10))

    obs, _ = env.reset()

    # 실제 시작 위치 저장
    if USE_PAPER_ENV:
        start_pos = env.true_pos.copy()
    elif hasattr(env, 'simulator'):
        start_pos = env.simulator.true_pos.copy()
    else:
        start_pos = np.array([0.0, 0.0, -20.0])

    # 데이터 기록
    drone_path = []
    spoof_path = []

    done = False
    step = 0
    total_reward = 0

    print(">>> 실시간 시뮬레이션 시작...")
    print(f"    시작 위치: {start_pos}")
    print(f"    실제 목적지: {env.true_dest}")
    print(f"    기만 목적지: {env.fake_dest}")

    while not done and step < max_steps:
        # 액션 예측
        if model is not None:
            action = predict_action(model, obs, deterministic=True)
        else:
            action = env.action_space.sample()

        # 환경 진행
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # 위치 기록 - info에서 가져오기 (auto-reset 전 데이터)
        if USE_PAPER_ENV:
            true_pos = info.get('true_pos', env.true_pos.copy())
            radar_est = info.get('radar_est', true_pos)
        elif hasattr(env, 'simulator'):
            true_pos = env.simulator.true_pos.copy()
            radar_est = true_pos
        else:
            true_pos = info.get('true_pos', np.zeros(3))
            radar_est = info.get('radar_est', true_pos)

        # 기만 위치 - ACTUAL position sent to UAV (1-step delay)
        # Use info['deceptive_pos'] which is what UAV actually received
        # NOT action-based calculation (that's for NEXT step!)
        if 'deceptive_pos' in info:
            spoof_pos = np.array(info['deceptive_pos'])
        else:
            # Fallback: use applied offset from info
            dx = info.get('spoof_offset_x', 0.0)
            dy = info.get('spoof_offset_y', 0.0)
            dz = info.get('spoof_offset_z', 0.0)
            spoof_pos = radar_est + np.array([dx, dy, dz])

        drone_path.append(true_pos)
        spoof_path.append(spoof_pos)

        # 주기적으로 화면 업데이트
        if step % update_interval == 0:
            ax.clear()

            # 목적지 표시 (실제 시작 위치 사용)
            ax.scatter(start_pos[0], start_pos[1], c='green', marker='s', s=150, label='Start', zorder=5)
            ax.scatter(env.true_dest[0], env.true_dest[1], c='blue', marker='*',
                      s=300, label='True Destination', zorder=5)
            ax.scatter(env.fake_dest[0], env.fake_dest[1], c='red', marker='X',
                      s=300, label='Fake Destination (Goal)', zorder=5)

            # 드론 실제 궤적 그리기
            path = np.array(drone_path)
            ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Drone Path (Real)', alpha=0.7)
            ax.scatter(path[-1, 0], path[-1, 1], c='blue', s=100, zorder=10, marker='o')  # 현재 위치

            # 기만 신호 궤적 그리기
            spoof = np.array(spoof_path)
            ax.plot(spoof[:, 0], spoof[:, 1], 'm--', linewidth=2, label='Spoofed Position', alpha=0.7)
            ax.scatter(spoof[-1, 0], spoof[-1, 1], c='magenta', s=100, zorder=10, marker='D')  # 현재 기만 위치

            # 현재 기만 오프셋 표시 (드론 위치에서 기만 위치로 화살표)
            ax.annotate('', xy=(spoof[-1, 0], spoof[-1, 1]), xytext=(path[-1, 0], path[-1, 1]),
                       arrowprops=dict(arrowstyle='->', color='orange', lw=2))

            # 정보 표시
            gamma_s = info.get('gamma_s', 0)
            dist_fake = info.get('dist_to_fake', 0)
            dist_true = info.get('dist_to_true', 0)
            spoof_dist = np.linalg.norm(spoof_pos[:2] - true_pos[:2])  # 기만 오프셋 거리

            # Get APPLIED offset (actually used this step, not next action)
            applied_x = info.get('spoof_offset_x', 0.0)
            applied_y = info.get('spoof_offset_y', 0.0)
            applied_z = info.get('spoof_offset_z', 0.0)
            applied_mag = np.sqrt(applied_x**2 + applied_y**2 + applied_z**2)

            title = (f"Step: {step} | Reward: {reward:.1f} | Total: {total_reward:.1f}\n"
                    f"γ^s: {gamma_s:.2f} | Dist to Fake: {dist_fake:.1f} | Dist to True: {dist_true:.1f}\n"
                    f"Spoof Offset: {spoof_dist:.1f}m | Applied |Δx^s|: {applied_mag:.1f}m")
            ax.set_title(title, fontsize=11)

            # 축 설정 - 기만 위치도 포함
            all_x = [0, env.true_dest[0], env.fake_dest[0]] + [p[0] for p in drone_path] + [p[0] for p in spoof_path]
            all_y = [0, env.true_dest[1], env.fake_dest[1]] + [p[1] for p in drone_path] + [p[1] for p in spoof_path]
            margin = 100
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_aspect('equal')

            plt.draw()
            plt.pause(0.01)

        step += 1

    plt.ioff()
    
    # 결과 출력
    final_dist_fake = info.get('dist_to_fake', 0)
    final_dist_true = info.get('dist_to_true', 0)
    
    print(f"\n>>> 에피소드 종료!")
    print(f"    총 스텝: {step}")
    print(f"    총 보상: {total_reward:.2f}")
    print(f"    기만 목적지까지 거리: {final_dist_fake:.1f}m")
    print(f"    실제 목적지까지 거리: {final_dist_true:.1f}m")
    
    if final_dist_fake < 20:
        print("    ✓ 성공! 드론이 기만 목적지 근처에 도달했습니다.")
    elif final_dist_true < 20:
        print("    ✗ 실패! 드론이 실제 목적지에 도달했습니다.")
    else:
        print("    - 아직 목적지에 도달하지 못했습니다.")
    
    return np.array(drone_path), np.array(spoof_path)


def plot_analysis(data, env):
    """분석 그래프 그리기 - 드론 실제 위치와 기만 신호 위치 모두 표시"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    steps = np.arange(len(data['rewards']))
    drone_pos = data['drone_positions']
    spoof_pos = data['spoof_positions']

    # 1. 2D 궤적 (드론 + 기만 신호)
    ax1 = axes[0, 0]

    # 드론 실제 궤적
    ax1.plot(drone_pos[:, 0], drone_pos[:, 1], 'b-', linewidth=2, label='Drone Path (Real)')
    # 기만 신호 궤적
    ax1.plot(spoof_pos[:, 0], spoof_pos[:, 1], 'm--', linewidth=1.5, label='Spoofed Position', alpha=0.7)

    # 시작점, 목적지
    ax1.scatter(0, 0, c='green', marker='s', s=150, label='Start', zorder=5)
    ax1.scatter(env.true_dest[0], env.true_dest[1], c='blue', marker='*', s=200, label='True Dest')
    ax1.scatter(env.fake_dest[0], env.fake_dest[1], c='red', marker='X', s=200, label='Fake Dest')

    # 최종 위치
    ax1.scatter(drone_pos[-1, 0], drone_pos[-1, 1], c='blue', s=150, label='Final Drone', zorder=10, marker='o')
    ax1.scatter(spoof_pos[-1, 0], spoof_pos[-1, 1], c='magenta', s=150, label='Final Spoof', zorder=10, marker='D')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('2D Trajectory (Real vs Spoofed)')
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 2. NIS / γ^s 변화
    ax2 = axes[0, 1]
    ax2.plot(steps, data['gamma_s_values'], 'r-', label='γ^s (Predicted NIS)', alpha=0.8)
    if 'nis_values' in data and len(data['nis_values']) > 0:
        ax2.plot(steps, data['nis_values'], 'b-', label='Drone NIS', alpha=0.5)
    ax2.axhline(y=7.815, color='k', linestyle='--', linewidth=2, label='Threshold (χ²=7.815)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('NIS Value')
    ax2.set_title('NIS History (Concealment)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 기만 오프셋 분석 (NEW)
    ax3 = axes[0, 2]
    spoof_offset_dist = np.linalg.norm(spoof_pos - drone_pos, axis=1)
    ax3.plot(steps[:600], spoof_offset_dist[:600], 'orange', linewidth=2, label='Spoof Offset Distance')
    ax3.axhline(y=50, color='r', linestyle='--', label='Max Offset (ρ_s_max=200m)')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Distance (m)')
    ax3.set_title('Spoofing Offset Distance (|x^s - x|)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 보상 변화
    ax4 = axes[1, 0]
    ax4.plot(steps, data['rewards'], 'g-', alpha=0.7)
    window = min(50, len(data['rewards']) // 5) if len(data['rewards']) > 10 else 1
    if window > 1:
        ax4.plot(steps, np.convolve(data['rewards'], np.ones(window)/window, mode='same'),
                 'r-', linewidth=2, label=f'Moving Avg ({window})')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Reward')
    ax4.set_title('Reward History')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 목적지까지 거리
    ax5 = axes[1, 1]
    ax5.plot(steps, data['distances_to_fake'], 'r-', label='To Fake Dest', linewidth=2)
    ax5.plot(steps, data['distances_to_true'], 'b-', label='To True Dest', linewidth=2)
    ax5.axhline(y=10, color='g', linestyle='--', label='Success Threshold')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Distance (m)')
    ax5.set_title('Distance to Destinations')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 액션 분석 (NEW)
    ax6 = axes[1, 2]
    actions = data['actions']
    ax6.plot(steps, actions[:, 0], 'r-', label='ρ (offset distance)', alpha=0.8)
    ax6.plot(steps, np.degrees(actions[:, 1]), 'g-', label='θ (azimuth, deg)', alpha=0.8)
    ax6.plot(steps, np.degrees(actions[:, 2]), 'b-', label='ψ (elevation, deg)', alpha=0.8)
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Action Value')
    ax6.set_title('Action History (ρ, θ, ψ)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('analysis_result.png', dpi=150)
    print(">>> 분석 그래프 저장: analysis_result.png")
    plt.show()


def plot_debug_analysis(data):
    """
    Z축 편향 디버깅 플롯 - 사용자 분석 가설 검증용

    검증 항목:
    0. ρ 분포 - ρ→0 붕괴 확인 (가장 중요!)
    1. M_radar 대각 성분 (xx, yy, zz) - zz >> yy이면 z축 선호 원인
    2. Spoofing offset xyz 성분 - z축 편향 확인
    3. θ/ψ 분포 - θ가 ±90°로 가는지 (dy 생성 의지 확인)
    4. Reward 성분 비교 - r_gamma가 너무 큰지 확인
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    steps = np.arange(len(data['rewards']))
    actions = data['actions']
    spoof_offset_xyz = data['spoof_offset_xyz']
    M_radar_diag = data['M_radar_diag']
    reward_components = data['reward_components']

    # 0. ρ 분포 - ρ→0 붕괴 확인 (핵심!)
    ax0 = axes[0, 0]
    rho_values = actions[:, 0]
    ax0.plot(steps, rho_values, 'purple', linewidth=2, alpha=0.7)
    ax0.axhline(y=0, color='r', linestyle='--', linewidth=2, label='ρ=0 (붕괴)')
    ax0.axhline(y=200, color='g', linestyle='--', linewidth=1.5, label='ρ_max=200m')
    ax0.set_xlabel('Step')
    ax0.set_ylabel('ρ (offset magnitude, m)')
    ax0.set_title('ρ Distribution Over Time\n[ρ→0이면 학습 붕괴! 방향 무의미]')
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    # 1. M_radar 대각 성분 (Σ_r 비등방성 확인)
    ax1 = axes[0, 1]
    ax1.plot(steps, M_radar_diag[:, 0], 'r-', label='M_radar_xx', linewidth=2, alpha=0.8)
    ax1.plot(steps, M_radar_diag[:, 1], 'g-', label='M_radar_yy', linewidth=2, alpha=0.8)
    ax1.plot(steps, M_radar_diag[:, 2], 'b-', label='M_radar_zz', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Variance (m²)')
    ax1.set_title('M_radar Diagonal (Radar State Covariance)\n[원인3 검증: zz >> yy이면 z축 선호]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Spoofing offset xyz 성분 (실제 편향 확인)
    ax2 = axes[0, 2]
    ax2.plot(steps, spoof_offset_xyz[:, 0], 'r-', label='dx', linewidth=2, alpha=0.7)
    ax2.plot(steps, spoof_offset_xyz[:, 1], 'g-', label='dy (목표: +100m)', linewidth=2, alpha=0.7)
    ax2.plot(steps, spoof_offset_xyz[:, 2], 'b-', label='dz (목표: 0m)', linewidth=2, alpha=0.7)
    ax2.axhline(y=100, color='g', linestyle='--', linewidth=1.5, label='Target dy=+100m')
    ax2.axhline(y=0, color='b', linestyle='--', linewidth=1.5, label='Target dz=0m')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Offset (m)')
    ax2.set_title('Spoofing Offset Components (Cartesian)\n[평균 0이면 평균이 아닌 분산/절댓값 확인 필요]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. θ (azimuth) 분포 - Y축 오프셋 생성 의지 확인
    ax3 = axes[1, 0]
    theta_deg = np.degrees(actions[:, 1])
    ax3.plot(steps, theta_deg, 'g-', linewidth=2, alpha=0.7)
    ax3.axhline(y=90, color='r', linestyle='--', linewidth=2, label='θ=+90° (dy>0)')
    ax3.axhline(y=-90, color='r', linestyle='--', linewidth=2, label='θ=-90° (dy<0)')
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, label='θ=0° (dx>0)')
    ax3.axhline(y=180, color='k', linestyle='--', linewidth=1, label='θ=±180° (dx<0)')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Azimuth θ (degrees)')
    ax3.set_title('Azimuth Angle θ Distribution\n[ρ=0이면 θ는 무의미. ρ>0일 때만 의미 있음]')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([-180, 180])

    # 4. ψ (elevation) 분포 - Gradient 소실 확인
    ax4 = axes[1, 1]
    psi_deg = np.degrees(actions[:, 2])
    ax4.plot(steps, psi_deg, 'b-', linewidth=2, alpha=0.7)
    ax4.axhline(y=0, color='g', linestyle='--', linewidth=2, label='ψ=0° (cos(ψ)=1, xy 민감)')
    ax4.axhline(y=90, color='r', linestyle='--', linewidth=2, label='ψ=+90° (dz max)')
    ax4.axhline(y=-90, color='r', linestyle='--', linewidth=2, label='ψ=-90° (dz min)')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Elevation ψ (degrees)')
    ax4.set_title('Elevation Angle ψ Distribution\n[원인1 검증: ψ→±90°이면 xy gradient 소실]')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([-90, 90])

    # 5. Reward 성분 비교 (r_x, r_v, r_gamma)
    ax5 = axes[1, 2]
    r_x = reward_components[:, 0] * data.get('alpha_1', 1.0)  # Scaled by alpha
    r_v = reward_components[:, 1] * data.get('alpha_2', 1.0)
    r_gamma = reward_components[:, 2] * data.get('alpha_3', 1.0)

    ax5.plot(steps, r_x, 'r-', label='α₁·r_x (position)', linewidth=2, alpha=0.7)
    ax5.plot(steps, r_v, 'g-', label='α₂·r_v (velocity)', linewidth=2, alpha=0.7)
    ax5.plot(steps, r_gamma, 'b-', label='α₃·r_γ (concealment)', linewidth=2, alpha=0.7)
    ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Reward Component')
    ax5.set_title('Reward Components Comparison\n[r_gamma >> r_x+r_v이면 concealment 지배]')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. M_radar 대각 성분 비율 (zz/yy)
    ax6 = axes[2, 0]
    variance_ratio_zy = M_radar_diag[:, 2] / (M_radar_diag[:, 1] + 1e-9)
    variance_ratio_zx = M_radar_diag[:, 2] / (M_radar_diag[:, 0] + 1e-9)
    ax6.plot(steps, variance_ratio_zy, 'b-', label='zz/yy ratio', linewidth=2, alpha=0.8)
    ax6.plot(steps, variance_ratio_zx, 'r-', label='zz/xx ratio', linewidth=2, alpha=0.8)
    ax6.axhline(y=1.0, color='k', linestyle='--', linewidth=2, label='Isotropic (ratio=1)')
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Variance Ratio')
    ax6.set_title('M_radar Anisotropy (zz/yy, zz/xx)\n[비율 >> 1이면 z축이 γ^s 관점에서 싸다]')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')

    # 7. Offset 절댓값 분포 (평균 상쇄 문제 해결)
    ax7 = axes[2, 1]
    abs_dx = np.abs(spoof_offset_xyz[:, 0])
    abs_dy = np.abs(spoof_offset_xyz[:, 1])
    abs_dz = np.abs(spoof_offset_xyz[:, 2])
    ax7.plot(steps, abs_dx, 'r-', label='|dx|', linewidth=2, alpha=0.7)
    ax7.plot(steps, abs_dy, 'g-', label='|dy| (목표: 100m)', linewidth=2, alpha=0.7)
    ax7.plot(steps, abs_dz, 'b-', label='|dz| (목표: 0m)', linewidth=2, alpha=0.7)
    ax7.axhline(y=100, color='g', linestyle='--', linewidth=1.5, label='Target |dy|=100m')
    ax7.set_xlabel('Step')
    ax7.set_ylabel('Absolute Offset (m)')
    ax7.set_title('Absolute Offset Magnitude\n[평균 0이어도 |dz| > |dy|이면 z 편향 확인]')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. dy vs dz Scatter plot
    ax8 = axes[2, 2]
    ax8.scatter(spoof_offset_xyz[:, 1], spoof_offset_xyz[:, 2], c=steps, cmap='viridis',
                s=10, alpha=0.6, edgecolors='none')
    ax8.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax8.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax8.axvline(x=100, color='g', linestyle='--', linewidth=2, label='Target dy=+100m')
    ax8.set_xlabel('dy (m)')
    ax8.set_ylabel('dz (m)')
    ax8.set_title('dy vs dz Scatter\n[점들이 dy축에 모여야 함]')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig('debug_z_bias_analysis.png', dpi=150)
    print(">>> Z축 편향 디버깅 그래프 저장: debug_z_bias_analysis.png")

    # 통계 요약 출력
    print("\n" + "="*70)
    print("Z축 편향 디버깅 통계 요약")
    print("="*70)

    print(f"\n[0] ρ 분포 (핵심!)")
    rho_values = actions[:, 0]
    print(f"    ρ (평균): {rho_values.mean():.2f} m")
    print(f"    ρ (중앙값): {np.median(rho_values):.2f} m")
    print(f"    ρ (표준편차): {rho_values.std():.2f} m")
    print(f"    ρ (최솟값): {rho_values.min():.2f} m")
    print(f"    ρ (최댓값): {rho_values.max():.2f} m")
    print(f"    ρ < 10m 비율: {(rho_values < 10).mean() * 100:.1f}%")
    print(f"    ρ < 1m 비율: {(rho_values < 1).mean() * 100:.1f}%")
    if rho_values.mean() < 10:
        print(f"    ⚠️  경고: ρ 평균이 10m 미만! → ρ=0 붕괴 의심!")

    print(f"\n[1] M_radar 대각 성분 평균:")
    print(f"    M_radar_xx (평균): {M_radar_diag[:, 0].mean():.4f} m²")
    print(f"    M_radar_yy (평균): {M_radar_diag[:, 1].mean():.4f} m²")
    print(f"    M_radar_zz (평균): {M_radar_diag[:, 2].mean():.4f} m²")
    print(f"    → zz/yy 비율: {M_radar_diag[:, 2].mean() / (M_radar_diag[:, 1].mean() + 1e-9):.2f}x")

    print(f"\n[2] Spoofing offset 평균 (목표: dy=+100m, dz=0m):")
    print(f"    dx (평균): {spoof_offset_xyz[:, 0].mean():+.2f} m")
    print(f"    dy (평균): {spoof_offset_xyz[:, 1].mean():+.2f} m  [목표: +100m]")
    print(f"    dz (평균): {spoof_offset_xyz[:, 2].mean():+.2f} m  [목표: 0m]")

    print(f"\n[2-1] Spoofing offset 절댓값 평균 (평균 상쇄 문제 해결):")
    abs_dx = np.abs(spoof_offset_xyz[:, 0])
    abs_dy = np.abs(spoof_offset_xyz[:, 1])
    abs_dz = np.abs(spoof_offset_xyz[:, 2])
    print(f"    |dx| (평균): {abs_dx.mean():.2f} m")
    print(f"    |dy| (평균): {abs_dy.mean():.2f} m  [목표: 100m]")
    print(f"    |dz| (평균): {abs_dz.mean():.2f} m  [목표: 0m]")
    print(f"    → |dy|/|dz| 비율: {abs_dy.mean() / (abs_dz.mean() + 1e-9):.2f}")
    if abs_dz.mean() > abs_dy.mean():
        print(f"    ⚠️  경고: |dz| > |dy|! → Z축 편향 확인!")

    print(f"\n[3] 각도 분포 (목표: θ≈±90°, ψ≈0°):")
    print(f"    θ (평균): {theta_deg.mean():+.1f}° (목표: ±90° for dy)")
    print(f"    ψ (평균): {psi_deg.mean():+.1f}° (목표: 0° for xy control)")
    print(f"    |ψ| > 30° 비율: {(np.abs(psi_deg) > 30).mean() * 100:.1f}%")

    print(f"\n[4] Reward 성분 평균:")
    print(f"    r_x (평균): {r_x.mean():+.4f}")
    print(f"    r_v (평균): {r_v.mean():+.4f}")
    print(f"    r_gamma (평균): {r_gamma.mean():+.4f}")
    print(f"    → |r_gamma| / (|r_x| + |r_v|): {abs(r_gamma.mean()) / (abs(r_x.mean()) + abs(r_v.mean()) + 1e-9):.2f}")

    print("\n" + "="*70)
    print("진단 결과:")
    print("="*70)

    # 가장 중요: ρ=0 붕괴 체크
    if rho_values.mean() < 10:
        print("🚨 [핵심 문제] ρ=0 붕괴 발생!")
        print("    → ρ 평균이 10m 미만입니다.")
        print("    → r_gamma가 너무 커서 '오프셋을 안 주는 게' 최적 전략이 됨")
        print("    → θ, ψ는 ρ=0이면 무의미. 방향 학습 자체가 죽음")
        print("    → 해결: r_gamma를 패널티로 바꾸거나 alpha_3 줄이기")
        print("")

    if abs(r_gamma.mean()) > abs(r_x.mean()) + abs(r_v.mean()):
        print("⚠️  [원인2/4] r_gamma가 r_x + r_v보다 큽니다!")
        print("    → Concealment 보상이 position/velocity 보상을 지배")
        print("    → 학습이 '은닉'에만 집중, '목적지 유도'는 무시")
        print("    → 해결: alpha_3 줄이기 (예: 0.3 → 0.1)")
        print("")

    if M_radar_diag[:, 2].mean() > 2 * M_radar_diag[:, 1].mean():
        print("⚠️  [원인3] M_radar의 zz 분산이 yy보다 2배 이상 큽니다!")
        print("    → z 오프셋이 γ^s 관점에서 '싸다' (덜 들킴)")
        print("")

    if abs(psi_deg.mean()) > 20:
        print("⚠️  [원인1] ψ(elevation)이 0°에서 멀리 떨어져 있습니다!")
        print("    → xy 평면 gradient 소실 가능")
        print("")

    if abs_dz.mean() > abs_dy.mean():
        print("⚠️  Z축 편향 확인: |dz| > |dy| (목표는 dy=+100m)")
        print("")

    print("="*70 + "\n")

    plt.show()


def plot_3d_trajectory(data, env):
    """3D 궤적 시각화 - 드론 실제 위치와 기만 신호 위치 모두 표시"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    start_pos = data['start_position']  # 실제 시작 위치 (랜덤 노이즈 포함)
    drone_pos = data['drone_positions']
    spoof_pos = data['spoof_positions']

    # 드론 실제 궤적
    ax.plot(drone_pos[:, 0], drone_pos[:, 1], drone_pos[:, 2],
            'b-', linewidth=2, label='Drone Path (Real)')

    # 기만 신호 궤적
    ax.plot(spoof_pos[:, 0], spoof_pos[:, 1], spoof_pos[:, 2],
            'm--', linewidth=1.5, label='Spoofed Position', alpha=0.7)

    # 시작점 (실제 시작 위치 사용)
    ax.scatter(start_pos[0], start_pos[1], start_pos[2],
               c='green', marker='s', s=150, label='Start')

    # 목적지들
    ax.scatter(env.true_dest[0], env.true_dest[1], env.true_dest[2],
               c='blue', marker='*', s=200, label='True Dest')
    ax.scatter(env.fake_dest[0], env.fake_dest[1], env.fake_dest[2],
               c='red', marker='X', s=200, label='Fake Dest')

    # 최종 위치
    ax.scatter(drone_pos[-1, 0], drone_pos[-1, 1], drone_pos[-1, 2],
               c='blue', s=150, label='Final Drone Pos', marker='o')
    ax.scatter(spoof_pos[-1, 0], spoof_pos[-1, 1], spoof_pos[-1, 2],
               c='magenta', s=150, label='Final Spoof Pos', marker='D')

    # 일부 스텝에서 드론→기만 오프셋 화살표 표시 (매 100스텝마다)
    for i in range(0, len(drone_pos), max(1, len(drone_pos) // 10)):
        ax.plot([drone_pos[i, 0], spoof_pos[i, 0]],
                [drone_pos[i, 1], spoof_pos[i, 1]],
                [drone_pos[i, 2], spoof_pos[i, 2]],
                'orange', linewidth=1, alpha=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory (Real vs Spoofed)')
    ax.legend()

    plt.savefig('trajectory_3d.png', dpi=150)
    print(">>> 3D 궤적 저장: trajectory_3d.png")
    plt.show()


def select_entropy_type_for_viz():
    """사용자에게 entropy type을 선택하도록 요청 (시각화용)"""
    print("\n사용할 모델의 Entropy 유형을 선택하세요:")
    print("  1. SIE (Spatial Information Entropy) - 논문 방식")
    print("  2. Action Entropy - 표준 SAC 방식")
    print()

    while True:
        try:
            choice = input("선택 (1 또는 2): ").strip()
            if choice == '1':
                print("→ SIE 모델 선택됨")
                return 'sie'
            elif choice == '2':
                print("→ Action Entropy 모델 선택됨")
                return 'action'
            else:
                print("잘못된 입력입니다. 1 또는 2를 입력하세요.")
        except KeyboardInterrupt:
            print("\n취소되었습니다.")
            exit(0)


def main():
    print("=" * 60)
    print("SIE-SAC 학습 결과 시각화")
    print("=" * 60)

    # Use configuration from SIE_SAC_paper.py (SINGLE SOURCE OF TRUTH)
    env_config = DEFAULT_ENV_CONFIG.copy()

    if USE_PAPER_ENV:
        print(">>> VectorizedSIEEnvPaper 래퍼 사용")
        env = SpoofingEnv(config=env_config)
    else:
        env = SpoofingEnv(config=env_config)

    # 사용자가 entropy type 선택
    entropy_type = select_entropy_type_for_viz()

    # 모델 경로 설정
    if USE_SB3:
        model_paths = [
            "models/SB3_SIE_SAC/sac_final.zip",
            "models/SB3_SIE_SAC/best/best_model.zip",
            "models/Pretrained/sac_pretrained_100k.zip",
        ]
    else:
        # entropy_type에 따라 모델 경로 설정
        model_paths = [
            f"models/SIE_SAC_paper_{entropy_type}/sie_sac_paper_final.pt",
            # f"models/SIE_SAC_paper_{entropy_type}/sie_sac_paper_900000.pt",
            # f"models/SIE_SAC_paper_{entropy_type}/sie_sac_paper_800000.pt",
            # f"models/SIE_SAC_paper_{entropy_type}/sie_sac_paper_700000.pt",
            # f"models/SIE_SAC_paper_{entropy_type}/sie_sac_paper_600000.pt",
            # f"models/SIE_SAC_paper_{entropy_type}/sie_sac_paper_500000.pt",
            # f"models/SIE_SAC_paper_{entropy_type}/sie_sac_paper_400000.pt",
            # f"models/SIE_SAC_paper_{entropy_type}/sie_sac_paper_300000.pt",
            # f"models/SIE_SAC_paper_{entropy_type}/sie_sac_paper_200000.pt",
            # f"models/SIE_SAC_paper_{entropy_type}/sie_sac_paper_100000.pt",
        ]

    print(f"\n>>> 모델 검색 경로: models/SIE_SAC_paper_{entropy_type}/")

    # 모델 로드 시도
    model = None
    for path in model_paths:
        if os.path.exists(path):
            print(f">>> 모델 파일 발견: {path}")
            model = load_model(path, env, env_config, entropy_type=entropy_type)
            if model is not None:
                break
    
    if model is None:
        print("\n!!! 학습된 모델이 없습니다.")
        print("    랜덤 에이전트로 시각화를 진행합니다.")
        print("    학습을 먼저 진행하려면:")
        print("      - SB3: python sie_sac_sb3.py")
        print("      - Custom: python sie_sac_train.py")
    
    # 사용자 선택
    print("\n>>> 시각화 모드 선택:")
    print("    1. 실시간 시뮬레이션")
    print("    2. 분석 그래프만 보기")
    print("    3. 둘 다")
    
    try:
        choice = input("선택 (1/2/3, 기본=3): ").strip()
        if choice == '':
            choice = '3'
    except:
        choice = '3'
    
    if choice in ['1', '3']:
        print("\n>>> 실시간 시뮬레이션 시작...")
        drone_path, spoof_path = visualize_realtime(env, model, max_steps=10000)
    
    if choice in ['2', '3']:
        print("\n>>> 분석용 에피소드 실행...")
        data, steps = run_episode(env, model, max_steps=10000)
        
        print(f"    총 {steps} 스텝 완료")

        # 분석 그래프
        plot_analysis(data, env)

        # 3D 궤적
        plot_3d_trajectory(data, env)

        # Z축 편향 디버깅 분석
        print("\n>>> Z축 편향 디버깅 분석 시작...")
        plot_debug_analysis(data)

        # 궤적 데이터를 텍스트 파일로 저장
        print("\n>>> 궤적 데이터 저장 중...")
        save_trajectory_data(data, env, filename='trajectory_data.txt')

    print("\n>>> 시각화 완료!")


if __name__ == '__main__':
    main()
