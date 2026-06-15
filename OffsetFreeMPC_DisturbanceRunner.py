import sys

from utils.online_disturbance_runner import main_offset_free_mpc_disturbance


EPISODES = 5


if __name__ == "__main__":
    argv = sys.argv[1:] or ["--episodes", str(EPISODES)]
    main_offset_free_mpc_disturbance(argv)
