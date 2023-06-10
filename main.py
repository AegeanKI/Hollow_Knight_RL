import hkenv
import time

if __name__ == "__main__":
    game = hkenv.HKEnv()

    game.test()

    while True:
        time.sleep(0.1)
        enemy_remain, enemy_full, character_remain, character_full = game.observe()
        print(f"enemy = {enemy_remain} / {enemy_full}, ", end='')
        print(f"character = {character_remain} / {character_full}")
