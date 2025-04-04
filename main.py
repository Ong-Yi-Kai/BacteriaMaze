from Game import Game
import pygame


if __name__ == "__main__":
    game = Game(num_cells=5, food_pos=[(25,25), (25,75), (75, 25), (75,75)],
                dimx=100, dimy=100, cellsize=2)

    paused = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            game.update()
