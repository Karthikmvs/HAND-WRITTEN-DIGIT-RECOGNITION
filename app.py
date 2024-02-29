import pygame , sys
from pygame.locals import *
import numpy as np
from keras.models import load_model 
import cv2


WINDOWSIZEX = 640
WINDOWSIZEY = 480

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0 , 0)

IMAGESAVE = False

BOUNDRY = 5
# load model
MODEL = load_model("best_model.h5")

LABELS ={0: "ZERO", 1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE" , 6: "SIX", 7: "SEVEN", 8: "EIGHT", 9: "NINE"}

# initialize pygame
pygame.init()

FONT = pygame.font.SysFont("Arial", 20)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY)) # set up window size

pygame.display.set_caption("KARTHIK'S HANDWRITTEN DIGIT RECOGNIZER ") # set up window title

iswriting = False

number_xcord = []
number_ycord = []
image_cnt = 1

PREDICT = True
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4,0)
            
            number_xcord.append(xcord)
            number_ycord.append(ycord)


        if event.type == MOUSEBUTTONDOWN:
            iswriting = True
        
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x , rect_max_x = max(number_xcord[0] - BOUNDRY, 0), min(number_xcord[-1] + BOUNDRY, WINDOWSIZEX)
            rect_min_y , rect_max_y = max(number_ycord[0] - BOUNDRY, 0), min(number_ycord[-1] + BOUNDRY, WINDOWSIZEY)
            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                img_cnt += 1
            
            if PREDICT:

                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10,10), "constant", constant_values=0)
                image = cv2.resize(image, (28, 28))/255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])
                textSurface = FONT.render(label, True, RED, WHITE)
                textRecObj = textSurface.get_rect()
                textRecObj.left , textRecObj.bottom = rect_min_x, rect_min_y

                DISPLAYSURF.blit(textSurface, textRecObj)

            if event.type == KEYDOWN :
               if event.unicode == "n":
                    DISPLAYSURF.fill(BLACK)
        

        pygame.display.update()
    


        


