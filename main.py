import cv2
import numpy as np 
from plate_detection import plate
from character_detection import character

#img = cv2.imread("file/character1.jpg")
img = cv2.imread("b.jpg")


identifyPlates = plate(img, type_of_plate = 'long_plate')
morp = identifyPlates.process(img)
contour = identifyPlates.extract_contours(morp)
clean_plate = identifyPlates.CleanAndRead(contour)

identifyCharacter = character(clean_plate, type_of_plate='long_plate')
thresh = identifyCharacter.find_character(clean_plate)
character = identifyCharacter.extract_contours_character(thresh)
convex = identifyCharacter.convex_hull(thresh, character)
final = identifyCharacter.show_contour(convex)
identifyCharacter.Read(final)
identifyCharacter.filter_img(final)

# identifyCharacter = character(img, type_of_plate='long_plate')
# thresh = identifyCharacter.find_character(img)
# character = identifyCharacter.extract_contours_character(thresh)
# convex = identifyCharacter.convex_hull(thresh, character)
# final = identifyCharacter.show_contour(convex)
# identifyCharacter.Read(final)


