from cv2 import aruco

#### SPECS TAKEN FROM GIZEM ####
# checkerboard / charuco / aruco§
board_type = "charuco"
# width and height of grid
board_size = [7, 6]
# number of bits in the markers, if aruco/charuco
board_marker_bits = 4
# number of markers in dictionary, if aruco/charuco
board_marker_dict_number = 250 #(```aruco.DICT_4X4_250```)
# length of marker side
board_marker_length = 0.225 # mm
# If charuco or checkerboard, square side length
board_square_side_length = 0.300 # mm

# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = board_size[1]
CHARUCOBOARD_COLCOUNT = board_size[0]
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

# Create constants to be passed into OpenCV and Aruco methods
CHARUCO_BOARD = aruco.CharucoBoard(
        (CHARUCOBOARD_COLCOUNT,CHARUCOBOARD_ROWCOUNT),
        squareLength=board_square_side_length*1e-3,
        markerLength=board_marker_length*1e-3,
        dictionary=ARUCO_DICT)

CHARUCO_BOARD.setLegacyPattern(True)