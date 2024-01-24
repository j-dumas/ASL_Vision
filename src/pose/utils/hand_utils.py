from src.pose.utils.math_utils import dist, round, keep
from src.pose.utils.list_utils import is_empty
from numba import njit

FINGER_TIPS = [4, 8, 12, 16, 20]

def inbound(id: int) -> bool:
    '''Checks if the id is between {0 and FINGER_TIPS[-1]}'''
    return 0 <= id <= FINGER_TIPS[-1]

def is_finger_tip(finger_id: int) -> bool:
    '''Tells if the {finger_id} is a finger\'s tip'''
    return finger_id in FINGER_TIPS

@njit()
def is_hand_up(all_positions: list, min_points_down = 2) -> bool:
    '''Return True if more than { min_points_down } finger's point were considered down.'''
    reference_y_balance = all_positions[0].y
    counter = 0
    for id, position in enumerate(all_positions):
        if position.y < reference_y_balance:
            counter = counter + 1
            if counter >= min_points_down:
                return True
    return False

@njit()
def is_finger_up(finger: object = None, threshold: float = -0.9) -> bool:
    '''
    Return True if the distance is below {threshold}
    @TODO: if y is almost the same, check for x instead. (waiting for hand rotations)
    '''
    if finger != None:
        min_x, min_y = finger.get_finger_at(0)
        top_x, top_y = finger.get_last_joint()
        diff = (top_y - min_y) * 100
        return diff <= threshold
    return False

@njit()
def calc_distance_between_two_fingers(fingers_location: list = []) -> float:
    '''The {fingers} should contain two finger in a list'''
    if len(fingers_location) == 2:
        x1, y1 = fingers_location[0]
        x2, y2 = fingers_location[1]
        return dist([x1, y1], [x2, y2])
    return 0


def get_finger_trail(finger_tip: int, is_thumb: bool = False) -> list:
    '''Return a list of all the points connected to a finger tip'''
    if is_finger_tip(finger_id=finger_tip):
        return [finger_tip - x for x in range([4, 2][is_thumb])] # 4 points connected to a finger

@njit()
def are_finger_touching(fingers_location: list = [], threshold: float = 0.06) -> bool:
    '''Return True if the two fingers are touching (depending on the threshold)'''
    return calc_distance_between_two_fingers(fingers_location) <= threshold

def update_fingers(list, fingers: list = []):
    '''Update all the fingers with the current joints list'''
    if not is_empty(fingers):
        for finger in fingers:
            finger.update(list)

@njit()
def is_not_real_hand(all_positions: list) -> bool:
    '''Tells if a displayed hand is not a user's hand'''
    l = []
    for i in range(21):
        x, y = all_positions[i].x, all_positions[i].y
        x1, y1 = all_positions[13 + i].x, all_positions[13 + i].y 
        a = dist([x, y], [x1, y1])
        if a < 0.09:
            l.append(a)
    if len(l) >= 13:
        # print(f'Anomaly spotted!')
        return True
    print(f'{len(l)} points too close!')
    return False

def is_hand_in_screen(all_positions: list, hands_showed: int) -> bool:
    '''
    Tells if the hand is in the screen.
    '''
    if hands_showed == 1:
        for p in all_positions:
            x, y = p.x, p.y
            if (y >= 1 or y <= 0) or (x >= .98 or x <= 0):
                return False
    return True

def get_finger_stage(finger: object, ground: list = [], is_thumb: bool = False, is_pinky: bool = False, is_right_hand: bool = True, hand=None) -> float:
    '''Return a value between { 0 and 1 } depending on how closed the finger is'''
    ground_calc = (dist(ground[0], ground[1]) * finger.get_buffer())
    if ground_calc == 0 or hand == None:
        return -1

    distance = get_calc_from_hand_rotation(hand, finger, ground[1], is_thumb=is_thumb)
    return keep(round(distance / ground_calc, 4), 0, 1)

def get_calc_from_hand_rotation(hand, finger, thumb_ground, is_thumb=False):
    top, bottom = (finger.get_joint_at(0), finger.get_last_joint())
    top_x, top_y = top
    bottom_x, bottom_y = bottom
    thumb_ground_x, thumb_ground_y = thumb_ground
    rotation = hand.get_rotation()
    if rotation == 0:
        if is_thumb:
            return keep([thumb_ground_x - top_x, top_x - thumb_ground_x][hand.get_type() == 1], 0, 100)
        return keep(bottom_y - top_y, 0, 100)
    elif rotation == 1:
        if is_thumb:
            return keep(thumb_ground_y - top_y, 0, 100)
        return keep([top_x - bottom_x, bottom_x - top_x][hand.get_type() == 1], 0, 100)
    elif rotation == 2:
        if is_thumb:
            return keep([thumb_ground_x - top_x, top_x - thumb_ground_x][hand.get_type() == 1], 0, 100)
        return keep(top_y - bottom_y, 0, 100)
    elif rotation == 3:
        if is_thumb:
            return keep(thumb_ground_y - top_y, 0, 100)
        return keep([bottom_x - top_x, top_x - bottom_x][hand.get_type() == 1], 0, 100)