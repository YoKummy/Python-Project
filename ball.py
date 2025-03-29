import pygame
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
screen_width = 800
screen_height = 600

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Ball properties
ball_radius = 10
ball_color = WHITE
ball_x = screen_width // 2
ball_y = screen_height // 2

# Hexagon properties
hex_side_length = 80
hex_center_x = screen_width // 2
hex_center_y = (screen_height - hex_side_length * 3 / 2) // 2 + hex_side_length
hex_rotation_angle = 0
hex_rotation_speed = 1

# Screen setup
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Rotating Hexagon with Bouncing Ball")

clock = pygame.time.Clock()

def rotate_point(point, angle):
    # Convert the angle to radians for Pygame's math functions
    rad_angle = math.radians(angle)
    x, y = point
    x_rotated = (x * math.cos(rad_angle)) - (y * math.sin(rad_angle))
    y_rotated = (x * math.sin(rad_angle)) + (y * math.cos(rad_angle))
    return x_rotated, y_rotated

def hexagon_points(center_x, center_y, side_length):
    points = []
    for i in range(6):
        angle = 2 * math.pi / 6 * i
        x = center_x + side_length * math.cos(angle)
        y = center_y - side_length * math.sin(angle)  # Inverted Y to match screen coordinates
        points.append((x, y))
    return points

def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def check_collision(ball_x, ball_y, hex_points):
    for i in range(6):
        x1, y1 = ball_x, ball_y
        x2, y2 = hex_points[(i + 1) % 6]
        
        d1 = distance((x1, y1), (hex_center_x, hex_center_y))
        d2 = distance((x2, y2), (hex_center_x, hex_center_y))
        cross_product = (y2 - y1) * (x1 - hex_center_x) - (x2 - x1) * (y1 - hex_center_y)
        
        if ((d1 < ball_radius and d2 > ball_radius) or
            (d2 < ball_radius and d1 > ball_radius)):
            t = cross_product / distance((x1, y1), (x2, y2))
            
            closest_x, closest_y = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
            
            if 0 <= t < 1 and 0.5 < d1 < 1.5:
                return True
    return False

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Clear the screen
    screen.fill(BLACK)
    
    hex_points = hexagon_points(hex_center_x, hex_center_y, hex_side_length)
    rotated_hex_points = [rotate_point(point, hex_rotation_angle) for point in hex_points]
    
    # Rotate and update ball position
    x_velocity = 10 * math.cos(math.radians(hex_rotation_angle))
    y_velocity = -10 * math.sin(math.radians(hex_rotation_angle))  # Inverted Y to match screen coordinates
    
    ball_x += x_velocity
    ball_y += y_velocity
    
    if not (screen_width / 2 - hex_side_length // 2 <= ball_x <= screen_width / 2 + hex_side_length // 2 and
            screen_height / 2 - hex_side_length * 3 / 4 <= ball_y <= screen_height / 2 + hex_side_length * 3 / 4):
        # Ball is outside the area, start over at center with new angle
        ball_x = screen_width // 2
        ball_y = screen_height // 2
    
    if check_collision(ball_x, ball_y, rotated_hex_points):
        # Ball hit a wall, reverse velocity
        x_velocity *= -1
        y_velocity *= -1

    hex_rotation_angle += hex_rotation_speed
    
    # Draw the hexagon and ball
    pygame.draw.polygon(screen, WHITE, [(x + screen_width // 2, y + screen_height // 2) for (x, y) in rotated_hex_points])
    pygame.draw.circle(screen, ball_color, (int(ball_x), int(ball_y)), ball_radius)
    
    # Update the display
    pygame.display.flip()
    
    clock.tick(60)

# Quit Pygame
pygame.quit()