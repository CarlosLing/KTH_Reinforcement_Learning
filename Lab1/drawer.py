from PIL import Image, ImageDraw, ImageColor

UNIT = 100

def draw_maze(walls):
    step_count = [8, 7]
    size = [UNIT*step_count[0], UNIT*step_count[1]]
    image = Image.new(mode='RGBA', size=size, color="white")

    # Draw some lines
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height
    step_size = int(image.width / step_count[0])

    for x in range(0, image.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=128)

    x_start = 0
    x_end = image.width

    for y in range(0, image.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=128)

    for point in walls:
        bounds = [UNIT*point[1], UNIT*point[0], UNIT*(1+point[1]), UNIT*(1+point[0])]
        draw.rectangle(bounds, fill="black")

    del draw
    return image

def draw_path(person_path, min_path, walls):

    person_col = "blue"
    min_col = "red"
    image = draw_maze(walls)
    draw = ImageDraw.Draw(image)

    radius = 7
    next_point = person_path[0]
    for i in range(len(person_path)-1):
        point = person_path[i]
        next_point = person_path[i+1]
        bounds = [UNIT*(0.5+point[1])-radius, UNIT*(0.5+point[0])-radius,
                  UNIT*(0.5+point[1])+radius, UNIT*(0.5+point[0])+radius]
        draw.ellipse(bounds, outline=person_col)

        line = [UNIT * (0.5 + point[1]), UNIT * (0.5 + point[0]),
                  UNIT * (0.5 + next_point[1]), UNIT * (0.5 + next_point[0])]
        draw.line(line, fill=person_col)
    point = next_point
    bounds = [UNIT * (0.5 + point[1]) - radius, UNIT * (0.5 + point[0]) - radius,
              UNIT * (0.5 + point[1]) + radius, UNIT * (0.5 + point[0]) + radius]
    draw.ellipse(bounds, fill=person_col)

    next_point = min_path[0]
    for i in range(len(min_path)-1):
        point = min_path[i]
        next_point = min_path[i+1]
        bounds = [UNIT*(0.5+point[1])-radius, UNIT*(0.5+point[0])-radius,
                  UNIT*(0.5+point[1])+radius, UNIT*(0.5+point[0])+radius]
        draw.ellipse(bounds, outline=min_col)

        line = [UNIT * (0.5 + point[1]), UNIT * (0.5 + point[0]),
                  UNIT * (0.5 + next_point[1]), UNIT * (0.5 + next_point[0])]
        draw.line(line, fill=min_col)
    point = next_point
    bounds = [UNIT * (0.5 + point[1]) - radius, UNIT * (0.5 + point[0]) - radius,
              UNIT * (0.5 + point[1]) + radius, UNIT * (0.5 + point[0]) + radius]
    draw.ellipse(bounds, fill=min_col)

    del draw
    return image


def draw_full_path(person_path, min_path, walls):
    images = []
    for i in range(1, len(person_path)+1):
        image = draw_path(person_path[0:i], min_path[0:i], walls)
        images.append(image)

    return images