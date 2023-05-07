import os


def downsample(data, observation_shape, downsampling_method):
    if downsampling_method == "simple":
        # print("observation_shape type: ", type(observation_shape))
        # print("observation_shape: ", observation_shape)
        obs_gap = int(1080 / observation_shape)
        processed_data = data[::obs_gap]
    else:
        processed_data = data

    return processed_data


def render_callback(env_renderer):
    # custom extra drawing function
    e = env_renderer
    # update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 700
    e.left = left - 800
    e.right = right + 800
    e.top = top + 800
    e.bottom = bottom - 800


def path_filler(path):
    abs_path = os.path.abspath(os.path.join('.', path))
    return abs_path
