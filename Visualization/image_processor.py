class image_processor:
    def __init__(self):
        pass

    def make_image(self, image, save_name=r'\Output\latest_run.png'):
        import numpy as np
        from PIL import Image
        image_array = np.array(image)
        classes = np.unique(image_array, axis=None)
        numbers = np.linspace(0, 255, len(classes))
        for idx, clas in enumerate(classes):
            image_array[image_array == clas] = numbers[idx]
        image_rgb = np.repeat(image_array[:, :, np.newaxis], 3, axis=2)
        img = Image.fromarray(image_rgb)
        img.save(save_name)
