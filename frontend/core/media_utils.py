import imageio.v2 as iio
from PIL import Image

def get_media_metadata(file_obj, is_video=False):
    default_video_meta = {'width': 0, 'height': 0, 'fps': 24, 'duration': 0}
    default_image_meta = {'width': 0, 'height': 0, 'fps': 24}

    if file_obj is None:
        return default_video_meta if is_video else default_image_meta

    if is_video:
        try:
            with iio.get_reader(file_obj, format='ffmpeg') as reader:
                meta = reader.get_meta_data()
                size = meta.get('size', meta.get('source_size', (0, 0)))
                width, height = size
                fps = meta.get('fps', 24)
                duration = meta.get('duration', 0)
                return {'width': width, 'height': height, 'fps': fps, 'duration': duration}
        except Exception as e:
            print(f"imageio failed for '{file_obj}': {e}. Falling back to pydub for audio duration.")
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(file_obj)
                duration = audio.duration_seconds
                return {'width': 0, 'height': 0, 'fps': 0, 'duration': duration}
            except Exception as pydub_e:
                print(f"pydub also failed for '{file_obj}': {pydub_e}")
                return default_video_meta
    else:
        if isinstance(file_obj, Image.Image):
            width, height = file_obj.size
            return {'width': width, 'height': height, 'fps': 24}
        else:
            print(f"Warning: Expected PIL Image but got {type(file_obj)}")
            return default_image_meta