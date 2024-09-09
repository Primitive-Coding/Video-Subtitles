import os
import shutil

import cv2
from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from tqdm import tqdm


import pandas as pd

from pydub import AudioSegment

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2


class VideoTranscriber:
    def __init__(self, output_folder: str) -> None:
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.source_frames_folder = f"{self.output_folder}/source_frames"
        self.output_frames_folder = f"{self.output_folder}/output_frames"
        os.makedirs(self.source_frames_folder, exist_ok=True)
        os.makedirs(self.output_frames_folder, exist_ok=True)

        self.audio_path = ""
        self.text_array = []
        self.fps = 0
        self.char_width = 0

    def transcribe_video(self, source_video_path: str, transcripts: pd.DataFrame):
        text = transcripts["text"].iloc[0]
        textsize = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        cap = cv2.VideoCapture(source_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        asp = 16 / 9
        ret, frame = cap.read()
        width = frame[
            :,
            int(int(width - 1 / asp * height) / 2) : width
            - int((width - 1 / asp * height) / 2),
        ].shape[1]
        width = width - (width * 0.1)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.char_width = int(textsize[0] / len(text))

        for i, row in tqdm(transcripts.iterrows()):
            lines = []
            text = row["text"]
            end = row["end"]
            start = row["start"]
            total_frames = int((end - start) * self.fps)
            start = start * self.fps
            total_chars = len(text)
            words = text.split(" ")
            i = 0

            while i < len(words):
                words[i] = words[i].strip()
                if words[i] == "":
                    i += 1
                    continue
                length_in_pixels = (len(words[i]) + 1) * self.char_width
                remaining_pixels = width - length_in_pixels
                line = words[i]

                while remaining_pixels > 0:
                    i += 1
                    if i >= len(words):
                        break
                    length_in_pixels = (len(words[i]) + 1) * self.char_width
                    remaining_pixels -= length_in_pixels
                    if remaining_pixels < 0:
                        continue
                    else:
                        line += " " + words[i]

                line_array = [
                    line,
                    int(start) + 15,
                    int(len(line) / total_chars * total_frames) + int(start) + 15,
                ]
                start = int(len(line) / total_chars * total_frames) + int(start)
                lines.append(line_array)
                self.text_array.append(line_array)

        cap.release()
        print(f"Text: {self.text_array}")
        print("Transcription complete")

    def extract_audio(
        self,
        source_video_path: str,
        output_audio_path: str,
    ):
        audio = AudioSegment.from_file(source_video_path, format="mp4")

        path = output_audio_path.split("/")
        extension = path[-1].split(".")[-1]

        audio.export(
            output_audio_path,
            format=extension,
        )

    def extract_frames(self, video_path: str):
        print("Extracting frames")
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        asp = width / height
        N_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = frame[
                :,
                int(int(width - 1 / asp * height) / 2) : width
                - int((width - 1 / asp * height) / 2),
            ]

            for i in self.text_array:
                if N_frames >= i[1] and N_frames <= i[2]:
                    text = i[0]
                    text_size, _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    )
                    text_x = int((frame.shape[1] - text_size[0]) / 2)
                    text_y = int(height / 2)

                    # Define the box coordinates (top-left and bottom-right points)
                    box_coords = (
                        (text_x - 10, text_y - text_size[1] - 10),  # top-left corner
                        (
                            text_x + text_size[0] + 10,
                            text_y + 10,
                        ),  # bottom-right corner
                    )

                    # Draw black rectangle as the background
                    cv2.rectangle(
                        frame,
                        box_coords[0],
                        box_coords[1],
                        (0, 0, 0),  # Black color
                        cv2.FILLED,
                    )

                    cv2.putText(
                        frame,
                        text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                    )
                    break

            cv2.imwrite(
                os.path.join(self.source_frames_folder, str(N_frames) + ".jpg"), frame
            )
            N_frames += 1

        cap.release()
        print("Frames extracted")

    def create_video(
        self,
        source_video_path: str,
        source_audio_path: str,
        output_video_path: str,
        transcripts: pd.DataFrame,
    ):
        print("Creating video")
        if not self.is_dir_filled(self.source_frames_folder):
            self.extract_frames(video_path=source_video_path)

        images = [
            img for img in os.listdir(self.source_frames_folder) if img.endswith(".jpg")
        ]
        images.sort(key=lambda x: int(x.split(".")[0]))

        frame = cv2.imread(os.path.join(self.source_frames_folder, images[0]))
        height, width, layers = frame.shape

        clip = ImageSequenceClip(
            [os.path.join(self.source_frames_folder, image) for image in images],
            fps=self.fps,
        )
        audio = AudioFileClip(source_audio_path)
        clip = clip.set_audio(audio)
        clip.write_videofile(output_video_path)
        # shutil.rmtree(image_folder)
        # os.remove(os.path.join(os.path.dirname(self.video_path), "audio.mp3"))

    def run(
        self,
        source_video_path: str,
        output_video_path: str,
        output_audio_path: str,
        transcripts: pd.DataFrame,
        recreate_frames: bool = False,
    ):
        if recreate_frames:
            self.delete_frames()

        # Step 1: Extract Audio
        if not os.path.exists(output_audio_path):
            self.extract_audio(
                source_video_path=source_video_path, output_audio_path=output_audio_path
            )

        # Step 2 Transcribe the video.
        self.transcribe_video(
            source_video_path=source_video_path, transcripts=transcripts
        )

        # Step 3 Create the video.
        self.create_video(
            source_video_path=source_video_path,
            source_audio_path=output_audio_path,
            output_video_path=output_video_path,
            transcripts=transcripts,
        )

    def is_dir_filled(self, path: str):
        dirs = os.listdir(path)
        if len(dirs) > 0:
            return True
        else:
            return False

    def delete_frames(self):

        dirs = os.listdir(self.source_frames_folder)
        for d in dirs:

            path = f"{self.source_frames_folder}/{d}"

            if os.path.exists(path):
                os.remove(path)
