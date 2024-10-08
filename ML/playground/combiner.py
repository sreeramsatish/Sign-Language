from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip

# Paths for the video and audio files
video_path = 'output.avi'  # Path to the AVI video
audio_path = 'final_output.mp3'  # Path to the MP3 audio

# Load the video and audio
video_clip = VideoFileClip(video_path)
audio_clip = AudioFileClip(audio_path)

# Extract frames from the video
frames = [frame for frame in video_clip.iter_frames()]

# Calculate the duration each frame needs to be displayed
frame_duration = audio_clip.duration / len(frames)

# Create a new video clip from the frames
extended_video = ImageSequenceClip(frames, fps=1/frame_duration)

# Set the audio of the extended video as the original audio clip
final_clip = extended_video.set_audio(audio_clip)

# Path for the output file
output_path = 'supream.mp4'  # Saving as MP4 as it's a more compatible format

# Write the result to a file
final_clip.write_videofile(output_path, codec='libx264')
