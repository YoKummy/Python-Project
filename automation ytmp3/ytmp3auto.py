from pytubefix import Playlist, YouTube
import os
import moviepy.editor as mp

def download_playlist(playlist_url, download_folder):
    # Create download folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
        
    # Download each video in the playlist
    playlist = Playlist(playlist_url)
    for url in playlist:
        try:
            # Download audio stream
            audio_stream = YouTube(url).streams.filter(only_audio=True).first()
            audio_stream.download(download_folder)
            # Convert to MP3
            mp4_file = os.path.join(download_folder, audio_stream.default_filename)
            mp3_file = os.path.join(download_folder, os.path.splitext(audio_stream.default_filename)[0] + '.mp3')
            audio_clip = mp.AudioFileClip(mp4_file)
            audio_clip.write_audiofile(mp3_file)
            # Remove the original MP4 file
            os.remove(mp4_file)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            
if __name__ == "__main__":
    playlist_url = input("Enter url:")
    #download_folder = r"C:\Users\User\Desktop\song" #windows
    download_folder = "/Users/Ash/Desktop/songs" #mac
    download_playlist(playlist_url, download_folder)