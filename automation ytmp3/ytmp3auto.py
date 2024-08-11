import os
import yt_dlp

def download_playlist(playlist_url, download_folder):
    # Create download folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',  # Get the best audio quality
        'outtmpl': os.path.join(download_folder, '%(title)s.%(ext)s'),  # Define output file format
        'postprocessors': [{  # Post-process the audio to MP3 format
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',  # Set the audio quality to 192kbps
        }],
    }

    # Download the playlist
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([playlist_url])

if __name__ == "__main__":
    playlist_url = "https://youtube.com/playlist?list=PLlesp14MvetUywdwtgES_49FPqjxR8QDF&si=fwNIwzoA_u4RuyhK"
    download_folder = "/Users/jonathanwheeler/Desktop/songs"

    download_playlist(playlist_url, download_folder)
