import os
import yt_dlp

def download_playlist(playlist_url, download_folder):
    # Create download folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # yt-dlp options
    ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': os.path.join(download_folder, '%(title)s.%(ext)s'),
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'cookies-from-browser': 'safari',  # Change this if using Firefox or Edge
}


    # Download the playlist
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([playlist_url])

if __name__ == "__main__":
    playlist_url = "https://youtube.com/playlist?list=PLlesp14MvetUywdwtgES_49FPqjxR8QDF&si=5YTrAwprfiMq2zP9"
    download_folder = "/Users/jonathanwheeler/Desktop/songs"

    download_playlist(playlist_url, download_folder)
x