import glob, os, shutil, sched, time
from threading import Timer
download = '/Users/User/Downloads/'
os.chdir(download)

event_schedule = sched.scheduler(time.time, time.sleep)

filelist = ["pdf", "png", "jpg", "pptx", "zip", "mov", "cube", "mp3", "mp4", "raw", "jar", 
            "exe", "7z", "docx", "xlsx"]

def organize_file():
    for ext in filelist:
        if not os.path.isdir(ext):
            os.mkdir(ext)

        dfiles = [f for f in os.listdir() if f.lower().endswith("." + ext)]

        for dfile in dfiles:
                src_path = os.path.join(download, dfile)
                dest_path = os.path.join(download, ext, dfile)
                shutil.move(src_path, dest_path)

def run_script():
    organize_file()
    Timer(1, run_script).start()

if __name__ == "__main__":
     run_script()

""" for file in glob.glob("*.jar"):
    print(file) """