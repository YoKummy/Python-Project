import glob, os, shutil, sched, time
download = '/Users/User/Downloads/'
os.chdir(download)

event_schedule = sched.scheduler(time.time, time.sleep)

filelist = ["pdf", "png", "jpg", "pptx", "zip", "mov", "cube", "mp3", "mp4", "raw", "jar", 
            "exe", "7z", "docx", "xlsx"]

for ext in filelist:
    if not os.path.isdir(ext):
        os.mkdir(ext)

    dfiles = [f for f in os.listdir(download) if f.lower().endswith("." + ext)]

    for dfile in dfiles:
            src_path = os.path.join(download, dfile)
            dest_path = os.path.join(download, ext, dfile)
            shutil.move(src_path, dest_path)

""" for file in glob.glob("*.jar"):
    print(file) """